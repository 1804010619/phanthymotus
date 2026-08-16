"""The review loop: an LLM with read-only tools, exploring a PR's checkout.

Modelled on agent-core's `subagent/agent.py` rather than its main `event/llm.py`
loop, for concrete reasons the main loop shows by counter-example:

- The main loop has no wall-clock timeout — 500 rounds x a 120s read timeout can
  run for hours. Here a turn is bounded in both rounds and seconds.
- The main loop calls `json.loads` on tool arguments unguarded, so one malformed
  argument blob from the model kills the whole turn. Here it degrades to `{}` and
  the tool reports the missing parameter, which the model can fix.
- The main loop breaks silently at its round ceiling. Here exhausting the budget
  is reported, because a review that stopped early and a review that found
  nothing must not look the same.

Tool failures are returned to the model as content rather than raised, so it can
correct itself instead of losing the review.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import tools as tk
from .components import ComponentContext
from .config import Config
from .reviewer import (
    chat_completions_url, describe_http_failure, explain_empty_completion,
)

logger = logging.getLogger(__name__)

# Cap on one tool result inside the transcript. Mirrors the subagent's 2500 but a
# little larger, since diffs are the main payload here.
MAX_TOOL_RESULT = 4000

# Consecutive rounds returning neither text nor tool calls before giving up. One
# is a transient; three in a row is a misconfiguration worth reporting.
MAX_EMPTY_ROUNDS = 3


@dataclass
class ReviewResult:
    markdown: str = ""
    rounds: int = 0
    stopped_reason: str = "finished"   # finished | max_rounds | timeout | error
    tool_calls: int = 0
    error: str = ""

    @property
    def complete(self) -> bool:
        return self.stopped_reason == "finished"


@dataclass
class PRFacts:
    """What the loop is told up front, instead of the whole diff."""

    repo: str
    pr_number: int
    base_ref: str
    changed_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    large_files: list[tuple[str, int]] = field(default_factory=list)
    infra_files: list[str] = field(default_factory=list)
    shared_base_files: list[str] = field(default_factory=list)


def _system_prompt(ctx: ComponentContext, facts: PRFacts) -> str:
    """Stable prefix: role, rules, and the facts about this PR.

    Kept in the system message so it forms a cacheable prefix across rounds,
    following the prefix-caching design in agent-core's prompt.py. Nothing here
    changes between rounds of the same review.
    """
    ref_lines = "\n".join(
        f"- `{p}` — {why}" for p, why in ctx.references
    ) or "- (none applicable)"
    doc_lines = "\n".join(f"- `{d}`" for d in ctx.docs) or "- (none)"

    large = "\n".join(
        f"- `{p}` — {s / 1024:.0f}KB" for p, s in facts.large_files
    ) or "- none"
    infra = "\n".join(f"- `{p}`" for p in facts.infra_files) or "- none"
    shared = "\n".join(f"- `{p}`" for p in facts.shared_base_files)

    files = "\n".join(f"- `{f}`" for f in facts.changed_files[:80])
    if len(facts.changed_files) > 80:
        files += f"\n- … and {len(facts.changed_files) - 80} more"

    shared_block = ""
    if shared:
        shared_block = (
            "\n**This PR changes a shared base image or shared code.** Changes "
            "here affect every component that builds on it, across both "
            "repositories:\n" + shared + "\n"
        )

    return f"""\
You are reviewing a pull request for an embodied-AI platform. You have read-only
tools over the PR's checkout and a limited number of rounds, so spend them on
reading what you actually need to judge the change.

Component under review: **{ctx.name}**

# Rules

{ctx.rules}

# This pull request

Repository: `{facts.repo}`, PR #{facts.pr_number}, merged onto `{facts.base_ref}`.

## Changed files

{files}

## Diff summary

```
{facts.diff_stat.strip()[:3000]}
```

## Files over the size limit (from a deterministic check)

{large}

## Infrastructure files touched (from a deterministic check)

{infra}
{shared_block}
# How to work

1. Read the authoritative docs for this component first:
{doc_lines}
2. Read the files this PR changes — use `file_diff(path)` for what changed and
   `read_file(path)` when you need the surrounding code to judge it.
3. Compare against an existing implementation of the same kind:
{ref_lines}
4. Then call `finish_review` exactly once.

The size and infrastructure lists above are already computed — do not re-derive
them, but do explain in your review whether each infrastructure change is
necessary and whether it grows the image.
"""


def _sanitize(messages: list[dict]) -> list[dict]:
    """Drop a trailing assistant message whose tool_calls were never answered.

    The API rejects the whole request if any `tool_call_id` is unanswered, so any
    path that can truncate the transcript — timeout, exception mid-dispatch —
    must run this first. agent-core learned this the same way.
    """
    if not messages:
        return messages
    out = list(messages)
    while out and out[-1].get("role") == "assistant" and out[-1].get("tool_calls"):
        answered = {
            m.get("tool_call_id") for m in out if m.get("role") == "tool"
        }
        wanted = {c.get("id") for c in out[-1]["tool_calls"]}
        if wanted <= answered:
            break
        out.pop()
    return out


class ReviewAgent:
    """One review: bounded rounds of tool use, ending in a written review."""

    def __init__(
        self,
        config: Config,
        worktree: Path,
        ctx: ComponentContext,
        facts: PRFacts,
    ):
        self._cfg = config
        self._ctx = ctx
        self._facts = facts
        self._sb = tk.Sandbox(worktree, base_ref=facts.base_ref)
        self._endpoint = chat_completions_url(config.llm_base_url)
        self._deadline = 0.0

    async def run(self) -> ReviewResult:
        if not self._cfg.llm_base_url or not self._cfg.llm_api_key:
            return ReviewResult(
                markdown="_LLM review skipped (not configured)_",
                stopped_reason="error",
                error="llm not configured",
            )

        self._deadline = time.monotonic() + self._cfg.review_timeout_seconds
        messages = [
            {"role": "system", "content": _system_prompt(self._ctx, self._facts)},
            {"role": "user", "content":
                "Review this pull request. Read the docs and the changed files "
                "first, then call finish_review."},
        ]
        result = ReviewResult()
        empties = 0

        async with httpx.AsyncClient(timeout=self._cfg.llm_timeout_seconds) as client:
            for rnd in range(1, self._cfg.review_max_rounds + 1):
                result.rounds = rnd

                if time.monotonic() > self._deadline:
                    result.stopped_reason = "timeout"
                    break

                try:
                    assistant = await self._call(client, messages)
                except Exception as e:
                    # A failure mid-loop still yields whatever was learned; the
                    # partial review is more useful than nothing.
                    logger.warning(f"review round {rnd} failed: {e}")
                    result.stopped_reason = "error"
                    result.error = f"{type(e).__name__}: {e}"
                    break

                messages.append(assistant)
                calls = assistant.get("tool_calls") or []

                if not calls:
                    prose = (assistant.get("content") or "").strip()
                    if not prose:
                        # Neither tools nor text. Accepting this posts an empty
                        # "Code Review" comment, which reads as "no comments"
                        # rather than "the call failed", so nudge instead of
                        # finishing — but bounded, because a too-small
                        # max_tokens produces this every round and spinning
                        # through the whole budget hides the real cause.
                        empties += 1
                        why = assistant.pop("_empty_reason", "")
                        logger.warning(
                            f"round {rnd} returned nothing ({why}); "
                            f"empty {empties}/{MAX_EMPTY_ROUNDS}"
                        )
                        if empties >= MAX_EMPTY_ROUNDS:
                            result.stopped_reason = "error"
                            result.error = why or "the model returned nothing"
                            break
                        messages.pop()   # keep the empty turn out of history
                        messages.append({
                            "role": "user",
                            "content": "You returned no text and called no "
                                       "tools. Continue: use a tool, or call "
                                       "finish_review with your findings.",
                        })
                        continue
                    # No tools requested — treat the prose as the review.
                    result.markdown = prose
                    result.stopped_reason = "finished"
                    break

                finished = await self._dispatch_all(calls, messages, result)
                if finished is not None:
                    result.markdown = finished
                    result.stopped_reason = "finished"
                    break
            else:
                result.stopped_reason = "max_rounds"

        if not result.markdown:
            result.markdown = self._salvage(messages, result)
        return result

    async def _call(self, client: httpx.AsyncClient, messages: list[dict]) -> dict:
        resp = await client.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._cfg.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._cfg.llm_model,
                "messages": _sanitize(messages),
                "tools": tk.SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": self._cfg.llm_max_tokens,
            },
        )
        # Shares the diagnosable failure messages with the single-call path: a
        # gateway answering 200 with its web UI otherwise surfaces as
        # "Expecting value: line 1 column 1", hiding the cause.
        data = describe_http_failure(resp, self._endpoint)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"unexpected response shape from {self._endpoint} ({e}): "
                f"{resp.text[:200]}"
            ) from e

        # The SDK-less path needs the same defensive fixups agent-core applies:
        # some models emit tool_calls: null, or entries missing `function`.
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if not out["content"] and not msg.get("tool_calls"):
            # Stashed rather than raised: one empty turn is recoverable, and the
            # reason only matters if it keeps happening.
            out["_empty_reason"] = explain_empty_completion(data)
        calls = msg.get("tool_calls")
        if isinstance(calls, list):
            valid = [
                c for c in calls
                if isinstance(c, dict) and isinstance(c.get("function"), dict)
                and c["function"].get("name")
            ]
            if valid:
                out["tool_calls"] = valid
        return out

    async def _dispatch_all(
        self, calls: list[dict], messages: list[dict], result: ReviewResult
    ) -> str | None:
        """Run each requested tool. Returns the review if finish was called."""
        finished = None
        for call in calls:
            name = call["function"]["name"]
            raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except (ValueError, TypeError):
                # Degrade rather than kill the review: the tool will report the
                # missing parameter and the model can retry.
                args = {}
            if not isinstance(args, dict):
                args = {}

            result.tool_calls += 1

            if name == tk.FINISH_TOOL:
                finished = _format_review(args)
                content = "review recorded"
            else:
                content = await self._run_tool(name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": content[:MAX_TOOL_RESULT],
            })
        return finished

    async def _run_tool(self, name: str, args: dict) -> str:
        fn = tk.DISPATCH.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            # Tools are sync and filesystem-bound; off the loop so a large grep
            # cannot stall the poller or a build's log pump.
            return await asyncio.to_thread(fn, self._sb, **args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:
            logger.warning(f"tool {name} failed: {e}")
            return f"[tool error] {type(e).__name__}: {e}"

    def _salvage(self, messages: list[dict], result: ReviewResult) -> str:
        """Recover something useful when the loop ended without finish_review."""
        prose = [
            (m.get("content") or "").strip()
            for m in messages
            if m.get("role") == "assistant" and (m.get("content") or "").strip()
        ]
        if prose:
            return prose[-1]
        return (
            "_The reviewer explored the change but did not produce a written "
            "review before its budget ran out._"
        )


def _format_review(args: dict) -> str:
    """Render finish_review arguments as the markdown posted to the PR."""
    summary = (args.get("summary") or "").strip()
    issues = (args.get("issues") or "").strip() or "No issues found."
    suggestions = (args.get("suggestions") or "").strip() or "No suggestions."
    parts = []
    if summary:
        parts.append(f"### Summary\n{summary}")
    parts.append(f"### Issues\n{issues}")
    parts.append(f"### Suggestions\n{suggestions}")
    return "\n\n".join(parts)
