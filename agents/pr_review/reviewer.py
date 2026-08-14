"""Reviewer — rule-based checks + LLM-powered code review."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    severity: str  # "error", "warning", "info"
    file: str
    message: str


# ── Rule-based checks ──────────────────────────────────────────────────────────


def run_rule_checks(changed_files: list[str], diff_stat: str) -> list[Finding]:
    """Run fast, deterministic rule checks on the changeset."""
    findings = []
    findings.extend(_check_dockerfile_changes(changed_files))
    findings.extend(_check_large_files(diff_stat))
    findings.extend(_check_sensitive_files(changed_files))
    return findings


def _check_dockerfile_changes(changed_files: list[str]) -> list[Finding]:
    """Warn if Dockerfiles are modified (minimal Dockerfile change principle)."""
    findings = []
    for f in changed_files:
        name = Path(f).name
        if name == "Dockerfile" or name.startswith("Dockerfile."):
            findings.append(Finding(
                severity="warning",
                file=f,
                message=(
                    "Dockerfile modified — confirm this is necessary "
                    "(minimal Dockerfile change principle)"
                ),
            ))
    return findings


def _check_large_files(diff_stat: str) -> list[Finding]:
    """Detect newly added large files (>1MB) from diff stat output."""
    findings = []
    # Binary entries look like: " path/to/file | Bin 0 -> 1234567 bytes"
    bin_pattern = re.compile(r"^\s*(.+?)\s*\|\s*Bin\s+\d+\s*->\s*(\d+)\s+bytes")
    for line in diff_stat.splitlines():
        m = bin_pattern.match(line)
        if m:
            filepath = m.group(1).strip()
            size_bytes = int(m.group(2))
            if size_bytes > 1_000_000:
                size_mb = size_bytes / 1_000_000
                findings.append(Finding(
                    severity="error",
                    file=filepath,
                    message=(
                        f"Large file ({size_mb:.1f}MB) — files over 1MB need "
                        "a strong justification"
                    ),
                ))
    return findings


def _check_sensitive_files(changed_files: list[str]) -> list[Finding]:
    """Check for potentially sensitive files being committed."""
    sensitive_patterns = {".env", "credentials", "secret", ".pem", ".key"}
    findings = []
    for f in changed_files:
        name_lower = Path(f).name.lower()
        # .env.example is a template, not a secret.
        if name_lower.endswith(".example") or name_lower.endswith(".sample"):
            continue
        for pattern in sensitive_patterns:
            if pattern in name_lower:
                findings.append(Finding(
                    severity="error",
                    file=f,
                    message=(
                        f"File may contain secrets (matched: {pattern}) — "
                        "verify nothing sensitive is committed"
                    ),
                ))
                break
    return findings


# ── LLM Review ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code reviewer for an embodied AI platform. The project has two repos:
- phanthymotus: Agent Core (FastAPI + LLM + ROS2) and Perception (ASR/TTS)
- phanthymotus-driver: Hardware drivers for robots/drones (Unitree, DJI, etc.)

Architecture: three layers — Driver (MCP HTTP) -> Perception (MCP) -> Agent Core
(FastAPI + LLM). All control-plane communication is MCP JSON-RPC 2.0 over HTTP.
Driver `dispatch()` must return a plain dict; the MCP handler wraps it, so
returning a pre-wrapped `[{"type": "text", ...}]` array double-encodes and
breaks the frontend.

Review priorities, in order:
1. Correctness — bugs, race conditions, unhandled errors, incorrect logic
2. Security — injected secrets, unvalidated input, unsafe subprocess/shell use
3. Architecture — do not break the Agent Core / Perception / Driver separation
4. Minimal Dockerfile changes — flag Dockerfile edits that are not necessary
5. No large files (>1MB) without strong justification
6. Code quality — naming, dead code, duplicated logic

Ground every point in the diff. Cite `file:line` where you can. Do not
speculate about code you cannot see, and do not restate what the diff does as
if it were a finding. If the PR looks fine, say so plainly rather than
manufacturing issues.

Write the review in English. Output exactly this markdown structure:

### Summary
[1-2 sentences on what this PR does]

### Issues
[Bulleted list, most severe first, each with a `file:line` reference and the
concrete consequence. Write "No issues found." if there are none.]

### Suggestions
[Optional improvements, clearly non-blocking. Write "No suggestions." if none.]
"""


async def llm_review(
    config: Config,
    changed_files: list[str],
    diff: str,
    rule_findings: list[Finding],
) -> str:
    """Run LLM-powered code review via OpenAI-compatible API.

    Returns markdown review text, or error message on failure.
    """
    if not config.llm_base_url or not config.llm_api_key:
        return "_LLM review skipped (not configured)_"

    # Build user prompt
    findings_text = ""
    if rule_findings:
        findings_text = "\n\nRule check findings (already detected):\n"
        for f in rule_findings:
            findings_text += f"- [{f.severity}] {f.file}: {f.message}\n"

    user_prompt = f"""\
PR changes {len(changed_files)} files:
{chr(10).join('- ' + f for f in changed_files[:50])}
{f'... and {len(changed_files) - 50} more files' if len(changed_files) > 50 else ''}
{findings_text}

Diff:
```
{diff}
```
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{config.llm_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.llm_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM review failed: {e}")
        return f"_LLM review failed: {e}_"
