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
        if Path(f).name == "Dockerfile" or Path(f).name.startswith("Dockerfile."):
            findings.append(Finding(
                severity="warning",
                file=f,
                message="Dockerfile 被修改 — 请确认是否必要（最小 Dockerfile 改动原则）",
            ))
    return findings


def _check_large_files(diff_stat: str) -> list[Finding]:
    """Detect newly added large files (>1MB) from diff stat output."""
    findings = []
    # git diff --stat lines look like: " path/to/file | 1234 +++"
    # For binary files: " path/to/file | Bin 0 -> 1234567 bytes"
    bin_pattern = re.compile(r"^\s*(.+?)\s*\|\s*Bin\s+\d+\s*->\s*(\d+)\s+bytes")
    for line in diff_stat.splitlines():
        m = bin_pattern.match(line)
        if m:
            filepath = m.group(1).strip()
            size_bytes = int(m.group(2))
            if size_bytes > 1_000_000:  # 1MB
                size_mb = size_bytes / 1_000_000
                findings.append(Finding(
                    severity="error",
                    file=filepath,
                    message=f"大文件 ({size_mb:.1f}MB) — >1MB 的文件必须有充分理由",
                ))
    return findings


def _check_sensitive_files(changed_files: list[str]) -> list[Finding]:
    """Check for potentially sensitive files being committed."""
    sensitive_patterns = {".env", "credentials", "secret", ".pem", ".key"}
    findings = []
    for f in changed_files:
        name_lower = Path(f).name.lower()
        for pattern in sensitive_patterns:
            if pattern in name_lower:
                findings.append(Finding(
                    severity="error",
                    file=f,
                    message=f"可能包含敏感信息的文件被修改 (匹配: {pattern})",
                ))
                break
    return findings


# ── LLM Review ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code reviewer for an embodied AI platform. The project has two repos:
- phanthymotus: Agent Core (FastAPI + LLM + ROS2) and Perception (ASR/TTS)
- phanthymotus-driver: Hardware drivers for robots/drones (Unitree, DJI, etc.)

Architecture: 3 layers — Driver (MCP HTTP) → Perception (MCP) → Agent Core (FastAPI + LLM).
All communication uses MCP JSON-RPC 2.0 over HTTP. Drivers implement `dispatch()` returning plain dicts.

Review guidelines:
1. Minimal Dockerfile changes — only modify when truly necessary
2. No large files (>1MB) without strong justification
3. Do not break the motus/driver architecture separation
4. Check for security issues, bugs, and code quality
5. Be concise and actionable — developers need specific feedback

Output your review in this markdown format:
### Summary
[1-2 sentence summary of the PR]

### Issues
[Bulleted list of issues found, with file:line references where possible]
[If no issues, write "No issues found."]

### Suggestions
[Bulleted list of optional improvements]
[If none, write "No suggestions."]

Review in the language matching the code comments (Chinese if comments are Chinese, English otherwise).
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
