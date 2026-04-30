"""Outbound sanitization: filter secrets and harness leaks from agent responses.

Two filter layers:
1. Secrets (API keys, tokens, credentials) -> [REDACTED]. Group-gated
   (only active when ALLOWED_USERS > 1; owner-only groups skip secret filter).
2. Harness leaks (e.g. assistant generating "No response requested." as a full
   reply when there is no real user task) -> empty string. Always active,
   even for owner: it is internal noise, not a secret. Returning "" tells
   the caller (response_builder) the message has no user-visible content.
"""

import re

# Harness-leak patterns: match the ENTIRE message body (after .strip()).
# These are full-text replies the assistant emits in idle/empty-prompt edge
# cases. Detected from real ccgram transcripts. Drop the whole message.
# IMPORTANT: callers must check for empty string after sanitize() and skip
# delivery, otherwise an empty Telegram message will be attempted.
_HARNESS_FULL_MATCH = [
    re.compile(r"^no response requested\.?$", re.IGNORECASE),
    re.compile(r"^continue from where you left off\.?$", re.IGNORECASE),
    re.compile(r"^continuing(?:\s+from\s+where\s+i\s+left\s+off)?\.{0,3}$", re.IGNORECASE),
    re.compile(r"^session\s+(?:was\s+)?paused\.?$", re.IGNORECASE),
    re.compile(r"^session\s+resumed\.?$", re.IGNORECASE),
]


# Prefix-based patterns (low false positive rate)
_PATTERNS = [
    # AWS
    (re.compile(r"(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}"), "[REDACTED:aws]"),
    # OpenAI / Anthropic
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED:anthropic]"),
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"), "[REDACTED:openai]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,60}"), "[REDACTED:api_key]"),
    # Stripe
    (re.compile(r"[sr]k_(live|test)_[A-Za-z0-9]{24,}"), "[REDACTED:stripe]"),
    # GitHub
    (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "[REDACTED:github]"),
    (re.compile(r"gho_[0-9a-zA-Z]{36}"), "[REDACTED:github]"),
    (re.compile(r"github_pat_[0-9a-zA-Z_]{82}"), "[REDACTED:github]"),
    # Supabase
    (re.compile(r"sbp_[a-f0-9]{20,}"), "[REDACTED:supabase]"),
    (re.compile(r"sb_secret_[A-Za-z0-9_\-]{10,}"), "[REDACTED:supabase]"),
    # Slack
    (re.compile(r"xox[pboa]-[0-9]{10,13}-[0-9a-zA-Z\-]{10,}"), "[REDACTED:slack]"),
    # Telegram bot token
    (re.compile(r"[0-9]{8,10}:[0-9A-Za-z_\-]{35}"), "[REDACTED:telegram]"),
    # Bearer token
    (re.compile(r"Bearer [A-Za-z0-9_\-\.]{20,}"), "[REDACTED:bearer]"),
    # Private key
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "[REDACTED:private_key]"),
    # Generic key=value (last, catches remaining secrets by context)
    (
        re.compile(
            r"(?i)(api[_-]?key|api[_-]?secret|secret[_-]?key|service[_-]?key"
            r"|password|passwd|private[_-]?key|access[_-]?token|refresh[_-]?token)"
            r"\s*[=:]\s*['\"]?([^\s'\"]{20,})"
        ),
        r"\1=[REDACTED]",
    ),
]


def sanitize(text: str, allowed_users_count: int = 1) -> str:
    """Sanitize outbound text: drop harness-leak phrases, redact secrets.

    Harness filter runs ALWAYS (even owner-only groups); secret filter is
    group-gated.

    Args:
        text: The text to sanitize.
        allowed_users_count: Number of users in ALLOWED_USERS.
            If 1 (owner only), secret-redaction skipped, harness filter still runs.

    Returns:
        Sanitized text. Empty string means the entire message was a harness
        leak and should NOT be delivered. Callers must check for "" and skip.
    """
    # Harness-leak full-match drop (run first; cheap; always active).
    stripped = text.strip()
    for pattern in _HARNESS_FULL_MATCH:
        if pattern.match(stripped):
            return ""

    if allowed_users_count <= 1:
        return text

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text
