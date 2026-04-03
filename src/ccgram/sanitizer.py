"""Outbound sanitization: filter secrets from agent responses.

Regex-based detection of API keys, tokens, and credentials.
Replaces matches with [REDACTED].

Rule: sanitize only when group has more than one person
(ALLOWED_USERS count > 1). Owner-only groups are not filtered.
"""

import re

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
    """Sanitize text by replacing detected secrets with [REDACTED].

    Args:
        text: The text to sanitize.
        allowed_users_count: Number of users in ALLOWED_USERS.
            If 1 (owner only), no filtering is applied.

    Returns:
        Sanitized text.
    """
    if allowed_users_count <= 1:
        return text

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text
