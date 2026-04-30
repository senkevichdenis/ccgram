"""Response message building for Telegram delivery.

Builds paginated response messages from Claude Code output:
  - Handles different content types (text, thinking, tool_use, tool_result)
  - Returns raw markdown strings (entity conversion happens at send time)
  - Splits long messages into pages within Telegram's 4096 char limit
  - Truncates thinking content to keep messages compact

Key function:
  - build_response_parts: Build paginated response messages
"""

from ..providers.base import EXPANDABLE_QUOTE_END, EXPANDABLE_QUOTE_START
from ..telegram_sender import split_message
from ..sanitizer import sanitize  # BRAIN FORK: outbound secret filtering
from ..config import config  # BRAIN FORK: for allowed_users count

# Max length for user messages before truncation
_MAX_USER_MSG_LENGTH = 3000


def build_response_parts(
    text: str,
    is_complete: bool,
    content_type: str = "text",
    role: str = "assistant",
) -> list[str]:
    """Build paginated response messages for Telegram.

    Returns a list of raw markdown strings.
    Entity conversion happens at send time in the message sender layer.
    Multi-part messages get a [1/N] suffix.
    """
    text = text.strip()

    # BRAIN FORK: sanitize secrets + drop harness leaks before Telegram delivery.
    # sanitize() returns "" when the entire message is a harness phrase
    # (e.g. assistant emitting "No response requested." with no real content).
    text = sanitize(text, len(config.allowed_users))
    if not text.strip():
        return []

    # BRAIN FORK: hide hook feedback from chat (internal instructions, not for user)
    if "hook feedback" in text.lower() or "stop hook" in text.lower() or "session hook" in text.lower():
        return []

    # BRAIN FORK: hide tool_result from chat (internal details, not for user)
    if content_type == "tool_result":
        return []

    # User messages: add emoji prefix (no newline)
    if role == "user":
        # BRAIN FORK: hide upload notifications and reply context echo from chat
        if "I've uploaded" in text or "I have uploaded" in text or "[Replying to" in text:
            return []
        prefix = "\U0001f464 "
        if len(text) > _MAX_USER_MSG_LENGTH:
            text = text[:_MAX_USER_MSG_LENGTH] + "\u2026"
        return [f"{prefix}{text}"]

    # Truncate thinking content to keep it compact
    if content_type == "thinking" and is_complete:
        start_tag = EXPANDABLE_QUOTE_START
        end_tag = EXPANDABLE_QUOTE_END
        max_thinking = 500
        if start_tag in text and end_tag in text:
            inner = text[text.index(start_tag) + len(start_tag) : text.index(end_tag)]
            if len(inner) > max_thinking:
                inner = inner[:max_thinking] + "\n\n\u2026 (thinking truncated)"
            text = start_tag + inner + end_tag
        elif len(text) > max_thinking:
            text = text[:max_thinking] + "\n\n\u2026 (thinking truncated)"

    # BRAIN FORK: thinking shows as temp status message (no emoji, disappears)
    if content_type == "thinking":
        return ["__STATUS__Thinking..."]
    else:
        # Plain text: no prefix
        prefix = ""
        separator = ""

    # If text contains expandable quote sentinels, don't split —
    # the quote must stay atomic. Truncation is handled by
    # _truncate_quote_text in entity_formatting.py.
    if EXPANDABLE_QUOTE_START in text:
        if prefix:
            return [f"{prefix}{separator}{text}"]
        else:
            return [text]

    # Split raw markdown text, then each chunk is sent individually.
    # Entity conversion happens at send time.
    max_text = 3000 - len(prefix) - len(separator)

    text_chunks = split_message(text, max_length=max_text)
    total = len(text_chunks)

    if total == 1:
        if prefix:
            return [f"{prefix}{separator}{text_chunks[0]}"]
        else:
            return [text_chunks[0]]

    parts = []
    for i, chunk in enumerate(text_chunks, 1):
        if prefix:
            parts.append(f"{prefix}{separator}{chunk}\n\n[{i}/{total}]")
        else:
            parts.append(f"{chunk}\n\n[{i}/{total}]")
    return parts
