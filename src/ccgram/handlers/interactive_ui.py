"""Interactive UI handling for Claude Code prompts.

Handles interactive terminal UIs displayed by Claude Code:
  - AskUserQuestion: Multi-choice question prompts
  - ExitPlanMode: Plan mode exit confirmation
  - Permission Prompt: Tool permission requests
  - RestoreCheckpoint: Checkpoint restoration selection

Provides:
  - Keyboard navigation (up/down/left/right/enter/esc)
  - Terminal capture and display
  - Interactive mode tracking per user and thread

State dicts are keyed by (user_id, thread_id_or_0) for Telegram topic support.
"""

import contextlib
import time

import structlog

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.error import BadRequest, RetryAfter, TelegramError

from ..providers import get_provider_for_window
from ..session import session_manager
from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_ASK_DOWN,
    CB_ASK_ENTER,
    CB_ASK_ESC,
    CB_ASK_LEFT,
    CB_ASK_REFRESH,
    CB_ASK_RIGHT,
    CB_ASK_SPACE,
    CB_ASK_TAB,
    CB_ASK_UP,
)
from .message_sender import NO_LINK_PREVIEW, is_thread_gone, rate_limit_send

logger = structlog.get_logger()

# Tool names that trigger interactive UI via JSONL (terminal capture + inline keyboard)
INTERACTIVE_TOOL_NAMES = frozenset(
    {
        "AskUserQuestion",
        "ExitPlanMode",
        # Codex native tool name before normalization/fallback.
        "request_user_input",
    }
)

# Track interactive UI message IDs: (user_id, thread_id_or_0) -> message_id
_interactive_msgs: dict[tuple[int, int], int] = {}

# Track interactive mode: (user_id, thread_id_or_0) -> window_id
_interactive_mode: dict[tuple[int, int], str] = {}

# Cooldown to prevent flood when interactive sends fail repeatedly
_send_cooldowns: dict[tuple[int, int], float] = {}
_send_fail_counts: dict[tuple[int, int], int] = {}  # BRAIN FORK: retry limit for interactive UI
_MAX_INTERACTIVE_RETRIES = 5  # auto-accept after 5 failed send attempts
_SEND_RETRY_INTERVAL = 5.0  # seconds between retries for failed sends
_DEAD_TOPIC_RETRY_INTERVAL = 60.0  # longer backoff when topic is deleted


def get_interactive_window(user_id: int, thread_id: int | None = None) -> str | None:
    """Get the window_id for user's interactive mode."""
    return _interactive_mode.get((user_id, thread_id or 0))


def set_interactive_mode(
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
) -> None:
    """Set interactive mode for a user."""
    logger.debug(
        "Set interactive mode: user=%d, window_id=%s, thread=%s",
        user_id,
        window_id,
        thread_id,
    )
    _interactive_mode[(user_id, thread_id or 0)] = window_id


def clear_interactive_mode(user_id: int, thread_id: int | None = None) -> None:
    """Clear interactive mode for a user (without deleting message)."""
    logger.debug("Clear interactive mode: user=%d, thread=%s", user_id, thread_id)
    _interactive_mode.pop((user_id, thread_id or 0), None)


def get_interactive_msg_id(user_id: int, thread_id: int | None = None) -> int | None:
    """Get the interactive message ID for a user."""
    return _interactive_msgs.get((user_id, thread_id or 0))


def _build_interactive_keyboard(
    window_id: str,
    ui_name: str = "",
    pane_id: str | None = None,
) -> InlineKeyboardMarkup:
    """Build keyboard for interactive UI navigation.

    ``ui_name`` controls the layout: ``RestoreCheckpoint`` omits ←/→ keys
    since only vertical selection is needed.

    When ``pane_id`` is set, it is appended to each callback data so
    responses route to a specific pane instead of the window's active pane.
    """
    vertical_only = ui_name == "RestoreCheckpoint"
    # Target suffix: "@12" or "@12:%5" when pane-targeted
    target = f"{window_id}:{pane_id}" if pane_id else window_id

    rows: list[list[InlineKeyboardButton]] = []
    # Row 1: directional keys
    rows.append(
        [
            InlineKeyboardButton(
                "␣ Space", callback_data=f"{CB_ASK_SPACE}{target}"[:64]
            ),
            InlineKeyboardButton("↑", callback_data=f"{CB_ASK_UP}{target}"[:64]),
            InlineKeyboardButton("⇥ Tab", callback_data=f"{CB_ASK_TAB}{target}"[:64]),
        ]
    )
    if vertical_only:
        rows.append(
            [
                InlineKeyboardButton("↓", callback_data=f"{CB_ASK_DOWN}{target}"[:64]),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton("←", callback_data=f"{CB_ASK_LEFT}{target}"[:64]),
                InlineKeyboardButton("↓", callback_data=f"{CB_ASK_DOWN}{target}"[:64]),
                InlineKeyboardButton("→", callback_data=f"{CB_ASK_RIGHT}{target}"[:64]),
            ]
        )
    # Row 2: action keys
    rows.append(
        [
            InlineKeyboardButton("⎋ Esc", callback_data=f"{CB_ASK_ESC}{target}"[:64]),
            InlineKeyboardButton("🔄", callback_data=f"{CB_ASK_REFRESH}{target}"[:64]),
            InlineKeyboardButton(
                "⏎ Enter", callback_data=f"{CB_ASK_ENTER}{target}"[:64]
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


async def _edit_interactive_msg(
    bot: Bot,
    chat_id: int,
    msg_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup,
    ikey: tuple[int, int],
    window_id: str,
) -> bool | None:
    """Try to edit an existing interactive message.

    Returns True/False on success/failure, or None if no edit was attempted.
    """
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=text,
            reply_markup=keyboard,
            link_preview_options=NO_LINK_PREVIEW,
        )
        _interactive_mode[ikey] = window_id
        return True
    except BadRequest as e:
        if "Message is not modified" in e.message:
            return True  # Content identical, no-op
        logger.warning("BadRequest editing interactive msg: %s", e.message)
        return False
    except RetryAfter:
        raise
    except TelegramError:
        logger.warning("Failed to edit interactive message", exc_info=True)
        return False


async def _capture_interactive_content(
    window_id: str,
    pane_id: str | None = None,
) -> tuple[str, str] | None:
    """Capture pane and extract interactive UI content.

    When *pane_id* is given, captures that specific pane (by stable ``%N`` ID)
    instead of the window's active pane.

    Returns (ui_name, text) if an interactive UI is detected, None otherwise.
    """
    if pane_id:
        pane_text = await tmux_manager.capture_pane_by_id(pane_id, window_id=window_id)
    else:
        w = await tmux_manager.find_window_by_id(window_id)
        if not w:
            return None
        pane_text = await tmux_manager.capture_pane(w.window_id)

    if not pane_text:
        logger.debug(
            "No pane text captured for window_id %s pane_id %s", window_id, pane_id
        )
        return None

    provider = get_provider_for_window(window_id)
    pane_title = ""
    if provider.capabilities.uses_pane_title and not pane_id:
        pane_title = await tmux_manager.get_pane_title(window_id)
    status = provider.parse_terminal_status(pane_text, pane_title=pane_title)
    if status is None or not status.is_interactive:
        logger.debug(
            "No interactive UI detected in window_id %s pane %s (last 3 lines: %s)",
            window_id,
            pane_id,
            pane_text.strip().split("\n")[-3:],
        )
        return None

    if not status.ui_type:
        logger.warning(
            "Interactive status with no ui_type in window_id %s pane %s",
            window_id,
            pane_id,
        )
        return None

    return status.ui_type, status.raw_text


import re as _re

# BRAIN FORK: callback prefix for clean option buttons
CB_OPTION = "opt:"  # opt:{window_id}:{option_index}


def _build_clean_ui(
    raw_text: str, window_id: str, ui_name: str
) -> tuple[str, InlineKeyboardMarkup]:
    """BRAIN FORK: parse interactive UI text, build clean message + option buttons.

    Two formats:
    - Numbered options (1. Yes / 2. No): clean buttons, one per option
    - Checkbox options (☐/✔/☒): navigation keyboard (Space to toggle, Enter to confirm)
    """
    lines = raw_text.strip().split("\n")

    # Detect checkbox format
    has_checkboxes = any("\u2610" in l or "\u2714" in l or "\u2612" in l for l in lines)

    if has_checkboxes:
        # Checkbox multi-select: show full text + navigation keyboard
        # Filter out hint lines
        clean_lines = [l.strip() for l in lines
                       if l.strip() and "Esc to cancel" not in l and "Tab to amend" not in l
                       and "ctrl+e" not in l and "Enter to select" not in l
                       and "ctrl-g" not in l]
        text = "\n".join(clean_lines) if clean_lines else raw_text
        # Navigation: Space (toggle), Up, Down, Enter (confirm), Esc (cancel)
        rows = [
            [
                InlineKeyboardButton("Toggle", callback_data=f"{CB_ASK_SPACE}{window_id}"[:64]),
                InlineKeyboardButton("\u2191", callback_data=f"{CB_ASK_UP}{window_id}"[:64]),
                InlineKeyboardButton("\u2193", callback_data=f"{CB_ASK_DOWN}{window_id}"[:64]),
            ],
            [
                InlineKeyboardButton("Confirm", callback_data=f"{CB_ASK_ENTER}{window_id}"[:64]),
                InlineKeyboardButton("Cancel", callback_data=f"{CB_ASK_ESC}{window_id}"[:64]),
            ],
        ]
        return text, InlineKeyboardMarkup(rows)

    # Numbered options: parse and build clean buttons
    options = []
    all_pre_option_lines = []
    for line in lines:
        stripped = line.strip().lstrip("\u276f ").lstrip("\u203a ")
        match = _re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if match:
            options.append((int(match.group(1)), match.group(2).strip()))
        elif "Esc to cancel" in line or "Tab to amend" in line or "ctrl+e" in line:
            continue
        elif not options:
            all_pre_option_lines.append(line.strip())

    # Question: last non-empty line before options
    question = ui_name
    for line in reversed(all_pre_option_lines):
        if line:
            question = line
            break

    # Buttons: one per option, no numbering
    rows = []
    for idx, (num, label) in enumerate(options):
        short_label = label if len(label) <= 45 else label[:42] + "..."
        cb_data = f"{CB_OPTION}{window_id}:{idx}"[:64]
        rows.append([InlineKeyboardButton(short_label, callback_data=cb_data)])

    if not rows:
        # Fallback: no numbered options found, show Enter/Esc
        rows = [
            [
                InlineKeyboardButton("Yes", callback_data=f"{CB_ASK_ENTER}{window_id}"[:64]),
                InlineKeyboardButton("No", callback_data=f"{CB_ASK_ESC}{window_id}"[:64]),
            ],
        ]
    else:
        rows.append([InlineKeyboardButton("Cancel", callback_data=f"{CB_ASK_ESC}{window_id}"[:64])])

    return question, InlineKeyboardMarkup(rows)


async def handle_interactive_ui(
    bot: Bot,
    user_id: int,
    window_id: str,
    thread_id: int | None = None,
    pane_id: str | None = None,
) -> bool:
    """Capture terminal and send interactive UI content to user.

    Handles AskUserQuestion, ExitPlanMode, Permission Prompt, and
    RestoreCheckpoint UIs. Returns True if UI was detected and sent,
    False otherwise.

    When *pane_id* is given, captures and targets a specific pane (for
    multi-pane windows such as agent teams).  The pane context is shown
    in the message and the keyboard routes responses to that pane.
    """
    # BRAIN FORK: selective interactive UI (patch 7 revised)
    # Settings/SelectModel: block (system UI, not for Telegram)
    # PermissionPrompt/RestoreCheckpoint: check can_write, deny if no edit rights
    # AskUserQuestion/ExitPlanMode/SelectionUI: always show in Telegram
    captured = await _capture_interactive_content(window_id, pane_id=pane_id)
    if not captured:
        return False

    ui_name, text = captured

    # BRAIN FORK (patch 48): filter trust/security dialogs at startup
    # These appear before Claude Code is ready and should be auto-accepted,
    # not sent to Telegram as interactive UI.
    _TRUST_MARKERS = ("trust the files", "trust settings", "do you trust", "Trust this project")
    if any(marker.lower() in text.lower() for marker in _TRUST_MARKERS):
        logger.info("Auto-accepting trust dialog (window=%s): %s", window_id, text[:80])
        await tmux_manager.send_keys(window_id, "Enter", raw=True)
        return True

    # BRAIN FORK: for PermissionPrompt, enrich text with context from full pane
    # Keep only: description line + "Claude requested..." lines
    if ui_name == "PermissionPrompt":
        try:
            w = await tmux_manager.find_window_by_id(window_id)
            if w:
                full_pane = await tmux_manager.capture_pane(w.window_id)
                if full_pane:
                    pane_lines = full_pane.strip().split("\n")
                    context_lines = []
                    in_ui_block = False
                    found_empty_after_header = False
                    for pl in pane_lines:
                        if "\u2500" * 10 in pl:
                            in_ui_block = True
                            context_lines = []
                            found_empty_after_header = False
                            continue
                        if in_ui_block and "Do you want to proceed" in pl:
                            break
                        if in_ui_block:
                            stripped = pl.strip()
                            # "Claude requested..." and everything after it
                            if "Claude requested" in stripped or "but you" in stripped or "granted" in stripped:
                                context_lines.append(stripped)
                            # Description line: non-empty after first empty line (skip tool name + command)
                            elif stripped and found_empty_after_header and "Claude" not in stripped and "/" not in stripped:
                                context_lines.append(stripped)
                            elif not stripped:
                                found_empty_after_header = True
                    if context_lines:
                        # description + Claude requested... on separate lines
                        desc_parts = []
                        claude_parts = []
                        for cl in context_lines:
                            if "Claude" in cl or "but you" in cl or "granted" in cl:
                                claude_parts.append(cl)
                            else:
                                desc_parts.append(cl)
                        parts = []
                        if desc_parts:
                            parts.append(" ".join(desc_parts))
                        if claude_parts:
                            parts.append(" ".join(claude_parts))
                        text = "\n".join(parts) + "\n" + text
        except Exception:
            pass

    # BRAIN FORK: filter by UI type (patch 7 revised)
    _BLOCKED_UI = {"Settings", "SelectModel"}
    _EDIT_REQUIRED_UI = {"PermissionPrompt", "RestoreCheckpoint"}

    if ui_name in _BLOCKED_UI:
        logger.debug("Blocked system UI: %s (window=%s)", ui_name, window_id)
        return False

    if ui_name in _EDIT_REQUIRED_UI:
        # Check if user has edit rights
        import os as _os
        _ctx = _os.getenv("BRAIN_CONTEXT", "")
        _has_edit = True  # default for non-RBAC contexts
        if _ctx:
            _allowed = _os.getenv("ALLOWED_USERS", "")
            _owner_id = _allowed.split(",")[0].strip() if _allowed else ""
            if str(user_id) == _owner_id:
                _has_edit = True  # owner always has edit
            else:
                # Read can_write from current user file
                try:
                    with open(f"/tmp/brain-current-user-{_ctx}") as _uf:
                        _udata = dict(l.strip().split("=", 1) for l in _uf if "=" in l)
                    _has_edit = _udata.get("BRAIN_CURRENT_USER_CAN_WRITE", "false") == "true"
                except (OSError, ValueError):
                    _has_edit = False

        if not _has_edit:
            # Auto-deny: send Esc to tmux, notify user
            logger.info("Auto-deny %s: user %d has no edit rights", ui_name, user_id)
            await tmux_manager.send_keys(window_id, "Escape", raw=True)
            chat_id = session_manager.resolve_chat_id(user_id, thread_id)
            thread_kwargs: dict[str, int] = {}
            # BRAIN FORK: skip thread_id for DM (positive chat_id)
            if thread_id is not None and chat_id < 0:
                thread_kwargs["message_thread_id"] = thread_id
            with contextlib.suppress(TelegramError):
                await bot.send_message(
                    chat_id=chat_id,
                    text="Нет прав на изменение в этом проекте.",
                    **thread_kwargs,
                )
            return True

    # BRAIN FORK: DM windows skip interactive UI (trust auto-accepted via readiness probe)
    _ws = session_manager.get_window_state(window_id)
    if _ws and _ws.is_dm:
        logger.debug("DM window %s: skip interactive UI, auto-accepting", window_id)
        await tmux_manager.send_keys(window_id, "Enter", raw=True)
        return True

    # BRAIN FORK: clean UI -- parse options, build simple buttons (patch 7 revised)
    ikey = (user_id, thread_id or 0)
    chat_id = session_manager.resolve_chat_id(user_id, thread_id)
    clean_msg, keyboard = _build_clean_ui(text, window_id, ui_name)
    text = clean_msg

    # Try editing existing interactive message first
    existing_msg_id = _interactive_msgs.get(ikey)
    if existing_msg_id:
        return (
            await _edit_interactive_msg(
                bot, chat_id, existing_msg_id, text, keyboard, ikey, window_id
            )
            or False
        )

    # Cooldown: prevent rapid retries when sends fail
    now = time.monotonic()
    last_attempt = _send_cooldowns.get(ikey, 0.0)
    if now - last_attempt < _SEND_RETRY_INTERVAL:
        return False

    # Send new message
    thread_kwargs: dict[str, int] = {}
    # BRAIN FORK: skip thread_id for DM (positive chat_id) and General topic (thread=1)
    # General topic in forum groups: message_thread_id=1 returns "thread not found".
    # Must omit message_thread_id entirely for General topic (same as message_sender.py).
    if thread_id is not None and thread_id != 1 and chat_id < 0:
        thread_kwargs["message_thread_id"] = thread_id

    logger.info(
        "Sending interactive UI to user %d for window_id %s", user_id, window_id
    )
    _send_cooldowns[ikey] = now
    # Send as plain text — terminal content should not be formatted.
    sent: Message | None = None
    await rate_limit_send(chat_id)
    try:
        sent = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=keyboard,
            **thread_kwargs,  # type: ignore[arg-type]
        )
    except BadRequest as e:
        if is_thread_gone(e):
            logger.warning(
                "Topic gone for interactive UI (chat=%s thread=%s window=%s), "
                "backing off %ss — use /sync to recreate",
                chat_id,
                thread_id,
                window_id,
                int(_DEAD_TOPIC_RETRY_INTERVAL),
            )
            _send_cooldowns[ikey] = (
                now + _DEAD_TOPIC_RETRY_INTERVAL - _SEND_RETRY_INTERVAL
            )
            # BRAIN FORK: count failures, auto-accept after max retries (patch 43)
            _send_fail_counts[ikey] = _send_fail_counts.get(ikey, 0) + 1
            if _send_fail_counts[ikey] >= _MAX_INTERACTIVE_RETRIES:
                logger.warning(
                    "Interactive UI failed %d times for window %s, auto-accepting",
                    _send_fail_counts[ikey], window_id,
                )
                _send_fail_counts.pop(ikey, None)
                _send_cooldowns.pop(ikey, None)
                await tmux_manager.send_keys(window_id, "Enter", raw=True)
                return True
        else:
            logger.error("Failed to send interactive UI to %s: %s", chat_id, e)
    except TelegramError as e:
        logger.error("Failed to send interactive UI to %s: %s", chat_id, e)
    if sent:
        _interactive_msgs[ikey] = sent.message_id
        _interactive_mode[ikey] = window_id
        _send_cooldowns.pop(ikey, None)
        _send_fail_counts.pop(ikey, None)  # reset on success
    return sent is not None


async def clear_interactive_msg(
    user_id: int,
    bot: Bot | None = None,
    thread_id: int | None = None,
) -> None:
    """Clear tracked interactive message, delete from chat, and exit interactive mode."""
    ikey = (user_id, thread_id or 0)
    msg_id = _interactive_msgs.pop(ikey, None)
    _interactive_mode.pop(ikey, None)
    _send_cooldowns.pop(ikey, None)
    _send_fail_counts.pop(ikey, None)
    logger.debug(
        "Clear interactive msg: user=%d, thread=%s, msg_id=%s",
        user_id,
        thread_id,
        msg_id,
    )
    if bot and msg_id:
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        with contextlib.suppress(TelegramError):
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
