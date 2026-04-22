"""Interactive UI callback handlers.

Handles inline keyboard callbacks for AskUserQuestion/ExitPlanMode/Permission UIs:
  - CB_ASK_* direction/action keys: navigate interactive UI via tmux keys
  - CB_ASK_REFRESH: refresh the interactive UI display

Key function: handle_interactive_callback (uniform callback handler signature).
"""

import asyncio
import structlog

from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from ..tmux_manager import tmux_manager
from .callback_data import (
    CB_ASK_AMEND,
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
from .interactive_ui import (
    AMEND_IKEY_KEY,
    AMEND_STATE_KEY,
    CB_OPTION,
    STATE_AMENDING_ANSWER,
    _interactive_options,
    _interactive_ui_names,
    clear_interactive_msg,
    enter_amend_mode,
    finalize_interactive_msg,
    handle_interactive_ui,
)

logger = structlog.get_logger()

# cb_prefix -> (tmux_key, refresh_ui_after)
INTERACTIVE_KEY_MAP: dict[str, tuple[str, bool]] = {
    CB_ASK_UP: ("Up", True),
    CB_ASK_DOWN: ("Down", True),
    CB_ASK_LEFT: ("Left", True),
    CB_ASK_RIGHT: ("Right", True),
    CB_ASK_ESC: ("Escape", False),
    CB_ASK_ENTER: ("Enter", True),
    CB_ASK_SPACE: ("Space", True),
    CB_ASK_TAB: ("Tab", True),
}

# Answer-toast labels for interactive key callbacks
INTERACTIVE_KEY_LABELS: dict[str, str] = {
    CB_ASK_ESC: "\u238b Esc",
    CB_ASK_ENTER: "\u23ce Enter",
    CB_ASK_SPACE: "\u2423 Space",
    CB_ASK_TAB: "\u21e5 Tab",
}

# All interactive prefixes (key map + refresh + option + amend)
INTERACTIVE_PREFIXES: tuple[str, ...] = (
    *INTERACTIVE_KEY_MAP,
    CB_ASK_REFRESH,
    CB_ASK_AMEND,  # BRAIN FORK (patch 59): "Your own answer" for AskUserQuestion
    CB_OPTION,
)


def match_interactive_prefix(data: str) -> tuple[str, str, str | None] | None:
    """Match callback data against interactive UI prefixes.

    Returns (cb_prefix, window_id, pane_id_or_None) or None.

    Callback data format:
      - ``"aq:enter:@12"``      → window @12, active pane
      - ``"aq:enter:@12:%5"``   → window @12, specific pane %5
    """
    for prefix in INTERACTIVE_PREFIXES:
        if data.startswith(prefix):
            remainder = data[len(prefix) :]
            # Check for pane_id suffix: "@12:%5"
            if ":%" in remainder:
                window_id, pane_id = remainder.split(":%", 1)
                return prefix, window_id, f"%{pane_id}"
            return prefix, remainder, None
    return None


async def handle_interactive_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle interactive UI callbacks (AskUserQuestion/ExitPlanMode navigation)."""
    matched = match_interactive_prefix(data)
    if not matched:
        return

    cb_prefix, window_id, pane_id = matched
    from .callback_helpers import get_thread_id, user_owns_window

    # BRAIN FORK: for option buttons, extract real window_id before ownership check
    _check_window_id = window_id
    if cb_prefix == CB_OPTION and ":" in window_id:
        _check_window_id = window_id.rsplit(":", 1)[0]

    if not user_owns_window(user_id, _check_window_id):
        await query.answer("Not your session", show_alert=True)
        return

    thread_id = get_thread_id(update)

    if cb_prefix == CB_ASK_REFRESH:
        await handle_interactive_ui(
            context.bot, user_id, window_id, thread_id, pane_id=pane_id
        )
        await query.answer("\U0001f504")
    elif cb_prefix == CB_OPTION:
        # BRAIN FORK: clean option button (patch 7 revised)
        # window_id format: "@10:2" where 2 is option index
        parts = window_id.rsplit(":", 1)
        real_window_id = parts[0]
        option_idx = int(parts[1]) if len(parts) > 1 else 0
        w = await tmux_manager.find_window_by_id(real_window_id)
        if w:
            # Move cursor down to selected option, then Enter
            for _ in range(option_idx):
                await tmux_manager.send_keys(w.window_id, "Down", enter=False, literal=False)
                await asyncio.sleep(0.1)
            await tmux_manager.send_keys(w.window_id, "Enter", enter=False, literal=False)
        # BRAIN FORK (patch 59): AskUserQuestion → edit-in-place with "Selected: X" echo.
        # Any other numbered UI (SelectionUI/RestoreCheckpoint/etc.) → delete as before.
        ikey = (user_id, thread_id or 0)
        ui_name = _interactive_ui_names.get(ikey, "")
        if ui_name == "AskUserQuestion":
            labels = _interactive_options.get(ikey, [])
            label = labels[option_idx] if 0 <= option_idx < len(labels) else ""
            # ✓︎ = CHECK MARK with Variation Selector-15 (forces text, not emoji render)
            result = f"✓︎ Selected: {label}" if label else "✓︎ Selected"
            await finalize_interactive_msg(user_id, context.bot, thread_id, result)
        else:
            await clear_interactive_msg(user_id, context.bot, thread_id)
        await query.answer("OK")
    elif cb_prefix == CB_ASK_AMEND:
        # BRAIN FORK (patch 59): "Your own answer" — send Tab, wait for next text
        if pane_id:
            sent = await tmux_manager.send_keys_to_pane(
                pane_id, "Tab", enter=False, literal=False, window_id=window_id
            )
        else:
            w = await tmux_manager.find_window_by_id(window_id)
            sent = bool(w) and await tmux_manager.send_keys(
                w.window_id, "Tab", enter=False, literal=False
            )
        if sent:
            ikey = (user_id, thread_id or 0)
            if context.user_data is not None:
                context.user_data[AMEND_STATE_KEY] = STATE_AMENDING_ANSWER
                context.user_data[AMEND_IKEY_KEY] = ikey
            await enter_amend_mode(user_id, context.bot, thread_id, window_id)
            await query.answer("Type your answer...")
        else:
            await query.answer("Tab failed", show_alert=True)
    else:
        tmux_key, refresh_ui = INTERACTIVE_KEY_MAP[cb_prefix]
        if pane_id:
            sent = await tmux_manager.send_keys_to_pane(
                pane_id, tmux_key, enter=False, literal=False, window_id=window_id
            )
        else:
            w = await tmux_manager.find_window_by_id(window_id)
            sent = bool(w) and await tmux_manager.send_keys(
                w.window_id, tmux_key, enter=False, literal=False
            )
        if sent and refresh_ui:
            await asyncio.sleep(0.5)
            await handle_interactive_ui(
                context.bot, user_id, window_id, thread_id, pane_id=pane_id
            )
        elif sent and not refresh_ui:
            # BRAIN FORK (patch 59): AskUserQuestion Esc → "Cancelled" echo in place.
            # ExitPlanMode / PermissionPrompt / others → delete (old behavior).
            # Always clears stale amend-mode flag (only AskUserQuestion could have set it).
            if cb_prefix == CB_ASK_ESC:
                ikey = (user_id, thread_id or 0)
                ui_name = _interactive_ui_names.get(ikey, "")
                if context.user_data is not None:
                    if context.user_data.get(AMEND_STATE_KEY) == STATE_AMENDING_ANSWER:
                        context.user_data.pop(AMEND_STATE_KEY, None)
                        context.user_data.pop(AMEND_IKEY_KEY, None)
                if ui_name == "AskUserQuestion":
                    # ✕︎ = MULTIPLICATION X with Variation Selector-15 (forces text render)
                    await finalize_interactive_msg(
                        user_id, context.bot, thread_id, "✕︎ Cancelled"
                    )
                else:
                    await clear_interactive_msg(user_id, context.bot, thread_id)
            else:
                await clear_interactive_msg(user_id, context.bot, thread_id)
        await query.answer(INTERACTIVE_KEY_LABELS.get(cb_prefix, ""))
