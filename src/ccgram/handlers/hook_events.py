"""Hook event dispatcher — routes structured events to handlers.

Receives HookEvent objects from the session monitor's event reader and
dispatches them to the appropriate handler based on event type. This
provides instant, structured notification of agent state changes instead
of relying solely on terminal scraping.

Key function: dispatch_hook_event().
"""

import structlog
from dataclasses import dataclass
from typing import Any

from telegram import Bot

from ..session import session_manager

# BRAIN FORK: Diary writing at session end (5.4)
import asyncio as _diary_asyncio
import os as _diary_os
import fcntl as _diary_fcntl
from datetime import datetime as _diary_datetime

_diary_background_tasks: set = set()
_DIARY_LOCK_FILE = "/home/agent/.claude.lock"
_DIARY_TIMEOUT = 120
_DIARY_MAX_TURNS = 5


def _get_context_from_ccgram_dir() -> str:
    ccgram_dir = _diary_os.environ.get("CCGRAM_DIR", "")
    if not ccgram_dir:
        return ""
    basename = _diary_os.path.basename(ccgram_dir.rstrip("/"))
    if basename.startswith(".ccgram-"):
        return basename[len(".ccgram-"):]
    return ""


async def _write_diary_background(context, transcript_path, cwd):
    await _diary_asyncio.sleep(3)
    today = _diary_datetime.now().strftime("%Y-%m-%d")
    diary_dir = "/home/agent/contexts/" + context + "/diary"
    diary_file = diary_dir + "/" + today + ".md"
    # No size-based skip: multiple sessions per day should all contribute
    # claude -p prompt handles dedup ("что НОВОГО с последней записи")
    try:
        lock_fd = open(_DIARY_LOCK_FILE, "w")
        _diary_fcntl.flock(lock_fd, _diary_fcntl.LOCK_EX | _diary_fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        logger.info("Claude lock busy, skipping diary write")
        try:
            lock_fd.close()
        except Exception:
            pass
        return
    try:
        config_dir = "/home/agent/contexts/" + context + "/config"
        prompt = (
            "Посмотри что произошло с последней записи в дневнике. Прочитай diary/ за сегодня если есть, допиши только НОВОЕ. "
            "Фокус на 6 категориях: (1) Решения и ПОЧЕМУ, (2) Бизнес-обсуждения и идеи (даже без кода), "
            "(3) Новые знания и коррекции от пользователя, (4) Ключевые изменения (суть, не список файлов), "
            "(5) Проблемы и как решили, (6) Открытые вопросы (что подвешено). "
            "Тест значимости: через неделю это будет ценно знать? "
            "Если за сессию ничего из 6 категорий не было, НЕ создавай файл. "
            "Если было значимое, запиши в " + diary_file + " (абсолютный путь). "
            "Формат: ## Сделано, ## Решения, ## Наблюдения, ## Блокеры, ## Завтра. "
            "Append если файл уже существует. Кратко, по делу, без воды."
        )
        env = _diary_os.environ.copy()
        env["CLAUDE_CONFIG_DIR"] = config_dir
        proc = await _diary_asyncio.create_subprocess_exec(
            "/home/agent/bin/claude-context.sh",
            "-p", prompt,
            "--max-turns", str(_DIARY_MAX_TURNS),
            stdout=_diary_asyncio.subprocess.DEVNULL,
            stderr=_diary_asyncio.subprocess.DEVNULL,
            cwd="/home/agent",
            env=env,
        )
        try:
            await _diary_asyncio.wait_for(proc.wait(), timeout=_DIARY_TIMEOUT)
            logger.info("Diary written for context=%s exit=%d", context, proc.returncode)
        except _diary_asyncio.TimeoutError:
            proc.kill()
            logger.warning("Diary write timed out for context=%s", context)
    except Exception as e:
        logger.error("Diary write failed: %s", e)
    finally:
        try:
            _diary_fcntl.flock(lock_fd, _diary_fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass


def trigger_diary_write(context, transcript_path, cwd):
    if not context:
        return
    task = _diary_asyncio.ensure_future(
        _write_diary_background(context, transcript_path, cwd)
    )
    _diary_background_tasks.add(task)
    task.add_done_callback(_diary_background_tasks.discard)
    logger.info("Diary write triggered for context=%s", context)


logger = structlog.get_logger()

_WINDOW_KEY_PARTS = 2


@dataclass
class HookEvent:
    """A structured event from the hook event log."""

    event_type: str  # "Notification", "Stop", etc.
    window_key: str  # "ccgram:@0"
    session_id: str
    data: dict[str, Any]
    timestamp: float


def _resolve_users_for_window_key(
    window_key: str,
) -> list[tuple[int, int, str]]:
    """Resolve window_key to list of (user_id, thread_id, window_id).

    The window_key format is "tmux_session:window_id" (e.g. "ccgram:@0").
    We extract the window_id part and look up thread bindings.
    """
    # Extract window_id from key (e.g. "ccgram:@0" -> "@0")
    parts = window_key.rsplit(":", 1)
    if len(parts) < _WINDOW_KEY_PARTS:
        return []
    window_id = parts[1]

    results: list[tuple[int, int, str]] = []
    for user_id, thread_id, bound_wid in session_manager.iter_thread_bindings():
        if bound_wid == window_id:
            results.append((user_id, thread_id, window_id))
    return results


async def _handle_notification(event: HookEvent, bot: Bot) -> None:
    """Handle a Notification event — render interactive UI."""
    from .interactive_ui import (
        get_interactive_window,
        handle_interactive_ui,
        set_interactive_mode,
    )

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        logger.debug(
            "No users bound for notification event window_key=%s", event.window_key
        )
        return

    tool_name = event.data.get("tool_name", "")
    logger.debug(
        "Hook notification: tool_name=%s, window_key=%s",
        tool_name,
        event.window_key,
    )

    for user_id, thread_id, window_id in users:
        # Skip if already in interactive mode for this window
        existing = get_interactive_window(user_id, thread_id)
        if existing == window_id:
            logger.debug(
                "Interactive mode already set for user=%d window=%s, skipping",
                user_id,
                window_id,
            )
            continue

        # Set interactive mode before rendering to prevent racing with terminal scraping
        set_interactive_mode(user_id, window_id, thread_id)

        # Wait briefly for Claude Code to render the UI in the terminal
        import asyncio

        await asyncio.sleep(0.3)

        handled = await handle_interactive_ui(bot, user_id, window_id, thread_id)
        if not handled:
            from .interactive_ui import clear_interactive_mode

            clear_interactive_mode(user_id, thread_id)


async def _handle_stop(event: HookEvent, bot: Bot) -> None:
    """Handle a Stop event — transition directly to idle.

    Edits the status message in-place to "Ready" (dedup catches identical
    text) and sets the topic emoji to idle without an intermediate active
    flicker.  Muted/errors_only windows get their status cleared instead.

    BRAIN FORK (patch 53): also clears stale status for other users who
    share the same thread_id through different windows, if those windows
    are idle (no recent transcript activity).
    """
    from .callback_data import IDLE_STATUS_TEXT
    from .message_queue import enqueue_status_update
    from .topic_emoji import update_topic_emoji

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return


    # BRAIN FORK (patch 48): clear transcript activity to stop typing immediately
    from .status_polling import get_active_monitor
    mon = get_active_monitor()
    if mon and users:
        _wid = users[0][2]
        _sid = session_manager.get_session_id_for_window(_wid)
        if _sid:
            mon.clear_activity(_sid)
    stop_reason = event.data.get("stop_reason", "")
    # BRAIN FORK (patch 48): reset failure counter on successful stop
    _stop_failure_counts.pop(event.window_key, None)
    logger.debug(
        "Hook stop: window_key=%s, stop_reason=%s",
        event.window_key,
        stop_reason,
    )

    for user_id, thread_id, window_id in users:
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        display = session_manager.get_display_name(window_id)
        await update_topic_emoji(bot, chat_id, thread_id, "idle", display)
        notif_mode = session_manager.get_notification_mode(window_id)
        # BRAIN FORK: don't show Ready status, just clear it
        status_text = None
        await enqueue_status_update(
            bot, user_id, window_id, status_text, thread_id=thread_id
        )

    # BRAIN FORK (patch 53): clear stale status for sibling users in shared topics.
    # When two users share a thread (e.g. thread_id=6 via windows @41 and @29),
    # Stop for @29 only clears user_B's status. User_A's "Thinking..." stays stuck
    # because @41 never fired Stop. Fix: find all other users bound to the same
    # thread_id(s) and clear their status if their window is also idle.
    import time as _time
    _SIBLING_ACTIVITY_THRESHOLD = 10.0  # seconds, matches status_polling threshold

    affected_thread_ids = {tid for _, tid, _ in users}
    handled_pairs = {(uid, tid) for uid, tid, _ in users}

    for thread_id in affected_thread_ids:
        for other_uid, other_tid, other_wid in session_manager.iter_thread_bindings():
            if other_tid != thread_id:
                continue
            if (other_uid, other_tid) in handled_pairs:
                continue

            # Check if the sibling window is actively working — don't clear if so
            other_sid = session_manager.get_session_id_for_window(other_wid)
            if other_sid and mon:
                last_activity = mon.get_last_activity(other_sid)
                if last_activity and (_time.monotonic() - last_activity) < _SIBLING_ACTIVITY_THRESHOLD:
                    logger.debug(
                        "Shared topic stop: skipping active sibling user=%d thread=%d window=%s",
                        other_uid, other_tid, other_wid,
                    )
                    continue

            logger.info(
                "Shared topic stop: clearing stale status for sibling user=%d thread=%d window=%s",
                other_uid, other_tid, other_wid,
            )
            await enqueue_status_update(
                bot, other_uid, other_wid, None, thread_id=other_tid
            )


# Track active subagents per window: window_id -> {subagent_id -> name}
_active_subagents: dict[str, dict[str, str]] = {}

_MAX_DISPLAYED_NAMES = 3


def get_subagent_names(window_id: str) -> list[str]:
    """Return names of active subagents for a window."""
    return list(_active_subagents.get(window_id, {}).values())


def build_subagent_label(names: list[str]) -> str | None:
    """Build a display label for active subagents.

    Returns None if no subagents are active.
    """
    if not names:
        return None
    if len(names) == 1:
        return f"\U0001f916 {names[0]}"
    joined = ", ".join(names[:_MAX_DISPLAYED_NAMES])
    return f"\U0001f916 {len(names)} subagents: {joined}"


def clear_subagents(window_id: str) -> None:
    """Clear all subagent tracking for a window."""
    _active_subagents.pop(window_id, None)


async def _handle_subagent_start(event: HookEvent, bot: Bot) -> None:
    """Handle SubagentStart — track active subagent and notify."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]  # all users share the same window_id
    subagent_id = event.data.get("subagent_id", "")
    name = (
        (event.data.get("name") or "").strip()
        or (event.data.get("description") or "").strip()
        or subagent_id[:12]
        or "subagent"
    )

    _active_subagents.setdefault(window_id, {})[subagent_id] = name

    logger.debug(
        "Subagent started: window=%s, count=%d, name=%s",
        window_id,
        len(_active_subagents[window_id]),
        name,
    )

    for user_id, thread_id, _ in users:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            f"\U0001f916 Subagent started: {name}",
            thread_id=thread_id,
        )


async def _handle_subagent_stop(event: HookEvent, bot: Bot) -> None:
    """Handle SubagentStop — remove subagent from tracking and notify."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    window_id = users[0][2]
    subagent_id = event.data.get("subagent_id", "")

    agents = _active_subagents.get(window_id)
    if not agents:
        return
    name = agents.pop(subagent_id, subagent_id[:12] or "subagent")
    if not agents:
        _active_subagents.pop(window_id, None)

    logger.debug(
        "Subagent stopped: window=%s, remaining=%d, name=%s",
        window_id,
        len(_active_subagents.get(window_id, {})),
        name,
    )

    for user_id, thread_id, _ in users:
        await enqueue_status_update(
            bot,
            user_id,
            window_id,
            f"\U0001f916 Subagent done: {name}",
            thread_id=thread_id,
        )


async def _handle_teammate_idle(event: HookEvent, bot: Bot) -> None:
    """Handle TeammateIdle — notify topic that a teammate went idle."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    teammate_name = event.data.get("teammate_name", "unknown")
    logger.info(
        "Teammate idle: window_key=%s, teammate=%s",
        event.window_key,
        teammate_name,
    )

    for user_id, thread_id, window_id in users:
        text = f"\U0001f4a4 Teammate '{teammate_name}' went idle"
        await enqueue_status_update(bot, user_id, window_id, text, thread_id=thread_id)


# BRAIN FORK (patch 48): auto-recovery for consecutive API failures
_stop_failure_counts: dict[str, int] = {}  # window_key -> consecutive failure count
_stop_failure_cooldowns: dict[str, float] = {}  # window_key -> last auto-restart time
_STOP_FAILURE_THRESHOLD = 3  # auto-restart after N consecutive failures
_STOP_FAILURE_COOLDOWN = 300.0  # 5 min cooldown between auto-restarts


async def _handle_stop_failure(event: HookEvent, bot: Bot) -> None:
    """Handle a StopFailure event — alert on API error, auto-restart on repeated failures."""
    from .message_sender import rate_limit_send_message

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    error = event.data.get("error", "unknown")
    error_details = event.data.get("error_details", "")
    logger.warning(
        "Hook StopFailure: window_key=%s, error=%s, details=%s",
        event.window_key,
        error,
        error_details,
    )

    detail = f": {error_details}" if error_details else ""
    text = f"\u26a0 API error — {error}{detail}"

    for user_id, thread_id, _window_id in users:
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        await rate_limit_send_message(bot, chat_id, text, message_thread_id=thread_id)

    # BRAIN FORK (patch 48): track consecutive failures and auto-restart
    import time as _time
    wkey = event.window_key
    _stop_failure_counts[wkey] = _stop_failure_counts.get(wkey, 0) + 1
    count = _stop_failure_counts[wkey]
    logger.warning("StopFailure #%d for %s", count, wkey)

    if count >= _STOP_FAILURE_THRESHOLD:
        last_restart = _stop_failure_cooldowns.get(wkey, 0.0)
        if _time.monotonic() - last_restart < _STOP_FAILURE_COOLDOWN:
            logger.info("Auto-restart cooldown active for %s, skipping", wkey)
            return

        _stop_failure_cooldowns[wkey] = _time.monotonic()
        _stop_failure_counts[wkey] = 0

        # Extract context and window_id from window_key (format: "context:@id")
        parts = wkey.split(":", 1)
        if len(parts) == 2:
            ctx, wid = parts
            logger.warning("Auto-restarting stuck session %s after %d failures", wkey, count)

            # Notify user
            for user_id, thread_id, _ in users:
                chat_id = session_manager.resolve_chat_id(user_id, thread_id)
                await rate_limit_send_message(
                    bot, chat_id,
                    "Сессия зависла на ошибке API. Перезапускаю...",
                    message_thread_id=thread_id,
                )

            # Restart via script (--silent --no-ccgram-restart to avoid killing ourselves)
            import asyncio, subprocess
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/home/agent/scripts/restart-claude.sh", ctx, wid,
                    "--silent", "--no-ccgram-restart",
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=90)
                logger.info("Auto-restart completed for %s (exit=%s)", wkey, proc.returncode)
            except (asyncio.TimeoutError, OSError) as e:
                logger.error("Auto-restart failed for %s: %s", wkey, e)


async def _handle_session_end(event: HookEvent, bot: Bot) -> None:
    """Handle a SessionEnd event — clean up session lifecycle."""
    from .message_queue import enqueue_status_update
    from .status_polling import clear_seen_status
    from .topic_emoji import update_topic_emoji

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    reason = event.data.get("reason", "")
    logger.info(
        "Hook SessionEnd: window_key=%s, reason=%s",
        event.window_key,
        reason,
    )

    # Clear session association and subagent tracking so next launch starts fresh
    if users:
        window_id = users[0][2]
        # BRAIN FORK: trigger diary write before clearing session (5.4)
        state = session_manager.get_window_state(window_id)
        context = _get_context_from_ccgram_dir()
        trigger_diary_write(context, state.transcript_path, state.cwd)

        session_manager.clear_window_session(window_id)
        clear_subagents(window_id)

    for user_id, thread_id, window_id in users:
        clear_seen_status(window_id)
        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        display = session_manager.get_display_name(window_id)
        await update_topic_emoji(bot, chat_id, thread_id, "done", display)
        await enqueue_status_update(bot, user_id, window_id, None, thread_id=thread_id)


async def _handle_task_completed(event: HookEvent, bot: Bot) -> None:
    """Handle TaskCompleted — notify topic that a task was completed."""
    from .message_queue import enqueue_status_update

    users = _resolve_users_for_window_key(event.window_key)
    if not users:
        return

    task_subject = event.data.get("task_subject", "")
    teammate_name = event.data.get("teammate_name", "")
    logger.info(
        "Task completed: window_key=%s, task=%s, by=%s",
        event.window_key,
        task_subject,
        teammate_name,
    )

    for user_id, thread_id, window_id in users:
        text = f"\u2705 Task completed: {task_subject}"
        if teammate_name:
            text += f" (by '{teammate_name}')"
        await enqueue_status_update(bot, user_id, window_id, text, thread_id=thread_id)


async def dispatch_hook_event(event: HookEvent, bot: Bot) -> None:
    """Route hook events to appropriate handlers."""
    match event.event_type:
        case "Notification":
            await _handle_notification(event, bot)
        case "Stop":
            await _handle_stop(event, bot)
        case "StopFailure":
            await _handle_stop_failure(event, bot)
        case "SessionEnd":
            await _handle_session_end(event, bot)
        case "SubagentStart":
            await _handle_subagent_start(event, bot)
        case "SubagentStop":
            await _handle_subagent_stop(event, bot)
        case "TeammateIdle":
            await _handle_teammate_idle(event, bot)
        case "TaskCompleted":
            await _handle_task_completed(event, bot)
        case (
            "SessionStart"
            | "UserPromptSubmit"
            | "PreToolUse"
            | "PostToolUse"
            | "PostToolUseFailure"
            | "PermissionRequest"
            | "ConfigChange"
            | "WorktreeCreate"
            | "WorktreeRemove"
            | "PreCompact"
        ):
            pass  # Not actionable for the bot — SessionStart handled via session_map.json
        case _:
            logger.debug("Ignoring unknown hook event type: %s", event.event_type)
