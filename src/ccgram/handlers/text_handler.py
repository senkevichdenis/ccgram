"""Text message handling — step functions for the text_handler orchestrator.

Routes incoming text messages through a bool early-return chain:
UI guards → unbound topic → dead window recovery → message forwarding.

Each step returns True if it handled the request (stop) or False to continue.
The orchestrator (handle_text_message) calls steps in sequence.
"""

import asyncio
import structlog
from pathlib import Path
from telegram import Bot, Message, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from .msg_batcher import enqueue as _batch_enqueue

from .callback_helpers import get_thread_id as _get_thread_id
from .directory_browser import (
    BROWSE_DIRS_KEY,
    BROWSE_PAGE_KEY,
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    STATE_SELECTING_WINDOW,
    UNBOUND_WINDOWS_KEY,
    build_directory_browser,
    build_window_picker,
    clear_browse_state,
    clear_window_picker_state,
)
from .interactive_ui import (
    AMEND_IKEY_KEY,
    AMEND_STATE_KEY,
    STATE_AMENDING_ANSWER,
    finalize_interactive_msg,
    get_interactive_window,
    handle_interactive_ui,
)
from .message_queue import enqueue_status_update
from .message_sender import (
    ack_reaction,
    edit_with_fallback,
    rate_limit_send_message,
    safe_reply,
)
from .recovery_callbacks import build_recovery_keyboard
from .status_polling import clear_probe_failures
from .user_state import PENDING_THREAD_ID, PENDING_THREAD_TEXT, RECOVERY_WINDOW_ID
from ..session import session_manager
from ..providers import get_provider_for_window
from ..tmux_manager import tmux_manager
from ..utils import handle_general_topic_message, is_general_topic, task_done_callback
from ..config import config as _config


logger = structlog.get_logger()

# Maximum characters for bash output before truncation (fits Telegram 4096-char limit)
_BASH_OUTPUT_LIMIT = 3800

# Active bash capture tasks: (user_id, thread_id) -> asyncio.Task
_bash_capture_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}


def _cancel_bash_capture(user_id: int, thread_id: int) -> None:
    """Cancel any running bash capture for this topic."""
    key = (user_id, thread_id)
    task = _bash_capture_tasks.pop(key, None)
    if task and not task.done():
        task.cancel()


async def _edit_bash_message(bot: Bot, chat_id: int, msg_id: int, output: str) -> None:
    """Edit an existing bash-output message with entity-based formatting fallback."""
    await edit_with_fallback(bot, chat_id, msg_id, output)


async def _capture_bash_output(
    bot: Bot,
    user_id: int,
    thread_id: int,
    window_id: str,
    command: str,
) -> None:
    """Background task: capture ``!`` bash command output from tmux pane.

    Sends the first captured output as a new message, then edits it
    in-place as more output appears.  Stops after 30 s or when cancelled
    (e.g. user sends a new message, which pushes content down).
    """
    try:
        # Wait for the command to start producing output
        await asyncio.sleep(2.0)

        chat_id = session_manager.resolve_chat_id(user_id, thread_id)
        msg_id: int | None = None
        last_output: str = ""

        for _ in range(30):
            raw = await tmux_manager.capture_pane(window_id)
            if raw is None:
                return

            output = get_provider_for_window(window_id).extract_bash_output(
                raw, command
            )
            if not output or output == last_output:
                await asyncio.sleep(1.0)
                continue

            last_output = output

            # Truncate to fit Telegram's 4096-char limit
            if len(output) > _BASH_OUTPUT_LIMIT:
                output = "\u2026 " + output[-_BASH_OUTPUT_LIMIT:]

            if msg_id is None:
                # First capture — send a new message
                sent = await rate_limit_send_message(
                    bot,
                    chat_id,
                    output,
                    message_thread_id=thread_id,
                )
                if sent:
                    msg_id = sent.message_id
            else:
                await _edit_bash_message(bot, chat_id, msg_id, output)

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    finally:
        key = (user_id, thread_id)
        if _bash_capture_tasks.get(key) is asyncio.current_task():
            _bash_capture_tasks.pop(key, None)


async def _check_ui_guards(
    user_data: dict | None, thread_id: int | None, message: Message
) -> bool:
    """Block text while a window picker or directory browser is active.

    Returns True if the message was handled (blocked), False to continue.
    """
    if not user_data:
        return False

    # Window picker guard
    if user_data.get(STATE_KEY) == STATE_SELECTING_WINDOW:
        pending_tid = user_data.get(PENDING_THREAD_ID)
        if pending_tid == thread_id:
            await safe_reply(
                message,
                "Please use the window picker above, or tap Cancel.",
            )
            return True
        # Stale picker state from a different thread — clear it
        clear_window_picker_state(user_data)
        user_data.pop(PENDING_THREAD_ID, None)
        user_data.pop(PENDING_THREAD_TEXT, None)

    # Directory browser guard
    if user_data.get(STATE_KEY) == STATE_BROWSING_DIRECTORY:
        pending_tid = user_data.get(PENDING_THREAD_ID)
        if pending_tid == thread_id:
            await safe_reply(
                message,
                "Please use the directory browser above, or tap Cancel.",
            )
            return True
        # Stale browsing state from a different thread — clear it
        clear_browse_state(user_data)
        user_data.pop(PENDING_THREAD_ID, None)
        user_data.pop(PENDING_THREAD_TEXT, None)

    return False



async def auto_bind_window_for_preset(
    user_id: int,
    thread_id: int,
    message: Message,
) -> str | None:
    """BRAIN FORK (file_handler unblock): create+bind window from topic_presets.json
    WITHOUT forwarding any text. Used by both text_handler (with text forwarding
    in `_try_auto_bind_from_preset` wrapper) and file_handler (voice/photo/file
    uploads), so first media in a new preset-backed topic doesn\'t fail with
    "No session bound to this topic.".

    Returns created window_id on success, None if no preset / preset path
    invalid / window creation failed.
    """
    import json as _json

    # Already bound? Caller should not reach here, but be defensive.
    existing_wid = session_manager.get_window_for_thread(user_id, thread_id)
    if existing_wid is not None:
        return existing_wid

    presets_file = _config.topic_presets_file
    if not presets_file.exists():
        return None
    try:
        with open(presets_file) as f:
            presets = _json.load(f)
    except (ValueError, OSError):
        return None

    preset = presets.get(str(thread_id))
    if not preset:
        return None

    path = preset.get("path", "")
    provider_name = preset.get("provider", "claude")
    approval_mode = preset.get("mode", "yolo")

    if not path:
        return None

    from pathlib import Path as _Path
    if not _Path(path).is_dir():
        logger.warning("Preset path does not exist: %s (thread=%d)", path, thread_id)
        return None

    logger.info(
        "Auto-binding from preset: thread=%d -> %s (provider=%s, mode=%s)",
        thread_id, path, provider_name, approval_mode,
    )

    await message.chat.send_action("typing")

    from ccgram.providers import resolve_launch_command
    launch_command = resolve_launch_command(provider_name, approval_mode=approval_mode)
    from ccgram.session import find_resumable_args_for_path
    auto_args = find_resumable_args_for_path(path, provider_name, user_id, thread_id)
    success, msg, created_wname, created_wid = await tmux_manager.create_window(
        path, launch_command=launch_command, agent_args=auto_args
    )
    if not success:
        from .message_sender import safe_reply
        await safe_reply(message, f"Failed to create window: {msg}")
        return None

    session_manager.update_user_mru(user_id, path)
    window_state = session_manager.get_window_state(created_wid)
    window_state.cwd = path
    session_manager.set_window_provider(created_wid, provider_name)
    session_manager.set_window_approval_mode(created_wid, approval_mode)
    await tmux_manager.stamp_pane_title(created_wid, provider_name)

    session_manager.bind_thread(user_id, thread_id, created_wid, window_name=created_wname)

    chat = message.chat
    if chat.type in ("group", "supergroup"):
        session_manager.set_group_chat_id(user_id, thread_id, chat.id)

    return created_wid


async def _try_auto_bind_from_preset(
    user_id: int,
    thread_id: int,
    text: str,
    message: Message,
    bot: Bot,
) -> bool:
    """text_handler entrypoint: auto-bind via preset, then forward the original
    text into the new window. Thin wrapper over auto_bind_window_for_preset.

    Returns True if preset was found and window created (handled), False to continue.
    """
    created_wid = await auto_bind_window_for_preset(user_id, thread_id, message)
    if created_wid is None:
        return False

    send_ok, send_msg = await session_manager.send_to_window(created_wid, text)
    if not send_ok:
        logger.warning("Failed to forward preset pending text: %s", send_msg)

    return True


async def _handle_unbound_topic(
    user_id: int,
    thread_id: int,
    text: str,
    user_data: dict | None,
    message: Message,
) -> bool:
    """Show window picker or directory browser for an unbound topic.

    Returns True if the topic is unbound (handled), False if already bound.
    """
    window_id = session_manager.get_window_for_thread(user_id, thread_id)
    if window_id is not None:
        return False

    all_windows = await tmux_manager.list_windows()
    external_windows = await tmux_manager.discover_external_sessions()
    all_windows.extend(external_windows)
    bound_ids = {
        bound_wid for _, _, bound_wid in session_manager.iter_thread_bindings()
    }
    unbound = [
        (w.window_id, w.window_name, w.cwd)
        for w in all_windows
        if w.window_id not in bound_ids
    ]
    logger.debug(
        "Window picker check: all=%s, bound=%s, unbound=%s",
        [w.window_name for w in all_windows],
        bound_ids,
        [name for _, name, _ in unbound],
    )

    if unbound:
        logger.info(
            "Unbound topic: showing window picker (%d unbound windows, user=%d, thread=%d)",
            len(unbound),
            user_id,
            thread_id,
        )
        msg_text, keyboard, win_ids = build_window_picker(unbound)
        if user_data is not None:
            user_data[STATE_KEY] = STATE_SELECTING_WINDOW
            user_data[UNBOUND_WINDOWS_KEY] = win_ids
            user_data[PENDING_THREAD_ID] = thread_id
            user_data[PENDING_THREAD_TEXT] = text
        await safe_reply(message, msg_text, reply_markup=keyboard)
        return True

    # No unbound windows — show directory browser to create a new session
    logger.info(
        "Unbound topic: showing directory browser (user=%d, thread=%d)",
        user_id,
        thread_id,
    )
    start_path = str(Path.cwd())
    msg_text, keyboard, subdirs = build_directory_browser(start_path, user_id=user_id)
    if user_data is not None:
        user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
        user_data[BROWSE_PATH_KEY] = start_path
        user_data[BROWSE_PAGE_KEY] = 0
        user_data[BROWSE_DIRS_KEY] = subdirs
        user_data[PENDING_THREAD_ID] = thread_id
        user_data[PENDING_THREAD_TEXT] = text
    await safe_reply(message, msg_text, reply_markup=keyboard)
    return True


async def _handle_dead_window(
    window_id: str,
    user_id: int,
    thread_id: int,
    text: str,
    user_data: dict | None,
    message: Message,
) -> bool:
    """Show recovery UI or directory browser for a dead (killed) window.

    Returns True if the window is dead (handled), False if still alive.
    """
    w = await tmux_manager.find_window_by_id(window_id)
    if w:
        return False

    display = session_manager.get_display_name(window_id)
    window_state = session_manager.get_window_state(window_id)
    cwd = window_state.cwd if window_state.cwd else ""

    if not cwd or not Path(cwd).is_dir():
        # No valid cwd — unbind and fall back to directory browser
        logger.info(
            "Dead window %s (no valid cwd), falling back to directory browser"
            " (user=%d, thread=%d)",
            window_id,
            user_id,
            thread_id,
        )
        session_manager.unbind_thread(user_id, thread_id)
        from .status_polling import clear_dead_notification

        clear_dead_notification(user_id, thread_id)
        start_path = str(Path.cwd())
        msg_text, keyboard, subdirs = build_directory_browser(
            start_path, user_id=user_id
        )
        if user_data is not None:
            user_data[STATE_KEY] = STATE_BROWSING_DIRECTORY
            user_data[BROWSE_PATH_KEY] = start_path
            user_data[BROWSE_PAGE_KEY] = 0
            user_data[BROWSE_DIRS_KEY] = subdirs
            user_data[PENDING_THREAD_ID] = thread_id
            user_data[PENDING_THREAD_TEXT] = text
        await safe_reply(message, msg_text, reply_markup=keyboard)
        return True

    # Show recovery UI
    logger.info(
        "Dead window %s (%s), showing recovery UI (user=%d, thread=%d)",
        window_id,
        display,
        user_id,
        thread_id,
    )
    if user_data is not None:
        user_data[PENDING_THREAD_ID] = thread_id
        user_data[PENDING_THREAD_TEXT] = text
        user_data[RECOVERY_WINDOW_ID] = window_id
    keyboard = build_recovery_keyboard(window_id)
    await safe_reply(
        message,
        f"\u26a0 Window `{display}` is no longer running.\n"
        f"\U0001f4c2 `{cwd}`\n\n"
        "How would you like to recover?",
        reply_markup=keyboard,
    )
    return True





async def _forward_message(
    window_id: str,
    user_id: int,
    thread_id: int,
    text: str,
    bot: Bot,
    message: Message,
) -> None:
    """Forward a text message to the bound tmux window."""
    await message.chat.send_action(ChatAction.TYPING)  # type: ignore[union-attr]
    # Enqueue a status clear to actually delete the Telegram message
    # (clear_status_msg_info only clears the tracking dict, leaving a ghost)
    await enqueue_status_update(bot, user_id, window_id, None, thread_id)

    # Cancel any running bash capture — new message pushes pane content down
    _cancel_bash_capture(user_id, thread_id)

    clear_probe_failures(window_id)

    # BRAIN FORK: use msg_batcher instead of direct send (patch 35)
    await _batch_enqueue(window_id, text)

    from .command_history import record_command

    record_command(user_id, thread_id, text)

    # Start background capture for ! bash command output
    if text.startswith("!") and len(text) > 1:
        bash_cmd = text[1:]  # strip leading "!"
        task = asyncio.create_task(
            _capture_bash_output(bot, user_id, thread_id, window_id, bash_cmd)
        )
        task.add_done_callback(task_done_callback)
        _bash_capture_tasks[(user_id, thread_id)] = task

    # If in interactive mode, refresh the UI after sending text
    interactive_window = get_interactive_window(user_id, thread_id)
    if interactive_window and interactive_window == window_id:
        await asyncio.sleep(0.2)
        await handle_interactive_ui(bot, user_id, window_id, thread_id)


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Orchestrate text message handling via bool early-return chain.

    Called after auth validation in bot.py's text_handler.
    """
    user = update.effective_user
    message = update.message
    assert user is not None  # guaranteed by caller
    assert message is not None and message.text  # guaranteed by caller

    text = message.text

    # BRAIN FORK: enrich with reply context (quoted message + attachments)
    reply = message.reply_to_message
    if reply:
        reply_parts = []
        # Sender info
        reply_user = reply.from_user
        sender = reply_user.first_name if reply_user and not reply_user.is_bot else "Fred"
        reply_parts.append(f"[Replying to {sender}]")
        # Text content
        if reply.text:
            reply_text = reply.text[:2000] + ("..." if len(reply.text) > 2000 else "")
            reply_parts.append(reply_text)
        elif reply.caption:
            reply_parts.append(reply.caption[:1000])
        # Download attachments to .ccgram-uploads/ for Claude to read
        try:
            upload_dir = None
            _tid_temp = _get_thread_id(update)
            if _tid_temp is not None:
                _wid = session_manager.get_window_for_thread(user.id, _tid_temp)
                if _wid:
                    _ws = session_manager.get_window_state(_wid)
                    if _ws.cwd:
                        upload_dir = Path(_ws.cwd) / ".ccgram-uploads"
            if reply.photo and upload_dir:
                photo = reply.photo[-1]  # highest resolution
                upload_dir.mkdir(parents=True, exist_ok=True)
                f = await photo.get_file()
                fname = f"reply_photo_{photo.file_unique_id}.jpg"
                await f.download_to_drive(upload_dir / fname)
                reply_parts.append(f"[Photo saved: .ccgram-uploads/{fname}]")
            if reply.document and upload_dir:
                doc = reply.document
                fname = doc.file_name or f"reply_doc_{doc.file_unique_id}"
                upload_dir.mkdir(parents=True, exist_ok=True)
                f = await doc.get_file()
                await f.download_to_drive(upload_dir / fname)
                reply_parts.append(f"[Document saved: .ccgram-uploads/{fname}]")
            if reply.voice and upload_dir:
                voice = reply.voice
                upload_dir.mkdir(parents=True, exist_ok=True)
                f = await voice.get_file()
                fname = f"reply_voice_{voice.file_unique_id}.ogg"
                await f.download_to_drive(upload_dir / fname)
                reply_parts.append(f"[Voice saved: .ccgram-uploads/{fname}]")
            if reply.video and upload_dir:
                video = reply.video
                upload_dir.mkdir(parents=True, exist_ok=True)
                f = await video.get_file()
                fname = f"reply_video_{video.file_unique_id}.mp4"
                await f.download_to_drive(upload_dir / fname)
                reply_parts.append(f"[Video saved: .ccgram-uploads/{fname}]")
            if reply.audio and upload_dir:
                audio = reply.audio
                upload_dir.mkdir(parents=True, exist_ok=True)
                f = await audio.get_file()
                fname = audio.file_name or f"reply_audio_{audio.file_unique_id}.mp3"
                await f.download_to_drive(upload_dir / fname)
                reply_parts.append(f"[Audio saved: .ccgram-uploads/{fname}]")
        except Exception as e:
            logger.debug("Failed to download reply attachment: %s", e)
        if reply_parts:
            reply_context = "\n".join(reply_parts)
            text = f"{reply_context}\n\n{text}"

    thread_id = _get_thread_id(update)

    # Store group chat_id for forum topic message routing
    chat = message.chat
    if chat.type in ("group", "supergroup") and thread_id is not None:
        session_manager.set_group_chat_id(user.id, thread_id, chat.id)

    # UI guards (window picker / directory browser active)
    if await _check_ui_guards(context.user_data, thread_id, message):
        return

    # BRAIN FORK (patch 59): intercept "Your own answer" free-text reply.
    # User clicked "Your own answer" → Tab sent to tmux → amend-mode flag set.
    # Next text message finalizes the interactive message with "Selected: <text>",
    # then falls through so the text still reaches Fred's TUI amend mode.
    if context.user_data and context.user_data.get(AMEND_STATE_KEY) == STATE_AMENDING_ANSWER:
        amend_ikey = context.user_data.pop(AMEND_IKEY_KEY, None)
        context.user_data.pop(AMEND_STATE_KEY, None)
        if amend_ikey:
            await finalize_interactive_msg(
                user_id=amend_ikey[0],
                bot=context.bot,
                thread_id=amend_ikey[1] or None,
                result=f"✓︎ Selected: {text}",
            )

    # Must be in a named topic
    if thread_id is None:
        if message and update.effective_chat and is_general_topic(message):
            await handle_general_topic_message(
                context.bot, message, update.effective_chat.id
            )
        else:
            await safe_reply(
                message,
                "\u274c Please use a named topic. Create a new topic to start a session.",
            )
        return

    # Auto-bind from topic presets (no UI needed)
    if await _try_auto_bind_from_preset(
        user.id, thread_id, text, message, context.bot
    ):
        return

    # Unbound topic — show picker or browser
    if await _handle_unbound_topic(
        user.id, thread_id, text, context.user_data, message
    ):
        return

    # Bound topic — check if window is still alive
    window_id = session_manager.get_window_for_thread(user.id, thread_id)
    assert window_id is not None  # _handle_unbound_topic returned False

    if await _handle_dead_window(
        window_id, user.id, thread_id, text, context.user_data, message
    ):
        return

    # Shell provider: route through LLM or raw execution
    provider = get_provider_for_window(window_id)
    if provider.capabilities.name == "shell":
        from .shell_commands import handle_shell_message

        await handle_shell_message(
            context.bot, user.id, thread_id, window_id, text, message
        )
        return

    # TEMP DEBUG: log Telegram message.date for timing analysis
    # BRAIN FORK: add forward metadata (patch 35)
    is_forward = message.forward_origin is not None
    if is_forward:
        origin = message.forward_origin
        sender = ""
        if hasattr(origin, "sender_user") and origin.sender_user:
            sender = origin.sender_user.first_name
        elif hasattr(origin, "sender_user_name") and origin.sender_user_name:
            sender = origin.sender_user_name
        elif hasattr(origin, "chat") and origin.chat:
            sender = origin.chat.title or ""
        text = f"[Forwarded from {sender}]\n{text}" if sender else f"[Forwarded]\n{text}"

    # Forward message to window (uses msg_batcher internally for 1s debounce)
    await _forward_message(window_id, user.id, thread_id, text, context.bot, message)
