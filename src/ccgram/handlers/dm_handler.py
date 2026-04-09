"""BRAIN FORK: DM (private chat) handler.

Handles /start in DM (tracks in brain.bot_user_starts for reminder delivery),
text messages in DM (routes to user's context via Claude Code),
and my_chat_member events (tracks block/unblock).

DM works like General topic: CWD=/home/agent, same permissions as General.
Each user gets their own tmux window for DM (virtual thread binding).

Patches 38-40.
"""

import os
import structlog

import httpx
from telegram import Update, ChatMemberUpdated
from telegram.ext import ContextTypes

from ..config import config

logger = structlog.get_logger()

# --- Configuration ---

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# DM uses virtual thread_id per user to avoid conflicts with group threads
_DM_THREAD_OFFSET = 900000


def _detect_context() -> str:
    """Detect context name from BRAIN_CONTEXT or CLAUDE_CONFIG_DIR."""
    brain_ctx = os.getenv("BRAIN_CONTEXT", "")
    if brain_ctx:
        return brain_ctx
    config_dir = os.getenv("CLAUDE_CONFIG_DIR", "")
    # CLAUDE_CONFIG_DIR = /home/agent/contexts/{context}/config
    if "/contexts/" in config_dir:
        parts = config_dir.split("/contexts/")
        if len(parts) > 1:
            return parts[1].split("/")[0]
    return "unknown"


def _dm_thread_id(user_id: int) -> int:
    """Generate a unique virtual thread_id for DM sessions."""
    return _DM_THREAD_OFFSET + (user_id % 100000)


# --- Supabase helpers ---

async def _upsert_bot_start(user_id: int, context_id: str) -> bool:
    """Record /start in brain.bot_user_starts. Returns True on success."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("DM: no Supabase credentials, cannot track /start")
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/brain_upsert_bot_start",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"p_user_id": user_id, "p_context_id": context_id},
            )
            if resp.status_code < 300:
                logger.info("DM: /start recorded", user_id=user_id, context=context_id)
                return True
            logger.warning("DM: failed to record /start", status=resp.status_code, body=resp.text[:200])
            return False
    except Exception as e:
        logger.error("DM: error recording /start", error=str(e))
        return False


async def _update_bot_blocked(user_id: int, context_id: str, blocked: bool) -> None:
    """Update blocked_at in brain.bot_user_starts."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/brain_update_bot_blocked",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"p_user_id": user_id, "p_context_id": context_id, "p_blocked": blocked},
            )
    except Exception as e:
        logger.error("DM: error updating blocked status", error=str(e))


# --- Handlers ---

async def dm_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start in private chat. Track for DM delivery."""
    user = update.effective_user
    if not user or not update.message:
        return

    user_id = user.id

    if not config.is_user_allowed(user_id):
        # Not in ALLOWED_USERS: silently ignore.
        # Random Telegram users writing to the bot are completely ignored (anti-spam).
        logger.debug("DM: /start from unknown user, ignoring", user_id=user_id)
        return

    # Record /start
    await _upsert_bot_start(user_id, _detect_context())

    await update.message.reply_text(
        "Привет! Я на месте. Можешь писать мне сюда: "
        "напоминания, уведомления, вопросы."
    )

    logger.info("DM: /start handled", user_id=user_id, allowed=True, context=_detect_context())


async def dm_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages in private chat. Route to user context via tmux.

    DM works like General topic: CWD=/home/agent, same permissions.
    Each user gets their own tmux window for DM (virtual thread binding).
    """
    user = update.effective_user
    if not user or not update.message or not update.message.text:
        return

    user_id = user.id
    text = update.message.text
    message = update.message

    if not config.is_user_allowed(user_id):
        logger.debug("DM: text from unknown user, ignoring", user_id=user_id)
        return

    # Record /start implicitly
    await _upsert_bot_start(user_id, _detect_context())

    # RBAC check (same as group General topic)
    _brain_ctx = os.getenv("BRAIN_CONTEXT", "")
    if _brain_ctx:
        from ..rbac import check_access, generate_settings_local
        _access = await check_access(user_id, None)  # None = no specific project (like General)
        if not _access.allowed:
            await message.reply_text("У тебя нет доступа.")
            return
        if not _access.is_owner:
            await generate_settings_local(
                _access, "/home/agent",
                current_project_slug=None, current_project_path=None,
            )

        # Write current user for pre-action-check.sh
        _user_file = "/tmp/brain-current-user-" + _brain_ctx
        try:
            with open(_user_file, "w") as f:
                f.write("BRAIN_CURRENT_USER_ID=" + str(user_id) + "\n")
                f.write("BRAIN_CURRENT_USER_NAME=" + getattr(_access, "display_name", "unknown") + "\n")
                can_write = str(getattr(_access, "can_write", False)).lower()
                f.write("BRAIN_CURRENT_USER_CAN_WRITE=" + can_write + "\n")
                can_del = str(getattr(_access, "can_delete", False)).lower()
                f.write("BRAIN_CURRENT_USER_CAN_DELETE=" + can_del + "\n")
        except OSError:
            pass

    from ..session import session_manager
    from ..tmux_manager import tmux_manager
    from ..providers import resolve_launch_command
    from .message_sender import safe_reply

    # Virtual thread_id for DM
    thread_id = _dm_thread_id(user_id)

    # Check if already bound and window alive
    window_id = session_manager.get_window_for_thread(user_id, thread_id)

    # Verify window actually exists in tmux (may have been killed by OOM/health-check)
    if window_id is not None:
        w = await tmux_manager.find_window_by_id(window_id)
        if w is None:
            # Window dead, unbind and recreate
            logger.info("DM: window dead, recreating", window_id=window_id, user_id=user_id)
            session_manager.unbind_thread(user_id, thread_id)
            window_id = None

    if window_id is None:
        # Create new window for DM (like auto-bind, CWD=/home/agent)
        logger.info("DM: creating window for user", user_id=user_id)
        await message.chat.send_action("typing")

        launch_command = resolve_launch_command("claude", approval_mode="yolo")
        success, msg, wname, wid = await tmux_manager.create_window(
            "/home/agent", launch_command=launch_command
        )
        if not success:
            await safe_reply(message, "Не удалось создать сессию. Попробуй позже.")
            return

        session_manager.update_user_mru(user_id, "/home/agent")
        ws = session_manager.get_window_state(wid)
        ws.cwd = "/home/agent"
        ws.is_dm = True  # BRAIN FORK: DM window flag
        session_manager.set_window_provider(wid, "claude")
        session_manager.set_window_approval_mode(wid, "yolo")
        await tmux_manager.stamp_pane_title(wid, "claude")

        # Bind virtual thread
        session_manager.bind_thread(user_id, thread_id, wid, window_name=wname)
        window_id = wid

    # Forward message to tmux with DM marker
    # Fred sees "[DM]" prefix and knows this is a private conversation
    dm_text = "[DM] " + text
    send_ok, send_msg = await session_manager.send_to_window(window_id, dm_text)
    if not send_ok:
        await safe_reply(message, "Сессия временно недоступна. Попробуй через минуту.")
        logger.warning("DM: failed to send to window", error=send_msg, user_id=user_id)


async def my_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Track when user blocks/unblocks the bot."""
    if not update.my_chat_member:
        return

    member: ChatMemberUpdated = update.my_chat_member

    # Only track DM block/unblock, ignore group member events
    if member.chat.type != "private":
        return

    user = member.from_user
    if not user:
        return

    new_status = member.new_chat_member.status
    old_status = member.old_chat_member.status if member.old_chat_member else None

    ctx = _detect_context()

    if new_status == "kicked":
        # User blocked the bot
        await _update_bot_blocked(user.id, ctx, blocked=True)
        logger.info("DM: user blocked bot", user_id=user.id, context=ctx)

        # Notify owner in group (Errors topic)
        _group_id = config.group_id
        _errors_topic = None
        _topics_path = os.path.expanduser("~/scripts/.env.topics." + ctx)

        if os.path.isfile(_topics_path):
            with open(_topics_path) as f:
                for line in f:
                    if line.startswith("TOPIC_ERRORS="):
                        try:
                            _errors_topic = int(line.strip().split("=", 1)[1])
                        except ValueError:
                            pass

        if _group_id:
            display = "@" + user.username if user.username else user.full_name
            msg = display + " заблокировал бота. DM уведомления для него отключены."
            try:
                kwargs: dict = {"chat_id": _group_id, "text": msg}
                if _errors_topic and _errors_topic > 1:
                    kwargs["message_thread_id"] = _errors_topic
                await context.bot.send_message(**kwargs)
            except Exception as e:
                logger.error("DM: failed to notify owner about block", error=str(e))

    elif old_status == "kicked" and new_status in ("member", "restricted"):
        # User unblocked the bot
        await _update_bot_blocked(user.id, ctx, blocked=False)
        logger.info("DM: user unblocked bot", user_id=user.id, context=ctx)
