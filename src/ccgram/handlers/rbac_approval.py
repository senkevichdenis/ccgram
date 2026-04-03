"""BRAIN FORK: Auto-approve handler for new users.

When an unknown user writes to the group, sends an inline keyboard
to the owner private chat. On Approve, adds user to Supabase with
basic permissions and to in-memory ALLOWED_USERS.
"""

import os
import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..rbac import BRAIN_CONTEXT, SUPABASE_URL, SUPABASE_KEY

logger = structlog.get_logger()

# Callback data prefix
CB_RBAC_APPROVE = "rbac_approve:"
CB_RBAC_DENY = "rbac_deny:"

# Owner telegram_id (first in ALLOWED_USERS = owner)
_OWNER_ID: int | None = None


def get_owner_id() -> int | None:
    """Get owner telegram_id from ALLOWED_USERS (first entry = owner)."""
    global _OWNER_ID
    if _OWNER_ID is not None:
        return _OWNER_ID
    allowed = os.getenv("ALLOWED_USERS", "")
    if allowed:
        try:
            _OWNER_ID = int(allowed.split(",")[0].strip())
        except ValueError:
            pass
    return _OWNER_ID


async def send_approval_request(bot: Bot, user_id: int, username: str | None, full_name: str) -> bool:
    """Send approval request to owner private chat.

    Returns True if request was sent, False if owner chat not available.
    """
    owner_id = get_owner_id()
    if not owner_id:
        logger.warning("RBAC: no owner_id configured, cannot send approval")
        return False

    display = f"@{username}" if username else full_name
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Approve", callback_data=f"{CB_RBAC_APPROVE}{user_id}:{full_name}"),
            InlineKeyboardButton("Deny", callback_data=f"{CB_RBAC_DENY}{user_id}:{full_name}"),
        ]
    ])

    try:
        await bot.send_message(
            chat_id=owner_id,
            text=f"Новый пользователь {display} (ID: {user_id}) пишет в группу.\nОдобрить?",
            reply_markup=keyboard,
        )
        logger.info("RBAC approval request sent", user_id=user_id, display=display)
        return True
    except Exception as e:
        logger.error("RBAC: failed to send approval to owner", error=str(e))
        return False


async def handle_approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Approve/Deny button press from owner."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    data = query.data
    is_approve = data.startswith(CB_RBAC_APPROVE)

    # Parse: rbac_approve:USER_ID:FULL_NAME
    prefix = CB_RBAC_APPROVE if is_approve else CB_RBAC_DENY
    payload = data[len(prefix):]
    parts = payload.split(":", 1)
    if len(parts) < 2:
        return

    try:
        new_user_id = int(parts[0])
    except ValueError:
        return
    full_name = parts[1]

    if is_approve:
        # Add to Supabase with basic permissions
        success = await _add_user_to_supabase(new_user_id, full_name)
        if success:
            # Add to in-memory ALLOWED_USERS
            from ..config import config
            config.allowed_users.add(new_user_id)
            # Also append to .env file for persistence
            _append_to_allowed_users(new_user_id)

            await query.edit_message_text(f"Approved: {full_name} (ID: {new_user_id}). Базовые права выданы (discuss, code:read).")
            logger.info("RBAC: user approved", user_id=new_user_id, name=full_name)
        else:
            await query.edit_message_text(f"Ошибка при добавлении {full_name}. Проверь логи.")
    else:
        await query.edit_message_text(f"Denied: {full_name} (ID: {new_user_id}).")
        logger.info("RBAC: user denied", user_id=new_user_id, name=full_name)


async def _add_user_to_supabase(telegram_id: int, full_name: str) -> bool:
    """Add new user to brain.agent_users with basic green permissions."""
    if not SUPABASE_URL or not SUPABASE_KEY or not BRAIN_CONTEXT:
        return False

    import httpx
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    # Use display_name from full_name (lowercase, first word)
    display_name = full_name.lower().split()[0] if full_name else str(telegram_id)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Create user
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/check_brain_permission",
                headers=headers,
                json={"p_telegram_id": telegram_id, "p_context": BRAIN_CONTEXT},
            )
            # If user already exists, skip creation
            result = resp.json()
            if isinstance(result, dict) and result.get("allowed"):
                logger.info("RBAC: user already exists in Supabase", telegram_id=telegram_id)
                return True

            # Create via raw SQL through public wrapper
            # We need a create function. For now, use the REST API directly
            # Insert into agent_users
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/agent_users",
                headers={**headers, "Content-Profile": "brain", "Prefer": "return=representation"},
                json={
                    "telegram_id": telegram_id,
                    "display_name": display_name,
                    "is_owner": False,
                    "context": BRAIN_CONTEXT,
                },
            )

            if resp.status_code not in (200, 201):
                logger.error("RBAC: failed to create user", status=resp.status_code, body=resp.text)
                return False

            user_data = resp.json()
            user_id = user_data[0]["id"] if isinstance(user_data, list) else user_data.get("id")

            if not user_id:
                logger.error("RBAC: no user_id returned after insert")
                return False

            # Get green permission IDs
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/agent_permissions?danger_level=eq.green&select=id",
                headers={**headers, "Accept-Profile": "brain"},
            )
            if resp.status_code != 200:
                logger.error("RBAC: failed to get permissions", status=resp.status_code)
                return False

            perm_ids = [p["id"] for p in resp.json()]

            # Get owner user_id for granted_by
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/agent_users?display_name=eq.den&context=eq.{BRAIN_CONTEXT}&select=id",
                headers={**headers, "Accept-Profile": "brain"},
            )
            owner_uuid = resp.json()[0]["id"] if resp.status_code == 200 and resp.json() else None

            # Assign green permissions
            for perm_id in perm_ids:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/agent_user_permissions",
                    headers={**headers, "Content-Profile": "brain"},
                    json={
                        "user_id": user_id,
                        "permission_id": perm_id,
                        "granted_by": owner_uuid,
                    },
                )

            logger.info("RBAC: user created with green permissions", telegram_id=telegram_id, name=display_name, perms=len(perm_ids))
            return True

    except Exception as e:
        logger.error("RBAC: Supabase error during user creation", error=str(e))
        return False


def _append_to_allowed_users(telegram_id: int) -> None:
    """Append new user to .env ALLOWED_USERS for persistence across restarts."""
    ccgram_dir = os.getenv("CCGRAM_DIR", "")
    if not ccgram_dir:
        return
    env_path = os.path.join(ccgram_dir, ".env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith("ALLOWED_USERS="):
                current = line.strip().split("=", 1)[1]
                line = f"ALLOWED_USERS={current},{telegram_id}\n"
            new_lines.append(line)
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        logger.info("RBAC: appended to ALLOWED_USERS in .env", telegram_id=telegram_id)
    except Exception as e:
        logger.error("RBAC: failed to update .env", error=str(e))
