"""RED Approval Handler for CCGram."""
import logging
from pathlib import Path
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)
APPROVAL_DIR = Path("/tmp/brain-approvals")

async def handle_red_approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    query = update.callback_query
    if not query or not query.data:
        return False
    data = query.data
    if not data.startswith("red_approve:") and not data.startswith("red_deny:"):
        return False
    parts = data.split(":", 1)
    if len(parts) != 2:
        await query.answer("Invalid callback data")
        return True
    action = "approve" if data.startswith("red_approve:") else "deny"
    approval_id = parts[1]
    APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
    approval_file = APPROVAL_DIR / approval_id
    try:
        approval_file.write_text(action)
    except OSError as e:
        logger.error("Failed to write approval file %s: %s", approval_file, e)
        await query.answer("Error processing approval")
        return True
    user = query.from_user
    user_name = user.first_name if user else "Unknown"
    status = "Approved" if action == "approve" else "Denied"
    new_text = f"{query.message.text}\n\n{status} by {user_name}"
    try:
        await query.edit_message_text(text=new_text, reply_markup=None)
    except Exception as e:
        logger.debug("Failed to edit approval message: %s", e)
    await query.answer(f"Action {action}d")
    logger.info("RED approval: %s by %s (id: %s)", action, user_name, approval_id)
    return True
