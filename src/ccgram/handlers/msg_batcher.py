"""BRAIN FORK: Message batcher (patch 35).

1s debounce on all messages to tmux. Batches forward + comment,
multiple forwards, photo + text into one combined message.
Used by text_handler and file_handler.
"""

import asyncio
import structlog

from ..session import session_manager

logger = structlog.get_logger()

_buffers: dict[str, list[str]] = {}  # window_id -> [texts]
_timers: dict[str, asyncio.Task] = {}
_DEBOUNCE_SECONDS = 1.0


async def enqueue(window_id: str, text: str) -> None:
    """Add message to buffer, reset 1s debounce timer."""
    if window_id not in _buffers:
        _buffers[window_id] = []
    _buffers[window_id].append(text)

    # Cancel existing timer
    old_timer = _timers.pop(window_id, None)
    if old_timer and not old_timer.done():
        old_timer.cancel()

    # Start new timer
    async def _flush(wid=window_id):
        await asyncio.sleep(_DEBOUNCE_SECONDS)
        await flush(wid)
    _timers[window_id] = asyncio.create_task(_flush())


async def flush(window_id: str) -> None:
    """Flush buffer: combine all texts and send to tmux."""
    texts = _buffers.pop(window_id, [])
    _timers.pop(window_id, None)
    if not texts:
        return
    combined = "\n\n".join(texts)
    logger.info("Message batch flushed: %d messages, %d chars", len(texts), len(combined))
    success, err = await session_manager.send_to_window(window_id, combined)
    if not success:
        logger.warning("Batch send failed: %s", err)
