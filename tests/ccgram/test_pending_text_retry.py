"""Tests for BRAIN FORK pending-text retry in _try_auto_bind_from_preset.

Reproduces den-context incident 2026-05-03 06:49:
  Failed to forward preset pending text: Сессия загружается. Попробуй через минуту.

Without retry, user's first message after auto-bind is silently dropped
(only a warning log line). After fix: 3 attempts with 1.5s/3s backoff,
then explicit user notification.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ccgram.handlers.text_handler import _try_auto_bind_from_preset


@pytest.fixture
def fake_message():
    """Minimal Telegram Message stub for safe_reply."""
    msg = AsyncMock()
    msg.chat = AsyncMock()
    msg.chat.type = "supergroup"
    msg.chat.id = 12345
    return msg


class TestPendingTextRetry:
    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self, fake_message):
        """No retry needed when send succeeds on first attempt."""
        with patch(
            "ccgram.handlers.text_handler.auto_bind_window_for_preset",
            new=AsyncMock(return_value="@1"),
        ), patch(
            "ccgram.handlers.text_handler.session_manager.send_to_window",
            new=AsyncMock(return_value=(True, "")),
        ) as send_mock:
            result = await _try_auto_bind_from_preset(
                user_id=1, thread_id=1, text="hi", message=fake_message, bot=None
            )

        assert result is True
        assert send_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, fake_message):
        """Reproduces 2026-05-03 06:49: bootstrap takes ~1.5s, second attempt
        succeeds after retry delay.
        """
        # Fail first, succeed second
        send_mock = AsyncMock(side_effect=[
            (False, "Сессия загружается. Попробуй через минуту."),
            (True, ""),
        ])
        with patch(
            "ccgram.handlers.text_handler.auto_bind_window_for_preset",
            new=AsyncMock(return_value="@1"),
        ), patch(
            "ccgram.handlers.text_handler.session_manager.send_to_window",
            new=send_mock,
        ), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await _try_auto_bind_from_preset(
                user_id=1, thread_id=1, text="hi", message=fake_message, bot=None
            )

        assert result is True
        assert send_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_all_three_attempts_fail_notifies_user(self, fake_message):
        """If all 3 retries fail, user MUST be notified (not silently dropped)."""
        send_mock = AsyncMock(return_value=(False, "Сессия загружается."))
        safe_reply_mock = AsyncMock()

        with patch(
            "ccgram.handlers.text_handler.auto_bind_window_for_preset",
            new=AsyncMock(return_value="@1"),
        ), patch(
            "ccgram.handlers.text_handler.session_manager.send_to_window",
            new=send_mock,
        ), patch(
            "asyncio.sleep", new=AsyncMock()
        ), patch(
            "ccgram.handlers.text_handler.safe_reply",
            new=safe_reply_mock,
        ):
            result = await _try_auto_bind_from_preset(
                user_id=1, thread_id=1, text="hi", message=fake_message, bot=None
            )

        assert result is True  # handled (don't fall through to other handlers)
        assert send_mock.await_count == 3, "must retry 3 times before giving up"
        safe_reply_mock.assert_awaited_once()
        # Verify user-facing message mentions the issue
        call_args = safe_reply_mock.await_args
        notification_text = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("text", "")
        assert "Сессия" in notification_text or "сообщение" in notification_text.lower() or "не доставлено" in notification_text.lower()

    @pytest.mark.asyncio
    async def test_no_preset_returns_false(self, fake_message):
        """When auto_bind_window_for_preset returns None (no preset), the
        wrapper returns False to let other handlers try.
        """
        with patch(
            "ccgram.handlers.text_handler.auto_bind_window_for_preset",
            new=AsyncMock(return_value=None),
        ):
            result = await _try_auto_bind_from_preset(
                user_id=1, thread_id=1, text="hi", message=fake_message, bot=None
            )

        assert result is False
