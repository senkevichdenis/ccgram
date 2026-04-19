"""Tests for ccgram.task_state — pure logic + asyncio debounce."""

from __future__ import annotations

import asyncio

import pytest

from ccgram import task_state


KEY = (42, 17, "cc@1")
KEY_OTHER_WIN = (42, 17, "cc@2")
KEY_OTHER_THREAD = (42, 99, "cc@1")


@pytest.fixture(autouse=True)
def _reset_state():
    task_state._reset_all()
    yield
    task_state._reset_all()


# ── on_event: create ─────────────────────────────────────────────────────


class TestOnEventCreate:
    def test_create_adds_task_with_defaults(self):
        task_state.on_event(
            KEY,
            {"op": "create", "id": "t1", "subject": "Write tests"},
        )
        snap = task_state.snapshot(KEY)
        assert snap == [("t1", "Write tests", "", "pending")]

    def test_create_preserves_all_fields(self):
        task_state.on_event(
            KEY,
            {
                "op": "create",
                "id": "t1",
                "subject": "Write tests",
                "activeForm": "Writing tests",
                "status": "pending",
            },
        )
        assert task_state.snapshot(KEY) == [
            ("t1", "Write tests", "Writing tests", "pending")
        ]

    def test_create_duplicate_id_is_noop(self):
        task_state.on_event(
            KEY, {"op": "create", "id": "t1", "subject": "First"}
        )
        task_state.on_event(
            KEY, {"op": "create", "id": "t1", "subject": "Second"}
        )
        assert task_state.snapshot(KEY) == [("t1", "First", "", "pending")]

    def test_create_preserves_insertion_order(self):
        for i in range(5):
            task_state.on_event(
                KEY, {"op": "create", "id": f"t{i}", "subject": f"Task {i}"}
            )
        ids = [row[0] for row in task_state.snapshot(KEY)]
        assert ids == ["t0", "t1", "t2", "t3", "t4"]

    def test_create_without_id_is_ignored(self):
        task_state.on_event(KEY, {"op": "create", "subject": "Nope"})
        task_state.on_event(KEY, {"op": "create", "id": "", "subject": "Also nope"})
        assert task_state.snapshot(KEY) == []


# ── on_event: update ─────────────────────────────────────────────────────


class TestOnEventUpdate:
    def _seed(self):
        task_state.on_event(
            KEY,
            {
                "op": "create",
                "id": "t1",
                "subject": "Write tests",
                "activeForm": "Writing tests",
            },
        )

    def test_update_changes_status(self):
        self._seed()
        task_state.on_event(
            KEY, {"op": "update", "id": "t1", "status": "in_progress"}
        )
        assert task_state.snapshot(KEY)[0][3] == "in_progress"

    def test_update_changes_subject(self):
        self._seed()
        task_state.on_event(
            KEY, {"op": "update", "id": "t1", "subject": "Refactored subject"}
        )
        assert task_state.snapshot(KEY)[0][1] == "Refactored subject"

    def test_update_unknown_id_is_noop(self):
        self._seed()
        task_state.on_event(
            KEY, {"op": "update", "id": "unknown", "status": "completed"}
        )
        assert task_state.snapshot(KEY)[0][3] == "pending"

    def test_update_none_values_ignored(self):
        self._seed()
        task_state.on_event(
            KEY,
            {
                "op": "update",
                "id": "t1",
                "subject": None,
                "activeForm": None,
                "status": "in_progress",
            },
        )
        row = task_state.snapshot(KEY)[0]
        assert row[1] == "Write tests"
        assert row[2] == "Writing tests"
        assert row[3] == "in_progress"


# ── on_event: delete ─────────────────────────────────────────────────────


class TestOnEventDelete:
    def test_delete_removes_task(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY, {"op": "create", "id": "t2", "subject": "B"})
        task_state.on_event(KEY, {"op": "delete", "id": "t1"})
        assert [row[0] for row in task_state.snapshot(KEY)] == ["t2"]

    def test_delete_unknown_id_is_noop(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY, {"op": "delete", "id": "ghost"})
        assert [row[0] for row in task_state.snapshot(KEY)] == ["t1"]


# ── on_event: list (read-only) ───────────────────────────────────────────


class TestOnEventList:
    def test_list_does_not_mutate_state(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY, {"op": "list"})
        assert task_state.snapshot(KEY) == [("t1", "A", "", "pending")]


# ── render_list ──────────────────────────────────────────────────────────


class TestRenderList:
    def test_empty_state_returns_empty_string(self):
        assert task_state.render_list(KEY) == ""

    def test_renders_header_and_items(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "One"})
        task_state.on_event(KEY, {"op": "create", "id": "t2", "subject": "Two"})
        out = task_state.render_list(KEY)
        assert out == "Todo list\n[ ] One\n[ ] Two"

    def test_icons_match_statuses(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "P"})
        task_state.on_event(KEY, {"op": "create", "id": "t2", "subject": "I"})
        task_state.on_event(KEY, {"op": "create", "id": "t3", "subject": "C"})
        task_state.on_event(KEY, {"op": "update", "id": "t2", "status": "in_progress"})
        task_state.on_event(KEY, {"op": "update", "id": "t3", "status": "completed"})
        out = task_state.render_list(KEY)
        assert "[ ] P" in out
        assert "[~] I" in out
        assert "[x] C" in out

    def test_in_progress_uses_active_form(self):
        task_state.on_event(
            KEY,
            {
                "op": "create",
                "id": "t1",
                "subject": "Fix bug",
                "activeForm": "Fixing bug",
            },
        )
        task_state.on_event(KEY, {"op": "update", "id": "t1", "status": "in_progress"})
        out = task_state.render_list(KEY)
        assert "Fixing bug" in out
        assert "Fix bug" not in out

    def test_pending_uses_subject_even_if_active_form_set(self):
        task_state.on_event(
            KEY,
            {
                "op": "create",
                "id": "t1",
                "subject": "Fix bug",
                "activeForm": "Fixing bug",
            },
        )
        out = task_state.render_list(KEY)
        assert "Fix bug" in out
        assert "Fixing bug" not in out

    def test_long_subject_truncated(self):
        long = "x" * 200
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": long})
        out = task_state.render_list(KEY)
        assert "\u2026" in out
        for line in out.splitlines():
            assert len(line) <= 85


# ── all_completed / has_tasks ────────────────────────────────────────────


class TestAggregates:
    def test_all_completed_false_for_empty(self):
        assert task_state.all_completed(KEY) is False

    def test_all_completed_false_when_any_pending(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY, {"op": "create", "id": "t2", "subject": "B"})
        task_state.on_event(KEY, {"op": "update", "id": "t1", "status": "completed"})
        assert task_state.all_completed(KEY) is False

    def test_all_completed_true_when_every_task_done(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY, {"op": "create", "id": "t2", "subject": "B"})
        task_state.on_event(KEY, {"op": "update", "id": "t1", "status": "completed"})
        task_state.on_event(KEY, {"op": "update", "id": "t2", "status": "completed"})
        assert task_state.all_completed(KEY) is True

    def test_has_tasks(self):
        assert task_state.has_tasks(KEY) is False
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        assert task_state.has_tasks(KEY) is True


# ── rendered flag ────────────────────────────────────────────────────────


class TestRenderedFlag:
    def test_mark_rendered_requires_state(self):
        task_state.mark_rendered(KEY)
        assert task_state.was_rendered(KEY) is False

    def test_mark_rendered_toggles(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        assert task_state.was_rendered(KEY) is False
        task_state.mark_rendered(KEY)
        assert task_state.was_rendered(KEY) is True
        task_state.mark_rendered(KEY, False)
        assert task_state.was_rendered(KEY) is False


# ── isolation across keys ────────────────────────────────────────────────


class TestIsolation:
    def test_different_windows_are_isolated(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY_OTHER_WIN, {"op": "create", "id": "t1", "subject": "B"})
        assert task_state.snapshot(KEY)[0][1] == "A"
        assert task_state.snapshot(KEY_OTHER_WIN)[0][1] == "B"

    def test_different_threads_are_isolated(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(
            KEY_OTHER_THREAD, {"op": "create", "id": "t1", "subject": "B"}
        )
        assert task_state.snapshot(KEY)[0][1] == "A"
        assert task_state.snapshot(KEY_OTHER_THREAD)[0][1] == "B"


# ── clear ────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_drops_state(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.clear(KEY)
        assert task_state.snapshot(KEY) == []
        assert task_state.has_tasks(KEY) is False

    def test_clear_for_thread_drops_all_windows(self):
        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.on_event(KEY_OTHER_WIN, {"op": "create", "id": "t1", "subject": "B"})
        task_state.on_event(
            KEY_OTHER_THREAD, {"op": "create", "id": "t1", "subject": "C"}
        )
        task_state.clear_for_thread(42, 17)
        assert task_state.has_tasks(KEY) is False
        assert task_state.has_tasks(KEY_OTHER_WIN) is False
        assert task_state.has_tasks(KEY_OTHER_THREAD) is True


# ── tuid placeholder + assign rekey ──────────────────────────────────────


class TestTuidAndAssign:
    """Tasks created with tuid placeholder get re-keyed on op=assign."""

    def test_create_uses_tuid_when_id_missing(self):
        task_state.on_event(
            KEY,
            {"op": "create", "tuid": "toolu_abc", "subject": "X"},
        )
        assert task_state.snapshot(KEY) == [("toolu_abc", "X", "", "pending")]

    def test_assign_rekeys_tuid_to_real_id(self):
        task_state.on_event(
            KEY, {"op": "create", "tuid": "toolu_abc", "subject": "X"}
        )
        task_state.on_event(
            KEY, {"op": "assign", "tuid": "toolu_abc", "id": "5"}
        )
        assert task_state.snapshot(KEY) == [("5", "X", "", "pending")]

    def test_assign_preserves_insertion_order(self):
        task_state.on_event(KEY, {"op": "create", "tuid": "tuid_a", "subject": "A"})
        task_state.on_event(KEY, {"op": "create", "tuid": "tuid_b", "subject": "B"})
        task_state.on_event(KEY, {"op": "create", "tuid": "tuid_c", "subject": "C"})
        task_state.on_event(KEY, {"op": "assign", "tuid": "tuid_b", "id": "2"})
        ids = [row[0] for row in task_state.snapshot(KEY)]
        assert ids == ["tuid_a", "2", "tuid_c"]

    def test_update_finds_task_after_assign(self):
        task_state.on_event(KEY, {"op": "create", "tuid": "toolu_xyz", "subject": "X"})
        task_state.on_event(KEY, {"op": "assign", "tuid": "toolu_xyz", "id": "7"})
        task_state.on_event(KEY, {"op": "update", "id": "7", "status": "completed"})
        assert task_state.snapshot(KEY)[0][3] == "completed"

    def test_assign_with_unknown_tuid_is_noop(self):
        task_state.on_event(KEY, {"op": "create", "tuid": "tuid_a", "subject": "A"})
        task_state.on_event(KEY, {"op": "assign", "tuid": "tuid_unknown", "id": "99"})
        assert task_state.snapshot(KEY) == [("tuid_a", "A", "", "pending")]

    def test_assign_with_same_id_is_noop(self):
        task_state.on_event(KEY, {"op": "create", "id": "5", "subject": "A"})
        task_state.on_event(KEY, {"op": "assign", "tuid": "5", "id": "5"})
        assert task_state.snapshot(KEY) == [("5", "A", "", "pending")]

    def test_update_with_status_deleted_removes_task(self):
        task_state.on_event(KEY, {"op": "create", "id": "1", "subject": "A"})
        task_state.on_event(KEY, {"op": "create", "id": "2", "subject": "B"})
        task_state.on_event(KEY, {"op": "update", "id": "1", "status": "deleted"})
        assert [row[0] for row in task_state.snapshot(KEY)] == ["2"]


# ── debounce / timer (async) ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestDebounce:
    async def test_timer_fires_after_debounce(self, monkeypatch):
        monkeypatch.setattr(task_state, "DEBOUNCE_SECONDS", 0.05)
        calls: list[int] = []

        async def cb():
            calls.append(1)

        task_state.schedule_render(KEY, cb)
        await asyncio.sleep(0.15)
        assert calls == [1]

    async def test_second_schedule_cancels_first(self, monkeypatch):
        monkeypatch.setattr(task_state, "DEBOUNCE_SECONDS", 0.08)
        calls: list[str] = []

        async def cb1():
            calls.append("first")

        async def cb2():
            calls.append("second")

        task_state.schedule_render(KEY, cb1)
        await asyncio.sleep(0.03)
        task_state.schedule_render(KEY, cb2)
        await asyncio.sleep(0.15)
        assert calls == ["second"]

    async def test_clear_cancels_pending_timer(self, monkeypatch):
        monkeypatch.setattr(task_state, "DEBOUNCE_SECONDS", 0.1)
        calls: list[int] = []

        async def cb():
            calls.append(1)

        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.schedule_render(KEY, cb)
        await asyncio.sleep(0.02)
        task_state.clear(KEY)
        await asyncio.sleep(0.15)
        assert calls == []

    async def test_clear_for_thread_cancels_timer(self, monkeypatch):
        monkeypatch.setattr(task_state, "DEBOUNCE_SECONDS", 0.1)
        calls: list[int] = []

        async def cb():
            calls.append(1)

        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.schedule_render(KEY, cb)
        await asyncio.sleep(0.02)
        task_state.clear_for_thread(42, 17)
        await asyncio.sleep(0.15)
        assert calls == []

    async def test_cancel_timer_keeps_state(self, monkeypatch):
        monkeypatch.setattr(task_state, "DEBOUNCE_SECONDS", 0.1)

        async def cb():
            pass

        task_state.on_event(KEY, {"op": "create", "id": "t1", "subject": "A"})
        task_state.schedule_render(KEY, cb)
        task_state.cancel_timer(KEY)
        assert task_state.has_tasks(KEY) is True

