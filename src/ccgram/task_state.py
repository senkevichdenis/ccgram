"""BRAIN FORK: Task* state tracker with debounce for live todo-list rendering.

Claude Code 2.1.84+ Task* tools (TaskCreate/TaskUpdate/TaskDelete/TaskList)
fire ONE call per task. This module accumulates per-thread state, then the
handler layer renders a single checklist after 500ms of quiet instead of
flickering one line per call.

Pure logic. No Telegram dependencies. The handler layer owns:
- decoding __TASK_EVENT__ markers from transcript_parser
- sending/editing status messages
- promoting the checklist to a persistent message on all-completed
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import structlog

logger = structlog.get_logger()

DEBOUNCE_SECONDS: float = 0.5

StateKey = tuple[int, int, str]  # (user_id, thread_id_or_0, window_id)

_ICONS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_MAX_LINE = 80
_ELLIPSIS = "\u2026"


@dataclass
class TaskInfo:
    subject: str = ""
    active_form: str = ""
    status: str = "pending"


@dataclass
class TaskState:
    tasks: "OrderedDict[str, TaskInfo]" = field(default_factory=OrderedDict)
    timer: Optional[asyncio.Task] = None
    list_rendered: bool = False


_states: dict[StateKey, TaskState] = {}


def _get_or_create(key: StateKey) -> TaskState:
    state = _states.get(key)
    if state is None:
        state = TaskState()
        _states[key] = state
    return state


def on_event(key: StateKey, event: dict) -> None:
    """Apply a create/update/delete/assign event to state. 'list' is a no-op.

    Task* tools in Claude Code 2.1.84+ do not include the task id in the
    tool_use input — it is assigned by the runtime and returned in the
    tool_result. The parser stores new TaskCreate tasks keyed by the
    tool_use id (`tuid`) and later emits op=assign to re-key the task to
    its real id. TaskUpdate/TaskDelete arrive with the real id in `taskId`.
    """
    op = event.get("op")
    if op == "list":
        return

    state = _get_or_create(key)

    if op == "assign":
        tuid = str(event.get("tuid", "") or "").strip()
        real_id = str(event.get("id", "") or "").strip()
        if not tuid or not real_id or tuid not in state.tasks:
            return
        if real_id == tuid:
            return
        # Rekey while preserving insertion order
        new_tasks: "OrderedDict[str, TaskInfo]" = OrderedDict()
        for tid, info in state.tasks.items():
            new_tasks[real_id if tid == tuid else tid] = info
        state.tasks = new_tasks
        return

    tuid = str(event.get("tuid", "") or "").strip()
    task_id = str(event.get("id", "") or "").strip()
    lookup_key = task_id or tuid
    if not lookup_key:
        return

    if op == "create":
        if lookup_key in state.tasks:
            return
        state.tasks[lookup_key] = TaskInfo(
            subject=str(event.get("subject", "") or "").strip(),
            active_form=str(event.get("activeForm", "") or "").strip(),
            status=str(event.get("status", "pending") or "pending"),
        )
    elif op == "update":
        new_status = event.get("status")
        # Fred's real Task system uses status="deleted" to remove a task
        # (not a dedicated TaskDelete op). Treat as delete.
        if new_status == "deleted":
            state.tasks.pop(lookup_key, None)
            return
        info = state.tasks.get(lookup_key)
        if info is None:
            return
        if "subject" in event and event["subject"] is not None:
            info.subject = str(event["subject"]).strip()
        if "activeForm" in event and event["activeForm"] is not None:
            info.active_form = str(event["activeForm"]).strip()
        if new_status:
            info.status = str(new_status)
    elif op == "delete":
        state.tasks.pop(lookup_key, None)


def render_list(key: StateKey) -> str:
    """Return the checklist as plain text, or '' if no tasks."""
    state = _states.get(key)
    if not state or not state.tasks:
        return ""
    lines = ["Todo list"]
    for info in state.tasks.values():
        if info.status == "in_progress" and info.active_form:
            text = info.active_form
        else:
            text = info.subject
        text = text.strip()
        if not text:
            continue
        if len(text) > _MAX_LINE:
            text = text[: _MAX_LINE - 1] + _ELLIPSIS
        icon = _ICONS.get(info.status, "[ ]")
        lines.append(f"{icon} {text}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def all_completed(key: StateKey) -> bool:
    """True iff state has tasks AND every task is completed."""
    state = _states.get(key)
    if not state or not state.tasks:
        return False
    return all(info.status == "completed" for info in state.tasks.values())


def has_tasks(key: StateKey) -> bool:
    state = _states.get(key)
    return bool(state and state.tasks)


def was_rendered(key: StateKey) -> bool:
    state = _states.get(key)
    return bool(state and state.list_rendered)


def mark_rendered(key: StateKey, rendered: bool = True) -> None:
    state = _states.get(key)
    if state:
        state.list_rendered = rendered


def schedule_render(
    key: StateKey,
    callback: Callable[[], Awaitable[None]],
) -> None:
    """Start/reset debounce timer. Cancels the previous one if still pending."""
    state = _get_or_create(key)
    old = state.timer
    if old is not None and not old.done():
        old.cancel()

    async def _fire() -> None:
        try:
            await asyncio.sleep(DEBOUNCE_SECONDS)
            await callback()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("task_state render callback failed: %s", exc)

    state.timer = asyncio.create_task(_fire())


def cancel_timer(key: StateKey) -> None:
    """Cancel a pending debounce timer without clearing task state."""
    state = _states.get(key)
    if state and state.timer is not None and not state.timer.done():
        state.timer.cancel()
        state.timer = None


def clear(key: StateKey) -> None:
    """Remove state for one key, cancelling any pending timer."""
    state = _states.pop(key, None)
    if state and state.timer is not None and not state.timer.done():
        state.timer.cancel()


def clear_for_thread(user_id: int, thread_id: int) -> None:
    """Remove every state belonging to (user_id, thread_id), any window_id."""
    keys = [k for k in list(_states.keys()) if k[0] == user_id and k[1] == thread_id]
    for k in keys:
        clear(k)


def snapshot(key: StateKey) -> list[tuple[str, str, str, str]]:
    """Debug/test helper: list of (id, subject, active_form, status) in order."""
    state = _states.get(key)
    if not state:
        return []
    return [
        (tid, info.subject, info.active_form, info.status)
        for tid, info in state.tasks.items()
    ]


def _reset_all() -> None:
    """Test-only: cancel all timers, drop all state."""
    for state in list(_states.values()):
        if state.timer is not None and not state.timer.done():
            state.timer.cancel()
    _states.clear()
