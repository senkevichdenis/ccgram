"""Tests for BRAIN FORK orphan-session recovery + huge-transcript protection.

Reproduces den-context incidents from 2026-05-03:
  * 5x window death → preset auto-bind opened fresh empty session because
    prune_session_map deleted session_map record before next message arrived
    → find_resumable_args_for_path had nothing to recover from.
  * 6.78 MB transcript triggered 3x StopFailure (prompt > 200k tokens) on resume.

Also tests sequence: bug → fix path stays passing.
"""

from __future__ import annotations

import json
import time

import pytest

from ccgram.session import (
    MAX_RESUME_TRANSCRIPT_MB,
    SessionManager,
    WindowState,
    find_resumable_args_for_path,
)


@pytest.fixture
def mgr(monkeypatch) -> SessionManager:
    monkeypatch.setattr(SessionManager, "_load_state", lambda self: None)
    monkeypatch.setattr(SessionManager, "_save_state", lambda self: None)
    return SessionManager()


@pytest.fixture
def sm_files(tmp_path, monkeypatch):
    """Set up isolated session_map.json + state.json + a tmpdir for transcripts."""
    sm_file = tmp_path / "session_map.json"
    state_file = tmp_path / "state.json"
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr("ccgram.session.config.session_map_file", sm_file)
    monkeypatch.setattr("ccgram.session.config.config_dir", tmp_path)
    monkeypatch.setattr("ccgram.session.config.tmux_session_name", "den")
    return sm_file, state_file, transcripts


# ============================================================================
# FIX A — prune marks orphan, find_resumable recovers from orphan record
# ============================================================================


class TestPruneMarksOrphanInsteadOfDelete:
    def test_dead_entry_kept_with_orphan_flag(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Reproduces den 2026-05-03 06:17: @13 dies, prune kills session_map
        entry, next message at 06:19 has no record to resume → fresh empty.
        Fix: keep entry, mark _orphan=True, preserve transcript_path.
        """
        sm_file, _, transcripts = sm_files
        tp = transcripts / "abc.jsonl"
        tp.write_text('{"sessionId":"abc"}\n')
        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": "abc-sid",
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "window_name": "agent",
            },
        }))
        mgr.window_states["@13"] = WindowState(session_id="abc-sid", cwd="/home/agent")

        # @13 is dead, only @14 is alive
        mgr.prune_session_map(live_window_ids={"@14"})

        result = json.loads(sm_file.read_text())
        assert "den:@13" in result, (
            "FIX A: orphan record must be PRESERVED so next user message "
            "in the topic can find it via find_resumable_args_for_path"
        )
        assert result["den:@13"].get("_orphan") is True
        assert "_orphaned_at" in result["den:@13"]
        assert result["den:@13"]["session_id"] == "abc-sid"
        assert result["den:@13"]["transcript_path"] == str(tp)

    def test_window_state_still_removed_for_dead(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Marking orphan in session_map MUST still remove window_states
        entry — the window genuinely doesn't exist in tmux anymore.
        """
        sm_file, _, transcripts = sm_files
        tp = transcripts / "abc.jsonl"
        tp.write_text('x\n')
        sm_file.write_text(json.dumps({
            "den:@13": {"session_id": "s", "cwd": "/x", "transcript_path": str(tp)},
        }))
        mgr.window_states["@13"] = WindowState(session_id="s", cwd="/x")

        mgr.prune_session_map(live_window_ids=set())  # 1 entry + empty live: not bulk

        assert "@13" not in mgr.window_states


class TestOrphanRecoverableViaFindResumable:
    def test_orphan_record_returns_resume_args(
        self, mgr: SessionManager, sm_files, monkeypatch
    ) -> None:
        """Reproduces 2026-05-03 5:17/6:17/etc: after prune marked orphan,
        next message in same topic must resume the orphan session.
        """
        sm_file, state_file, transcripts = sm_files
        tp = transcripts / "abc.jsonl"
        tp.write_text("\n".join(['{"x":1}'] * 100))  # nontrivial size > 0

        # Use UUID-format session_id (claude provider validates format)
        valid_sid = "abc12345-6789-4abc-9def-0123456789ab"
        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": valid_sid,
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "window_name": "agent",
                "_orphan": True,
                "_orphaned_at": time.time(),
            },
        }))
        state_file.write_text(json.dumps({
            "thread_bindings": {},  # No binding, fresh after dead window
            "window_display_names": {},
            "window_states": {},
        }))

        # tmux returns no live windows (sid not used elsewhere)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: type("R", (), {"returncode": 0, "stdout": "@1\n@2\n", "stderr": ""})(),
        )

        args = find_resumable_args_for_path(
            "/home/agent", "claude", user_id=331129551, topic_id=1
        )

        assert args, "FIX A: orphan session_map record must be findable for resume"
        assert valid_sid in args


# ============================================================================
# FIX B — huge transcript guard
# ============================================================================


class TestHugeTranscriptGuard:
    def test_refuses_resume_above_5mb(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Reproduces den 2026-05-03 5:50: 6.78MB transcript caused 3x
        'prompt is too long: 206244 tokens > 200000 maximum' StopFailures.
        Fix: don't even try to resume transcripts > 5 MB.
        """
        sm_file, state_file, transcripts = sm_files
        tp = transcripts / "huge.jsonl"
        # Create a transcript file > 5 MB
        chunk = ('{"x":' + "0" * 1000 + "}\n").encode()
        with open(tp, "wb") as f:
            for _ in range(int(MAX_RESUME_TRANSCRIPT_MB * 1024 * 1024 / len(chunk)) + 100):
                f.write(chunk)
        assert tp.stat().st_size > MAX_RESUME_TRANSCRIPT_MB * 1024 * 1024

        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": "huge-sid",
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "window_name": "agent",
            },
        }))
        state_file.write_text(json.dumps({}))

        args = find_resumable_args_for_path(
            "/home/agent", "claude", user_id=1, topic_id=1
        )

        assert args == "", (
            "FIX B: transcripts > 5 MB must NOT be auto-resumed (causes "
            "prompt-too-long cap StopFailures)"
        )

    def test_under_5mb_still_resumes(
        self, mgr: SessionManager, sm_files, monkeypatch
    ) -> None:
        """Smaller transcripts continue to resume normally."""
        sm_file, state_file, transcripts = sm_files
        tp = transcripts / "small.jsonl"
        tp.write_text("\n".join(['{"x":1}'] * 1000))  # ~10 KB

        small_sid = "11111111-2222-4333-8444-555555555555"
        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": small_sid,
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "window_name": "agent",
            },
        }))
        state_file.write_text(json.dumps({}))

        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        )

        args = find_resumable_args_for_path(
            "/home/agent", "claude", user_id=1, topic_id=1
        )

        assert small_sid in args


# ============================================================================
# FIX A regression — load_session_map ignores orphan entries
# ============================================================================


class TestLoadSessionMapIgnoresOrphans:
    @pytest.mark.asyncio
    async def test_orphan_entry_does_not_revive_window_state(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Orphan entries kept on disk for find_resumable MUST NOT be loaded
        as live window states by load_session_map (would re-create stub).
        """
        sm_file, _, transcripts = sm_files
        tp = transcripts / "abc.jsonl"
        tp.write_text("x")
        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": "abc-sid",
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "_orphan": True,
                "_orphaned_at": time.time(),
            },
        }))
        # Pre-state: window_states empty
        assert "@13" not in mgr.window_states

        await mgr.load_session_map()

        assert "@13" not in mgr.window_states, (
            "Orphan entries must be skipped by load_session_map "
            "(otherwise revived as zombie window_state)"
        )


# ============================================================================
# Cleanup of stale orphans
# ============================================================================


class TestCleanupOrphanSessions:
    def test_old_orphans_removed(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Orphans older than max_age_days must be removed (avoid unbounded growth)."""
        sm_file, _, transcripts = sm_files
        old_tp = transcripts / "old.jsonl"
        old_tp.write_text("x")
        new_tp = transcripts / "new.jsonl"
        new_tp.write_text("x")

        now = time.time()
        sm_file.write_text(json.dumps({
            "den:@1": {
                "session_id": "old", "transcript_path": str(old_tp),
                "_orphan": True, "_orphaned_at": now - 8 * 86400,  # 8 days old
            },
            "den:@2": {
                "session_id": "new", "transcript_path": str(new_tp),
                "_orphan": True, "_orphaned_at": now - 3 * 86400,  # 3 days old
            },
            "den:@3": {  # alive (no _orphan flag)
                "session_id": "alive", "transcript_path": str(new_tp),
            },
        }))

        mgr.cleanup_orphan_sessions(max_age_days=7)

        result = json.loads(sm_file.read_text())
        assert "den:@1" not in result, "8-day orphan must be cleaned"
        assert "den:@2" in result, "3-day orphan must be kept"
        assert "den:@3" in result, "live entry must be untouched"


# ============================================================================
# Anti-bulk-prune still applies (sanity: didn't break patch 60 guard)
# ============================================================================


class TestEdgeCases:
    def test_resume_skips_when_transcript_file_deleted(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """If transcript .jsonl was manually deleted from disk, resume must
        gracefully skip (not crash with FileNotFoundError)."""
        sm_file, state_file, transcripts = sm_files
        ghost_tp = transcripts / "ghost.jsonl"  # never created
        sm_file.write_text(json.dumps({
            "den:@13": {
                "session_id": "11111111-2222-4333-8444-555555555555",
                "cwd": "/home/agent",
                "transcript_path": str(ghost_tp),
                "window_name": "agent",
            },
        }))
        state_file.write_text(json.dumps({}))

        args = find_resumable_args_for_path(
            "/home/agent", "claude", user_id=1, topic_id=1
        )

        assert args == "", "missing transcript file must skip silently"

    def test_cleanup_orphan_handles_missing_orphaned_at(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Legacy orphan entries without _orphaned_at field — treat as old
        (cutoff > 0 always) and remove safely.
        """
        sm_file, _, transcripts = sm_files
        sm_file.write_text(json.dumps({
            "den:@1": {
                "session_id": "legacy",
                "transcript_path": str(transcripts / "x.jsonl"),
                "_orphan": True,
                # NO _orphaned_at field
            },
        }))

        # Should not crash, should remove the entry (treats missing as 0 = ancient)
        mgr.cleanup_orphan_sessions(max_age_days=7)

        result = json.loads(sm_file.read_text())
        assert "den:@1" not in result

    def test_cleanup_orphan_handles_corrupted_json(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Cleanup must not crash on malformed session_map.json."""
        sm_file, _, _ = sm_files
        sm_file.write_text("{ not valid json")

        mgr.cleanup_orphan_sessions()  # must not raise

    def test_cleanup_orphan_handles_missing_file(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Cleanup must not crash when session_map.json doesn't exist."""
        sm_file, _, _ = sm_files
        # Don't create the file
        mgr.cleanup_orphan_sessions()  # must not raise


class TestAntiBulkPruneStillWorks:
    def test_skips_when_live_empty_and_2plus_live_entries(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """patch 60 guard: empty live set + multiple live entries = skip.
        Orphan-marked entries don't count as live for the guard threshold.
        """
        sm_file, _, transcripts = sm_files
        sm_file.write_text(json.dumps({
            "den:@1": {"session_id": "s1", "cwd": "/a"},  # live
            "den:@2": {"session_id": "s2", "cwd": "/b"},  # live
        }))

        # Empty live set + 2 live entries → must skip
        mgr.prune_session_map(live_window_ids=set())

        result = json.loads(sm_file.read_text())
        assert "den:@1" in result
        assert "den:@2" in result
        # Neither marked orphan (skip happened before marking)
        assert "_orphan" not in result["den:@1"]
        assert "_orphan" not in result["den:@2"]


# ============================================================================
# FIX B — set_window_provider(hookless) orphans entry instead of deleting
# Reproduces den 2026-05-07 06:34 OOM: claude OOM-killed in @3438, pane
# fell back to bash, status_polling called set_window_provider("shell"),
# session_map entry was DELETED, find_resumable found nothing, fresh empty
# session spawned 11 minutes later when user returned. Fix: mark orphan.
# ============================================================================


class TestSetWindowProviderOrphansOnHooklessTransition:
    def _seed_claude_entry(self, sm_file, transcripts, sid="bfd76f00-sid"):
        tp = transcripts / f"{sid}.jsonl"
        tp.write_text('{"sessionId":"' + sid + '"}\n')
        sm_file.write_text(json.dumps({
            "den:@3438": {
                "session_id": sid,
                "cwd": "/home/agent",
                "transcript_path": str(tp),
                "window_name": "agent-2",
                "provider_name": "claude",
            },
        }))
        return tp

    @pytest.mark.parametrize("hookless_provider", ["shell", "codex", "gemini"])
    def test_claude_to_hookless_marks_orphan_preserves_sid(
        self, mgr: SessionManager, sm_files, hookless_provider: str
    ) -> None:
        """The OOM bug: claude → ANY hookless provider must preserve entry
        as orphan. Covers shell (OOM/crash path) + codex/gemini (deliberate
        provider switch via UI). All three have supports_hook=False.
        """
        sm_file, _, transcripts = sm_files
        self._seed_claude_entry(sm_file, transcripts)
        # state already knows about live claude session
        mgr.window_states["@3438"] = WindowState(
            session_id="bfd76f00-sid",
            cwd="/home/agent",
            transcript_path=str(transcripts / "bfd76f00-sid.jsonl"),
            provider_name="claude",
        )

        # Simulate status_polling/UI switching to a hookless provider
        mgr.set_window_provider("@3438", hookless_provider)

        # Entry MUST exist (not deleted) and be marked orphan
        result = json.loads(sm_file.read_text())
        assert "den:@3438" in result, (
            f"FIX B: entry must NOT be deleted on {hookless_provider} "
            f"hookless transition — needed for find_resumable recovery"
        )
        assert result["den:@3438"]["_orphan"] is True
        assert result["den:@3438"]["_orphaned_at"] > 0
        assert result["den:@3438"]["session_id"] == "bfd76f00-sid"
        # State.session_id wiped (in-memory) — fresh hookless session has no sid
        assert mgr.window_states["@3438"].session_id == ""

    def test_orphan_mark_idempotent_no_double_write(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Hot-path I/O guard: status_polling can flicker shell→claude→shell.
        Repeated orphan-mark on already-orphan entry must not rewrite file.
        """
        sm_file, _, transcripts = sm_files
        self._seed_claude_entry(sm_file, transcripts)

        # First call sets orphan
        mgr._orphan_session_map_entry("@3438", expected_sid="bfd76f00-sid")
        first_mtime = sm_file.stat().st_mtime_ns
        first_data = json.loads(sm_file.read_text())
        assert first_data["den:@3438"]["_orphan"] is True

        # Second call with same expected_sid: must be no-op
        time.sleep(0.01)  # ensure different mtime if write happened
        mgr._orphan_session_map_entry("@3438", expected_sid="bfd76f00-sid")
        second_mtime = sm_file.stat().st_mtime_ns
        assert second_mtime == first_mtime, (
            "Idempotent orphan-mark must skip write when entry is already "
            "orphan with same sid — prevents I/O storm on status_polling flicker"
        )

    def test_orphan_mark_cas_skips_when_sid_changed_by_hook(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """CAS guard: if hook beat us with a fresh live session_id between
        snapshot and lock acquisition, we must NOT stamp orphan on the live.
        Prevents silently disabling a live session for 7 days.
        """
        sm_file, _, transcripts = sm_files
        # Entry has a NEW sid (hook overwrote already)
        new_tp = transcripts / "new-sid.jsonl"
        new_tp.write_text('{"sessionId":"new-sid"}\n')
        sm_file.write_text(json.dumps({
            "den:@3438": {
                "session_id": "new-live-sid",
                "cwd": "/home/agent",
                "transcript_path": str(new_tp),
                "window_name": "agent-2",
                "provider_name": "claude",
            },
        }))

        # We snapshotted "old-sid" before mutation — but hook wrote "new-live-sid"
        mgr._orphan_session_map_entry("@3438", expected_sid="old-sid")

        # Entry must NOT be orphaned — CAS detected hook race
        result = json.loads(sm_file.read_text())
        assert "_orphan" not in result["den:@3438"], (
            "CAS check: orphan-mark must skip when sid mismatches — "
            "hook raced ahead with a fresh live session"
        )
        assert result["den:@3438"]["session_id"] == "new-live-sid"

    def test_orphan_mark_disables_cas_with_empty_expected_sid(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """When caller has no expected_sid (e.g. cwd-only switch), CAS is
        disabled and orphan-mark proceeds based on key existence alone.
        """
        sm_file, _, transcripts = sm_files
        self._seed_claude_entry(sm_file, transcripts, sid="some-sid")

        mgr._orphan_session_map_entry("@3438", expected_sid="")

        result = json.loads(sm_file.read_text())
        assert result["den:@3438"]["_orphan"] is True

    def test_orphan_mark_no_op_for_missing_entry(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """No entry in session_map → orphan-mark is no-op, not error."""
        sm_file, _, _ = sm_files
        sm_file.write_text("{}")

        # Should not raise
        mgr._orphan_session_map_entry("@9999", expected_sid="ghost")

        assert json.loads(sm_file.read_text()) == {}

    def test_set_window_provider_to_claude_does_not_orphan(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Switching TO a hook-based provider (e.g. shell→claude after restart)
        must NOT orphan-mark — hook will write a fresh entry shortly.
        """
        sm_file, _, transcripts = sm_files
        # Live shell entry (no claude record yet)
        sm_file.write_text(json.dumps({
            "den:@5000": {"session_id": "", "cwd": "/x", "provider_name": "shell"},
        }))
        mgr.window_states["@5000"] = WindowState(
            session_id="",
            cwd="/x",
            provider_name="shell",
        )

        mgr.set_window_provider("@5000", "claude")

        # Entry should remain unchanged (no orphan flag)
        result = json.loads(sm_file.read_text())
        assert "_orphan" not in result["den:@5000"]

    def test_oom_full_cycle_orphan_state_is_resumable_via_resolver(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """End-to-end up to find_resumable's internal resolver: OOM → orphan
        → orphan record passes _is_resumable_transcript check → would yield
        --resume <sid> if tmux were available. Reproduces the 2026-05-07
        Errors-topic memory loss path. Full tmux integration covered by E2E.
        """
        sm_file, _, transcripts = sm_files
        sid = "bfd76f00-7e17-4e70-b6ce-54a835b23dbe"
        tp = self._seed_claude_entry(sm_file, transcripts, sid=sid)
        mgr.window_states["@3438"] = WindowState(
            session_id=sid,
            cwd="/home/agent",
            transcript_path=str(tp),
            provider_name="claude",
        )

        # Step 1: OOM simulation
        mgr.set_window_provider("@3438", "shell")

        # Step 2: orphan record is preserved AND its transcript is intact
        # so a resumer would accept it.
        result = json.loads(sm_file.read_text())
        entry = result["den:@3438"]
        assert entry["_orphan"] is True
        assert entry["session_id"] == sid
        assert entry["transcript_path"] == str(tp)
        # Transcript file exists, non-empty, under cap → resumable
        from pathlib import Path
        st = Path(entry["transcript_path"]).stat()
        assert st.st_size > 0
        from ccgram.session import MAX_RESUME_TRANSCRIPT_MB
        assert st.st_size <= MAX_RESUME_TRANSCRIPT_MB * 1024 * 1024

    def test_clear_window_session_does_not_touch_session_map(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """The /clear Telegram command path (bot.py:751) only clears state,
        not session_map. Verify the fix doesn't accidentally affect this.
        """
        sm_file, _, transcripts = sm_files
        self._seed_claude_entry(sm_file, transcripts, sid="active-sid")
        mgr.window_states["@3438"] = WindowState(
            session_id="active-sid",
            cwd="/home/agent",
            provider_name="claude",
        )

        mgr.clear_window_session("@3438")

        # Session_map entry untouched
        result = json.loads(sm_file.read_text())
        assert result["den:@3438"]["session_id"] == "active-sid"
        assert "_orphan" not in result["den:@3438"]

    def test_clear_session_map_entry_still_hard_deletes(
        self, mgr: SessionManager, sm_files
    ) -> None:
        """Sessions Kill button uses _clear_session_map_entry to fully delete.
        Verify it remains a hard delete (not silently switched to orphan-mark).
        """
        sm_file, _, transcripts = sm_files
        self._seed_claude_entry(sm_file, transcripts, sid="killed-sid")

        mgr._clear_session_map_entry("@3438")

        result = json.loads(sm_file.read_text())
        assert "den:@3438" not in result, (
            "_clear_session_map_entry must hard-delete (used by Kill button "
            "to make sure killed sessions are NOT auto-resumed later)"
        )
