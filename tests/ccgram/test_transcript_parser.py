"""Tests for ccgram.transcript_parser — pure logic, no I/O."""

import pytest

from ccgram.providers.base import EXPANDABLE_QUOTE_END as EXPQUOTE_END
from ccgram.providers.base import EXPANDABLE_QUOTE_START as EXPQUOTE_START
from ccgram.transcript_parser import (
    ParsedMessage,
    TranscriptParser,
)

# ── parse_line ───────────────────────────────────────────────────────────


class TestParseLine:
    @pytest.mark.parametrize(
        "line, expected",
        [
            ('{"type": "user"}', {"type": "user"}),
            ("not-json", None),
            ("", None),
            ("   \t  ", None),
        ],
        ids=["valid_json", "invalid_json", "empty", "whitespace"],
    )
    def test_parse_line(self, line: str, expected: dict | None):
        assert TranscriptParser.parse_line(line) == expected


# ── extract_text_only ────────────────────────────────────────────────────


class TestExtractTextOnly:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("plain string", "plain string"),
            (
                [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
                "hello\nworld",
            ),
            (
                [
                    {"type": "text", "text": "keep"},
                    {"type": "tool_use", "name": "Read"},
                ],
                "keep",
            ),
            ([], ""),
            (42, ""),
        ],
        ids=["string", "text_blocks", "mixed", "empty_list", "non_list_non_string"],
    )
    def test_extract_text_only(self, content: list | str | int, expected: str):
        assert TranscriptParser.extract_text_only(content) == expected  # type: ignore[arg-type]

    def test_ansi_stripped_from_extract_text_only(self):
        content = [
            {"type": "text", "text": "\x1b[32mgreen\x1b[0m and \x1b[1;31mred\x1b[0m"}
        ]
        assert TranscriptParser.extract_text_only(content) == "green and red"


# ── ANSI stripping in parse_entries ──────────────────────────────────────


class TestAnsiStripping:
    def test_ansi_stripped_from_assistant_text_block(self):
        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "\x1b[32mhello\x1b[0m world"}]
                },
            }
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_ansi_stripped_from_user_text_block(self):
        entries = [
            {
                "type": "user",
                "message": {
                    "content": [{"type": "text", "text": "\x1b[1;34muser input\x1b[0m"}]
                },
            }
        ]
        result, _ = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].text == "user input"


# ── format_tool_use_summary ──────────────────────────────────────────────


class TestFormatToolUseSummary:
    @pytest.mark.parametrize(
        "name, input_data, expected",
        [
            ("Read", {"file_path": "src/main.py"}, "\U0001f4d6 **Read** `src/main.py`"),
            ("Write", {"file_path": "out.txt"}, "\U0001f4dd **Write** `out.txt`"),
            ("Bash", {"command": "ls -la"}, "\u26a1 **Bash** `ls -la`"),
            ("Grep", {"pattern": "TODO"}, "\U0001f50d **Grep** `TODO`"),
            ("Glob", {"pattern": "*.py"}, "\U0001f4c2 **Glob** `*.py`"),
            (
                "Task",
                {"description": "analyze code"},
                "\U0001f916 **Task** `analyze code`",
            ),
            (
                "WebFetch",
                {"url": "https://example.com"},
                "\U0001f310 **WebFetch** `https://example.com`",
            ),
            (
                "WebSearch",
                {"query": "python async"},
                "\U0001f50e **WebSearch** `python async`",
            ),
            (
                "TodoWrite",
                {"todos": [1, 2, 3]},
                "\u2705 **TodoWrite** `3 item(s)`",
            ),
            ("TodoRead", {}, "\U0001f4cb **TodoRead**"),
            (
                "AskUserQuestion",
                {"questions": [{"question": "Continue?"}]},
                "\u2753 **AskUserQuestion** `Continue?`",
            ),
            ("ExitPlanMode", {}, "\U0001f4cb **ExitPlanMode**"),
            (
                "Skill",
                {"skill": "code-review"},
                "\u2699\ufe0f **Skill** `code-review`",
            ),
            (
                "CustomTool",
                {"first_key": "value1"},
                "**CustomTool** `value1`",
            ),
        ],
        ids=[
            "Read",
            "Write",
            "Bash",
            "Grep",
            "Glob",
            "Task",
            "WebFetch",
            "WebSearch",
            "TodoWrite",
            "TodoRead",
            "AskUserQuestion",
            "ExitPlanMode",
            "Skill",
            "unknown_tool",
        ],
    )
    def test_tool_summary(self, name: str, input_data: dict, expected: str):
        assert TranscriptParser.format_tool_use_summary(name, input_data) == expected

    def test_non_dict_input(self):
        assert (
            TranscriptParser.format_tool_use_summary("Read", "not a dict")
            == "\U0001f4d6 **Read**"
        )

    def test_truncation_at_200_chars(self):
        long_value = "x" * 250
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": long_value}
        )
        assert len(long_value) > 200
        assert result == f"\u26a1 **Bash** `{'x' * 200}\u2026`"



class TestTodoRendering:
    """BRAIN FORK: TodoWrite → status with checklist, TodoRead → Checking tasks."""

    def test_todowrite_all_pending_returns_checklist(self):
        result = TranscriptParser.format_tool_use_summary(
            "TodoWrite",
            {
                "todos": [
                    {"content": "Fix bug", "status": "pending"},
                    {"content": "Write tests", "status": "pending"},
                    {"content": "Commit", "status": "pending"},
                ]
            },
        )
        assert result.startswith("__STATUS__")
        assert "Planning tasks" in result
        assert "[ ] Fix bug" in result
        assert "[ ] Write tests" in result
        assert "[ ] Commit" in result

    def test_todowrite_mixed_statuses_renders_correct_icons(self):
        result = TranscriptParser.format_tool_use_summary(
            "TodoWrite",
            {
                "todos": [
                    {"content": "Done task", "status": "completed"},
                    {"content": "Active task", "status": "in_progress"},
                    {"content": "Not started", "status": "pending"},
                ]
            },
        )
        assert "[x] Done task" in result
        assert "[~] Active task" in result
        assert "[ ] Not started" in result

    def test_todowrite_empty_list_falls_back_to_thinking(self):
        result = TranscriptParser.format_tool_use_summary(
            "TodoWrite", {"todos": []}
        )
        assert result == "__STATUS__Thinking..."

    def test_todowrite_missing_field_falls_back_to_thinking(self):
        result = TranscriptParser.format_tool_use_summary("TodoWrite", {})
        assert result == "__STATUS__Thinking..."

    def test_todowrite_uses_activeForm_when_content_missing(self):
        result = TranscriptParser.format_tool_use_summary(
            "TodoWrite",
            {"todos": [{"activeForm": "Fixing bug", "status": "in_progress"}]},
        )
        assert "[~] Fixing bug" in result

    def test_todowrite_truncates_long_items(self):
        long_content = "x" * 200
        result = TranscriptParser.format_tool_use_summary(
            "TodoWrite",
            {"todos": [{"content": long_content, "status": "pending"}]},
        )
        # 80-char cap + ellipsis
        assert "…" in result
        # No line exceeds ~85 chars (prefix + 80 + ellipsis)
        for line in result.split("\n"):
            assert len(line) <= 95

    def test_todoread_returns_checking_status(self):
        result = TranscriptParser.format_tool_use_summary("TodoRead", {})
        assert result == "__STATUS__Checking tasks..."

    def test_silent_tools_no_longer_hides_todowrite(self):
        # After patch: TodoWrite falls through to custom handler, not hidden
        assert "TodoWrite" not in TranscriptParser.SILENT_TOOLS
        assert "TodoRead" not in TranscriptParser.SILENT_TOOLS


class TestTaskStarEvents:
    """BRAIN FORK: Claude Code 2.1.84+ Task* tools emit __TASK_EVENT__ markers."""

    @staticmethod
    def _parse_marker(result: str) -> dict:
        import json as _json

        assert result.startswith("__TASK_EVENT__"), (
            f"expected __TASK_EVENT__ prefix, got {result!r}"
        )
        return _json.loads(result[len("__TASK_EVENT__"):])

    def test_task_create_emits_marker_with_fields(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate",
            {
                "id": "t1",
                "subject": "Fix auth bug",
                "activeForm": "Fixing auth bug",
                "description": "Detailed description here",
            },
        )
        payload = self._parse_marker(result)
        assert payload["op"] == "create"
        assert payload["id"] == "t1"
        assert payload["subject"] == "Fix auth bug"
        assert payload["activeForm"] == "Fixing auth bug"

    def test_task_create_accepts_task_id_alias(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate", {"task_id": "abc", "subject": "X"}
        )
        payload = self._parse_marker(result)
        assert payload["op"] == "create"
        assert payload["id"] == "abc"

    def test_task_create_accepts_taskId_alias(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate", {"taskId": "zzz", "subject": "X"}
        )
        payload = self._parse_marker(result)
        assert payload["id"] == "zzz"

    def test_task_update_emits_marker(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskUpdate", {"id": "t1", "status": "in_progress"}
        )
        payload = self._parse_marker(result)
        assert payload["op"] == "update"
        assert payload["id"] == "t1"
        assert payload["status"] == "in_progress"

    def test_task_delete_emits_marker(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskDelete", {"id": "t1"}
        )
        payload = self._parse_marker(result)
        assert payload["op"] == "delete"
        assert payload["id"] == "t1"

    def test_task_list_emits_marker_without_id(self):
        result = TranscriptParser.format_tool_use_summary("TaskList", {})
        payload = self._parse_marker(result)
        assert payload == {"op": "list"}

    def test_task_create_preserves_cyrillic_in_subject(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate", {"id": "t1", "subject": "Вычитка: priorities.md"}
        )
        payload = self._parse_marker(result)
        assert payload["subject"] == "Вычитка: priorities.md"

    def test_task_create_missing_id_still_emits_marker(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate", {"subject": "No id"}
        )
        payload = self._parse_marker(result)
        assert payload["op"] == "create"
        assert "id" not in payload
        assert payload["subject"] == "No id"

    def test_task_create_includes_tuid_from_tool_use_id(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskCreate",
            {"_tool_use_id": "toolu_abc123", "subject": "Real usage"},
        )
        payload = self._parse_marker(result)
        assert payload["tuid"] == "toolu_abc123"
        assert "id" not in payload  # TaskCreate input has no id

    def test_task_update_includes_tuid_alongside_id(self):
        result = TranscriptParser.format_tool_use_summary(
            "TaskUpdate",
            {
                "_tool_use_id": "toolu_upd",
                "taskId": "5",
                "status": "completed",
            },
        )
        payload = self._parse_marker(result)
        assert payload["id"] == "5"
        assert payload["tuid"] == "toolu_upd"
        assert payload["status"] == "completed"


class TestFileToolsAsStatus:
    """BRAIN FORK: Read/Edit/Write/NotebookEdit → temp status (not persistent)."""

    def test_read_returns_status_prefix(self):
        result = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/home/agent/mcp-tools/mcp-proxy.js"}
        )
        assert result.startswith("__STATUS__"), \
            f"expected __STATUS__ prefix, got {result!r}"
        assert "Read" in result
        assert "mcp-proxy" in result

    def test_write_returns_status_prefix(self):
        result = TranscriptParser.format_tool_use_summary(
            "Write", {"file_path": "/tmp/new_file.py", "content": "print(1)"}
        )
        assert result.startswith("__STATUS__")
        assert "Write" in result
        assert "new_file" in result

    def test_edit_returns_status_prefix(self):
        result = TranscriptParser.format_tool_use_summary(
            "Edit",
            {
                "file_path": "/home/agent/foo.py",
                "old_string": "a",
                "new_string": "b",
            },
        )
        assert result.startswith("__STATUS__")
        assert "Edit" in result
        assert "foo" in result

    def test_notebook_edit_returns_status_prefix(self):
        result = TranscriptParser.format_tool_use_summary(
            "NotebookEdit",
            {
                "notebook_path": "/tmp/analysis.ipynb",
                "new_source": "import pandas",
            },
        )
        assert result.startswith("__STATUS__")

    def test_read_from_ccgram_uploads_keeps_existing_status(self):
        """Existing upload-media status (Viewing image / Reading document / etc)
        should NOT be double-prefixed."""
        result = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/home/agent/.ccgram-uploads/image.jpg"}
        )
        # Single __STATUS__ prefix only
        assert result.count("__STATUS__") == 1
        assert "Viewing image" in result

    def test_read_strips_file_extension(self):
        result = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/path/to/session.py"}
        )
        # `.py` stripped for clean display
        assert ".py" not in result
        assert "session" in result

    def test_read_generic_basename_includes_parent_dir(self):
        """SKILL.md in different skill dirs must render distinguishably."""
        result_pw = TranscriptParser.format_tool_use_summary(
            "Read",
            {"file_path": "/home/agent/.claude/skills/post-writer/SKILL.md"},
        )
        result_rs = TranscriptParser.format_tool_use_summary(
            "Read",
            {"file_path": "/home/agent/.claude/skills/research/SKILL.md"},
        )
        assert "post-writer/SKILL" in result_pw
        assert "research/SKILL" in result_rs
        assert result_pw != result_rs

    def test_read_generic_basename_readme(self):
        result = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/home/agent/projects/foo/README.md"}
        )
        assert "foo/README" in result

    def test_read_non_generic_basename_stays_short(self):
        result = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/home/agent/src/utils.py"}
        )
        assert "utils" in result
        assert "src/utils" not in result  # non-generic keeps bare basename

    def test_read_generic_basename_no_parent_falls_back(self):
        """Relative path and root-level path with no parent dir must not
        crash and must fall back to bare `SKILL` instead of emitting an
        empty prefix or a stray `/SKILL`."""
        # Reset dedup cache — identical results between the two calls
        # would otherwise collapse the second via _last_tool_summary.
        TranscriptParser._last_tool_summary.pop("_global", None)
        result_rel = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "SKILL.md"}
        )
        TranscriptParser._last_tool_summary.pop("_global", None)
        result_root = TranscriptParser.format_tool_use_summary(
            "Read", {"file_path": "/SKILL.md"}
        )
        assert "SKILL" in result_rel
        assert "SKILL" in result_root
        # No stray slash or empty prefix
        assert "/SKILL" not in result_rel
        for value in (result_rel, result_root):
            assert "  SKILL" not in value
            assert not value.endswith("/")


class TestBashAllThinking:
    """BRAIN FORK: ALL Bash commands must return __STATUS__Thinking..."""

    @pytest.mark.parametrize(
        "command",
        [
            "git push origin main",
            "rm -rf /tmp/foo",
            "mkdir -p /path",
            "pnpm build",
            "docker run hello",
            "ls -la",
            "cd /tmp",
            "cat /etc/hostname",
            "grep foo bar.txt",
            "git log --oneline",
            "git status",
            "git diff",
            "cd /home/agent && node script.js",
            "cd /tmp && pnpm build",
            "export X=1 && git push",
            "source .env && ls",
            "python3 -c 'print(1)'",
            "",
        ],
    )
    def test_bash_without_description_returns_thinking_status(self, command):
        result = TranscriptParser.format_tool_use_summary("Bash", {"command": command})
        assert result == "__STATUS__Thinking...", \
            f"Expected __STATUS__Thinking... for {command!r}, got {result!r}"

    def test_bash_with_description_uses_description_as_status(self):
        result = TranscriptParser.format_tool_use_summary(
            "Bash",
            {"command": "systemctl --user restart ccgram-den", "description": "Restart ccgram-den"},
        )
        assert result == "__STATUS__Restart ccgram-den"

    def test_bash_with_long_description_gets_truncated(self):
        long_desc = "x" * 100
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": "some command", "description": long_desc}
        )
        assert result.startswith("__STATUS__")
        assert result.endswith("…")
        assert len(result) == len("__STATUS__") + 80

    def test_bash_with_empty_description_falls_back_to_thinking(self):
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": "ls -la", "description": ""}
        )
        assert result == "__STATUS__Thinking..."

    def test_bash_with_whitespace_description_falls_back_to_thinking(self):
        result = TranscriptParser.format_tool_use_summary(
            "Bash", {"command": "ls -la", "description": "   "}
        )
        assert result == "__STATUS__Thinking..."


# ── extract_tool_result_text ─────────────────────────────────────────────


class TestExtractToolResultText:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ("raw string", "raw string"),
            (
                [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                "line1\nline2",
            ),
            (
                [{"type": "text", "text": "keep"}, {"type": "image", "data": "..."}],
                "keep",
            ),
            (None, ""),
        ],
        ids=["string", "text_blocks", "mixed", "none"],
    )
    def test_extract_tool_result_text(self, content: str | list | None, expected: str):
        assert TranscriptParser.extract_tool_result_text(content) == expected


# ── parse_message ────────────────────────────────────────────────────────


class TestParseMessage:
    def test_user_text(self):
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="user", text="hello")

    def test_assistant_text(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi there"}]},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="hi there")

    def test_local_command_with_stdout(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<command-name>/help</command-name>"
                            "<local-command-stdout>Available commands</local-command-stdout>"
                        ),
                    }
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command"
        assert result.text == "Available commands"
        assert result.tool_name == "/help"

    def test_local_command_invoke(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "<command-name>/clear</command-name>"}
                ]
            },
        }
        result = TranscriptParser.parse_message(data)
        assert result is not None
        assert result.message_type == "local_command_invoke"
        assert result.text == ""
        assert result.tool_name == "/clear"

    def test_non_user_assistant_returns_none(self):
        data = {
            "type": "summary",
            "message": {"content": "summary text"},
        }
        assert TranscriptParser.parse_message(data) is None

    def test_string_content(self):
        data = {
            "type": "assistant",
            "message": {"content": "plain response"},
        }
        result = TranscriptParser.parse_message(data)
        assert result == ParsedMessage(message_type="assistant", text="plain response")


# ── _format_edit_diff ────────────────────────────────────────────────────


class TestFormatEditDiff:
    @pytest.mark.parametrize(
        "old, new, check",
        [
            (
                "hello",
                "world",
                lambda r: "-hello" in r and "+world" in r,
            ),
            (
                "line1\nline2\nline3",
                "line1\nchanged\nline3",
                lambda r: "-line2" in r and "+changed" in r,
            ),
            (
                "same",
                "same",
                lambda r: r == "",
            ),
        ],
        ids=["single_line", "multi_line", "identical"],
    )
    def test_format_edit_diff(self, old: str, new: str, check):
        result = TranscriptParser._format_edit_diff(old, new)
        assert check(result), f"Check failed for ({old!r}, {new!r}): {result!r}"


# ── _format_tool_result_text ─────────────────────────────────────────────


class TestFormatToolResultText:
    @pytest.mark.parametrize(
        "text, tool_name, check",
        [
            (
                "line1\nline2\nline3",
                "Read",
                lambda r: r == "  \u23bf  3 lines",
            ),
            (
                "line1\nline2",
                "Write",
                lambda r: r == "  \u23bf  2 lines written",
            ),
            (
                "output line",
                "Bash",
                lambda r: (
                    r.startswith("  \u23bf  1 lines")
                    and EXPQUOTE_START in r
                    and EXPQUOTE_END in r
                ),
            ),
            (
                "file1.py\nfile2.py\n",
                "Grep",
                lambda r: "2 matches" in r and EXPQUOTE_START in r,
            ),
            (
                "a.py\nb.py\nc.py",
                "Glob",
                lambda r: "3 files" in r and EXPQUOTE_START in r,
            ),
            (
                "agent says hello",
                "Task",
                lambda r: "1 lines" in r and EXPQUOTE_START in r,
            ),
            (
                "page content here",
                "WebFetch",
                lambda r: (
                    f"{len('page content here')} chars" in r and EXPQUOTE_START in r
                ),
            ),
            (
                "",
                "Read",
                lambda r: r == "",
            ),
        ],
        ids=["Read", "Write", "Bash", "Grep", "Glob", "Task", "WebFetch", "empty"],
    )
    def test_format_tool_result_text(self, text: str, tool_name: str, check):
        result = TranscriptParser._format_tool_result_text(text, tool_name)
        assert check(result), f"Failed check for {tool_name!r}: {result!r}"


# ── parse_entries ────────────────────────────────────────────────────────


class TestParseEntries:
    def test_assistant_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("assistant", [make_text_block("Hello!")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].text == "Hello!"
        assert result[0].content_type == "text"

    def test_user_text(self, make_jsonl_entry, make_text_block):
        entries = [make_jsonl_entry("user", [make_text_block("Hi bot")])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].text == "Hi bot"

    def test_tool_use_and_result_pairing(
        self,
        make_jsonl_entry,
        make_text_block,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "app.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "file contents line1\nline2\nline3")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        tool_use_entries = [e for e in result if e.content_type == "tool_use"]
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_use_entries) == 1
        assert tool_use_entries[0].tool_use_id == "t1"
        assert "\U0001f4d6 **Read**" in tool_use_entries[0].text
        assert len(tool_result_entries) == 1
        assert tool_result_entries[0].tool_use_id == "t1"
        assert not pending

    def test_thinking_block(self, make_jsonl_entry, make_thinking_block):
        entries = [
            make_jsonl_entry("assistant", [make_thinking_block("reasoning here")])
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "thinking"
        assert EXPQUOTE_START in result[0].text
        assert EXPQUOTE_END in result[0].text
        assert "reasoning here" in result[0].text

    def test_local_command_with_stdout(self, make_jsonl_entry, make_text_block):
        xml = (
            "<command-name>/status</command-name>"
            "<local-command-stdout>all good</local-command-stdout>"
        )
        entries = [make_jsonl_entry("user", [make_text_block(xml)])]
        result, pending = TranscriptParser.parse_entries(entries)
        assert len(result) == 1
        assert result[0].content_type == "local_command"
        assert "/status" in result[0].text
        assert "all good" in result[0].text

    def test_exit_plan_mode_emits_plan(self, make_jsonl_entry, make_tool_use_block):
        block = make_tool_use_block(
            "t1", "ExitPlanMode", {"plan": "Step 1: do X\nStep 2: do Y"}
        )
        entries = [make_jsonl_entry("assistant", [block])]
        result, pending = TranscriptParser.parse_entries(entries)
        texts = [e for e in result if e.content_type == "text"]
        tool_uses = [e for e in result if e.content_type == "tool_use"]
        assert len(texts) == 1
        assert "Step 1: do X" in texts[0].text
        assert len(tool_uses) >= 1

    def test_edit_tool_diff_stats(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        edit_input = {
            "file_path": "main.py",
            "old_string": "old line",
            "new_string": "new line",
        }
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Edit", edit_input)],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "OK")],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        # BRAIN FORK (patch 54): tool_result emits 0 visible bytes.
        # pending_tools is popped but no ParsedEntry is appended.
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 0
        assert pending == {}

    def test_error_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Bash", {"command": "rm -rf /"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", "Permission denied", is_error=True)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        # BRAIN FORK (patch 54): errors from tool_result not shown in chat;
        # Fred surfaces them in his text response instead.
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 0
        assert pending == {}

    def test_interrupted_tool_result(
        self,
        make_jsonl_entry,
        make_tool_use_block,
        make_tool_result_block,
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "x.py"})],
            ),
            make_jsonl_entry(
                "user",
                [make_tool_result_block("t1", TranscriptParser._INTERRUPTED_TEXT)],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        # BRAIN FORK (patch 54): interrupted tool_result not shown in chat.
        tool_result_entries = [e for e in result if e.content_type == "tool_result"]
        assert len(tool_result_entries) == 0
        assert pending == {}

    def test_pending_tools_carry_over(self, make_jsonl_entry, make_tool_use_block):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools={})
        assert "t1" in pending
        flushed = [
            e for e in result if e.content_type == "tool_use" and e.tool_use_id == "t1"
        ]
        assert len(flushed) == 1

    def test_pending_tools_flushed_without_carry_over(
        self, make_jsonl_entry, make_tool_use_block
    ):
        entries = [
            make_jsonl_entry(
                "assistant",
                [make_tool_use_block("t1", "Read", {"file_path": "a.py"})],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries, pending_tools=None)
        tool_entries = [e for e in result if e.tool_use_id == "t1"]
        assert len(tool_entries) == 2
        assert tool_entries[0].content_type == "tool_use"
        assert tool_entries[1].content_type == "tool_use"

    def test_system_tag_filtered(self, make_jsonl_entry, make_text_block):
        entries = [
            make_jsonl_entry(
                "user",
                [
                    make_text_block(
                        "<system-reminder>secret instructions</system-reminder>"
                    )
                ],
            ),
        ]
        result, pending = TranscriptParser.parse_entries(entries)
        user_entries = [e for e in result if e.role == "user"]
        assert len(user_entries) == 0
