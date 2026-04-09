"""RBAC middleware for Brain multi-user access control.

Checks user permissions via Supabase (schema brain) before forwarding
messages to Claude Code. Generates settings.local.json with deny rules
per-user per-project, scoped to the current Telegram topic.

BRAIN FORK: added for multi-user permission management.
Topic scoping: each topic = one project. User can only work with that project.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()

# --- Configuration ---

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
BRAIN_CONTEXT = os.getenv("BRAIN_CONTEXT", "")

# Topics where Fred does not respond (e.g. Chat topic for human-only conversation)
_ignored_raw = os.getenv("CCGRAM_IGNORED_TOPICS", "")
IGNORED_TOPICS: set[int] = {int(t.strip()) for t in _ignored_raw.split(",") if t.strip()} if _ignored_raw else set()

# --- Cache ---

_CACHE_TTL = 300  # 5 minutes
_profile_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_all_projects_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_thread_project_cache: dict[int, tuple[float, str | None]] = {}


def _get_cached_profile(telegram_id: int) -> dict[str, Any] | None:
    if telegram_id in _profile_cache:
        ts, data = _profile_cache[telegram_id]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _set_cached_profile(telegram_id: int, data: dict[str, Any]) -> None:
    _profile_cache[telegram_id] = (time.time(), data)


def invalidate_cache(telegram_id: int | None = None) -> None:
    """Clear cache for a specific user or all users."""
    if telegram_id is None:
        _profile_cache.clear()
        _all_projects_cache.clear()
        _thread_project_cache.clear()
    else:
        _profile_cache.pop(telegram_id, None)


# --- Supabase RPC calls ---

def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


async def _fetch_user_profile(telegram_id: int) -> dict[str, Any] | None:
    """Fetch full user profile from Supabase (permissions + projects)."""
    if not SUPABASE_URL or not SUPABASE_KEY or not BRAIN_CONTEXT:
        logger.warning("RBAC not configured (missing SUPABASE_URL/KEY/BRAIN_CONTEXT)")
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/get_brain_user_profile",
                headers=_headers(),
                json={"p_telegram_id": telegram_id, "p_context": BRAIN_CONTEXT},
            )
            resp.raise_for_status()
            data = resp.json()
            if data is None:
                return None
            return data
    except Exception as e:
        logger.error("RBAC fetch failed", error=str(e), telegram_id=telegram_id)
        return None


async def get_all_projects(context: str | None = None) -> list[dict[str, Any]]:
    """Get all projects for a context from Supabase (cached).
    Returns list of {slug, path, thread_id}.
    """
    ctx = context or BRAIN_CONTEXT
    if not ctx:
        return []

    if ctx in _all_projects_cache:
        ts, projects = _all_projects_cache[ctx]
        if time.time() - ts < _CACHE_TTL:
            return projects

    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/get_brain_all_projects",
                headers=_headers(),
                json={"p_context": ctx},
            )
            resp.raise_for_status()
            projects = resp.json() or []
            _all_projects_cache[ctx] = (time.time(), projects)
            return projects
    except Exception as e:
        logger.error("RBAC: failed to fetch all projects", error=str(e))
        return []


async def get_project_for_thread(thread_id: int) -> str | None:
    """Get project_slug for a thread_id from Supabase (cached)."""
    if thread_id in _thread_project_cache:
        ts, slug = _thread_project_cache[thread_id]
        if time.time() - ts < _CACHE_TTL:
            return slug

    if not SUPABASE_URL or not SUPABASE_KEY or not BRAIN_CONTEXT:
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/get_brain_project_for_thread",
                headers=_headers(),
                json={"p_thread_id": thread_id, "p_context": BRAIN_CONTEXT},
            )
            resp.raise_for_status()
            slug = resp.json()
            _thread_project_cache[thread_id] = (time.time(), slug)
            return slug
    except Exception as e:
        logger.error("RBAC: failed to lookup project for thread", error=str(e), thread_id=thread_id)
        return None


async def _log_audit(
    telegram_id: int, project_slug: str | None, action: str, status: str, text: str | None = None
) -> None:
    """Log an audit entry to Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY or not BRAIN_CONTEXT:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/log_brain_audit",
                headers=_headers(),
                json={
                    "p_telegram_id": telegram_id,
                    "p_context": BRAIN_CONTEXT,
                    "p_project_slug": project_slug,
                    "p_action": action,
                    "p_status": status,
                    "p_request_text": text[:500] if text else None,
                },
            )
    except Exception as e:
        logger.error("RBAC audit log failed", error=str(e))


# --- Permission check ---

async def get_user_profile(telegram_id: int) -> dict[str, Any] | None:
    """Get user profile (cached or fresh from Supabase)."""
    cached = _get_cached_profile(telegram_id)
    if cached is not None:
        return cached

    profile = await _fetch_user_profile(telegram_id)
    if profile is not None:
        _set_cached_profile(telegram_id, profile)
    return profile


class AccessResult:
    """Result of an access check."""
    def __init__(
        self,
        allowed: bool,
        reason: str,
        is_owner: bool = False,
        can_write: bool = False,
        can_delete: bool = False,
        project_slug: str | None = None,
        display_name: str | None = None,
        permissions: list[str] | None = None,
        telegram_id: int | None = None,
        user_projects: dict[str, Any] | None = None,
    ):
        self.allowed = allowed
        self.reason = reason
        self.is_owner = is_owner
        self.can_write = can_write
        self.can_delete = can_delete
        self.project_slug = project_slug
        self.display_name = display_name
        self.permissions = permissions or []
        self.telegram_id = telegram_id
        self.user_projects = user_projects or {}


async def check_access(
    telegram_id: int, project_slug: str | None = None
) -> AccessResult:
    """Check if a user has access, optionally to a specific project."""
    if not BRAIN_CONTEXT:
        # RBAC not configured, allow all (backwards compatible)
        return AccessResult(allowed=True, reason="rbac_disabled", is_owner=True)

    profile = await get_user_profile(telegram_id)

    if profile is None:
        return AccessResult(allowed=False, reason="user_not_found", telegram_id=telegram_id)

    is_owner = profile.get("is_owner", False)
    display_name = profile.get("display_name", "unknown")
    permissions = profile.get("permissions", [])
    projects = profile.get("projects", {})

    if is_owner:
        return AccessResult(
            allowed=True,
            reason="owner",
            is_owner=True,
            can_write=True,
            can_delete=True,
            project_slug=project_slug,
            display_name=display_name,
            permissions=permissions,
            telegram_id=telegram_id,
            user_projects=projects,
        )

    if project_slug is not None:
        project_access = projects.get(project_slug)
        if project_access is None:
            return AccessResult(
                allowed=False,
                reason="no_project_access",
                display_name=display_name,
                project_slug=project_slug,
                telegram_id=telegram_id,
                user_projects=projects,
            )
        return AccessResult(
            allowed=True,
            reason="member",
            is_owner=False,
            can_write=project_access.get("can_write", False),
            can_delete=project_access.get("can_delete", False),
            project_slug=project_slug,
            display_name=display_name,
            permissions=permissions,
            telegram_id=telegram_id,
            user_projects=projects,
        )

    # No project specified (General/informational topic), user exists and is active
    return AccessResult(
        allowed=True,
        reason="member",
        is_owner=False,
        display_name=display_name,
        permissions=permissions,
        telegram_id=telegram_id,
        user_projects=projects,
    )


# --- Topic-scoped settings.local.json generation ---

def _expand_path(path: str) -> str:
    """Expand ~/projects/... to /home/agent/projects/..."""
    if path.startswith("~/"):
        return f"/home/agent/{path[2:]}"
    return path


def _project_deny_all(project_path: str, project_slug: str) -> list[str]:
    """Deny rules that block ALL access to a project (read + write + tools).
    Uses universal Bash pattern to block ANY command containing project path."""
    path = _expand_path(project_path)
    return [
        f"Read(//{path}/**)",
        f"Edit(//{path}/**)",
        f"Write(//{path}/**)",
        f"Glob(//{path}/**)",
        f"Grep(//{path}/**)",
    ]


def _project_deny_write(project_path: str, project_slug: str) -> list[str]:
    """Deny rules that block WRITE access to a project (read allowed)."""
    path = _expand_path(project_path)
    return [
        f"Edit(//{path}/**)",
        f"Write(//{path}/**)",
        f"Bash(rm */{project_slug}/*)",
        f"Bash(mv */{project_slug}/*)",
        f"Bash(git -C */{project_slug} commit*)",
        f"Bash(git -C */{project_slug} push*)",
    ]


async def generate_settings_local(
    access: AccessResult,
    cwd: str,
    current_project_slug: str | None = None,
    current_project_path: str | None = None,
) -> None:
    """Write .claude/settings.local.json with topic-scoped deny rules.

    Topic scoping: user can only work with the project bound to current topic.
    Other projects are blocked even if user has access to them elsewhere.

    Owner: no file generated (full access).
    Non-owner in project topic: only current project accessible.
    Non-owner in General/informational: read own projects, no writes anywhere.
    """
    if access.is_owner:
        settings_path = Path(cwd) / ".claude" / "settings.local.json"
        if settings_path.exists():
            try:
                settings_path.unlink()
            except OSError:
                pass
        return

    all_projects = await get_all_projects()
    deny_rules: list[str] = []

    if current_project_slug is None:
        # --- General / informational topic ---
        # No file modifications anywhere.
        # Edit(**)/Write(**) do NOT block absolute paths outside CWD.
        # Must explicitly deny each project path.
        deny_rules.extend([
            "Bash(rm *)",
            "Bash(mv *)",
            "Bash(git commit*)",
            "Bash(git push*)",
            "Bash(pnpm install*)",
            "Bash(npm install*)",
        ])
        for p in all_projects:
            path = _expand_path(p["path"])
            if p["slug"] in access.user_projects:
                # User has access: allow read, block write
                deny_rules.extend([
                    f"Edit(//{path}/**)",
                    f"Write(//{path}/**)",
                ])
            else:
                # No access: block everything
                deny_rules.extend(_project_deny_all(p["path"], p["slug"]))
    else:
        # --- Project topic ---
        # Block ALL other projects (even if user has access elsewhere)
        for p in all_projects:
            if p["slug"] == current_project_slug:
                continue
            deny_rules.extend(_project_deny_all(p["path"], p["slug"]))

        # Current project: apply can_write/can_delete restrictions
        if not access.can_write and current_project_path:
            deny_rules.extend(_project_deny_write(current_project_path, current_project_slug))

        if not access.can_delete:
            deny_rules.extend([
                "Bash(rm *)",
                "Bash(git push --force *)",
                "Bash(git reset *)",
            ])

    # Always block shared/ write for non-owner
    deny_rules.extend([
        "Write(//home/agent/shared/**)",
        "Edit(//home/agent/shared/**)",
    ])

    # MCP restrictions based on permissions
    if "deploy" not in access.permissions:
        deny_rules.append("Bash(vercel *)")

    if "mcp:write" not in access.permissions:
        deny_rules.extend([
            "mcp__supabase__apply_migration",
            "mcp__supabase__execute_sql",
            "mcp__n8n__n8n_create_workflow",
            "mcp__n8n__n8n_update_*",
            "mcp__slack__slack_post_message",
        ])

    if "mcp:destroy" not in access.permissions:
        deny_rules.extend([
            "mcp__supabase__execute_sql",
            "mcp__n8n__n8n_delete_workflow",
        ])

    if not deny_rules:
        return

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_deny: list[str] = []
    for rule in deny_rules:
        if rule not in seen:
            seen.add(rule)
            unique_deny.append(rule)

    settings_dir = Path(cwd) / ".claude"
    settings_dir.mkdir(exist_ok=True)

    settings = {"permissions": {"deny": unique_deny}}

    settings_path = settings_dir / "settings.local.json"
    try:
        settings_path.write_text(json.dumps(settings, indent=2))
        logger.info(
            "RBAC settings.local.json written (topic-scoped)",
            user=access.display_name,
            project=current_project_slug or "non-project",
            deny_count=len(unique_deny),
            path=str(settings_path),
        )
    except OSError as e:
        logger.error("Failed to write settings.local.json", error=str(e))
