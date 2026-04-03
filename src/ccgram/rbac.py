"""RBAC middleware for Brain multi-user access control.

Checks user permissions via Supabase (schema brain) before forwarding
messages to Claude Code. Generates settings.local.json with deny rules
per-user per-project.

BRAIN FORK: added for multi-user permission management.
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

# --- Thread-to-project mapping cache ---

_thread_project_cache: dict[int, tuple[float, str | None]] = {}


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


# --- Cache ---

_CACHE_TTL = 300  # 5 minutes
_profile_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_project_cache: dict[tuple[int, str], tuple[float, str | None]] = {}


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
        _project_cache.clear()
    else:
        _profile_cache.pop(telegram_id, None)
        keys_to_remove = [k for k in _project_cache if k[0] == telegram_id]
        for k in keys_to_remove:
            _project_cache.pop(k, None)


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
    ):
        self.allowed = allowed
        self.reason = reason
        self.is_owner = is_owner
        self.can_write = can_write
        self.can_delete = can_delete
        self.project_slug = project_slug
        self.display_name = display_name
        self.permissions = permissions or []


async def check_access(
    telegram_id: int, project_slug: str | None = None
) -> AccessResult:
    """Check if a user has access, optionally to a specific project."""
    if not BRAIN_CONTEXT:
        # RBAC not configured, allow all (backwards compatible)
        return AccessResult(allowed=True, reason="rbac_disabled", is_owner=True)

    profile = await get_user_profile(telegram_id)

    if profile is None:
        return AccessResult(allowed=False, reason="user_not_found")

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
        )

    if project_slug is not None:
        project_access = projects.get(project_slug)
        if project_access is None:
            return AccessResult(
                allowed=False,
                reason="no_project_access",
                display_name=display_name,
                project_slug=project_slug,
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
        )

    # No project specified, user exists and is active
    return AccessResult(
        allowed=True,
        reason="member",
        is_owner=False,
        display_name=display_name,
        permissions=permissions,
    )


# --- settings.local.json generation ---

def generate_settings_local(access: AccessResult, cwd: str) -> None:
    """Write .claude/settings.local.json with deny rules for this user.

    Owner: no file generated (full access).
    Member: deny rules based on can_write/can_delete and permissions.
    """
    if access.is_owner:
        # Remove any leftover settings.local.json from previous restricted user
        settings_path = Path(cwd) / ".claude" / "settings.local.json"
        if settings_path.exists():
            try:
                settings_path.unlink()
            except OSError:
                pass
        return

    deny_rules: list[str] = []

    # If no write access to this project
    if not access.can_write:
        deny_rules.extend([
            "Edit(*)",
            "Write(*)",
            "Bash(git commit *)",
            "Bash(git push *)",
            "Bash(git add *)",
        ])

    # If no delete access
    if not access.can_delete:
        deny_rules.extend([
            "Bash(rm *)",
            "Bash(git push --force *)",
            "Bash(git reset *)",
        ])

    # If no deploy permission
    if "deploy" not in access.permissions:
        deny_rules.append("Bash(vercel *)")

    # If no MCP write permission
    if "mcp:write" not in access.permissions:
        deny_rules.extend([
            "mcp__supabase__apply_migration",
            "mcp__supabase__execute_sql",
            "mcp__n8n__n8n_create_workflow",
            "mcp__n8n__n8n_update_*",
            "mcp__slack__slack_post_message",
        ])

    # If no MCP destroy permission
    if "mcp:destroy" not in access.permissions:
        deny_rules.extend([
            "mcp__supabase__execute_sql",
            "mcp__n8n__n8n_delete_workflow",
        ])

    if not deny_rules:
        return

    settings_dir = Path(cwd) / ".claude"
    settings_dir.mkdir(exist_ok=True)

    settings = {"permissions": {"deny": deny_rules}}

    settings_path = settings_dir / "settings.local.json"
    try:
        settings_path.write_text(json.dumps(settings, indent=2))
        logger.info(
            "RBAC settings.local.json written",
            user=access.display_name,
            project=access.project_slug,
            deny_count=len(deny_rules),
            path=str(settings_path),
        )
    except OSError as e:
        logger.error("Failed to write settings.local.json", error=str(e))
