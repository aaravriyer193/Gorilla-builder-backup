"""
E2B Sandbox Manager — Speed-optimised
========================================================

Speed fixes applied:
  - Parallel port health checks instead of sequential curl calls
  - npm run dev readiness detected via log polling (no fixed sleep)
  - Tar upload chunk size increased 10× (50KB → 500KB, fewer round-trips)
  - File tree cached on session (30s TTL) to avoid full FS dump every turn
  - Reviewer gated: only fires on substantial new builds, not edits/debug
"""

from __future__ import annotations

import os
import re
import io
import time
import base64
import tarfile
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Set, Tuple

try:
    from e2b import Sandbox
except ImportError:
    Sandbox = None
    print("⚠️ e2b package not installed. Run: pip install e2b")

from backend.ai.lineage_agent import LineageAgent, log_agent, review_output

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
E2B_API_KEY = os.getenv("E2B_API_KEY", "")
SANDBOX_TEMPLATE = os.getenv("E2B_TEMPLATE", "base")
IDLE_TIMEOUT_S = 900
BILLING_TOKENS_PER_HOUR = 50_000
BILLING_TICK_S = 1
APP_DIR = "/home/user/app"
MAX_COMMANDS_PER_TURN = 1600
MAX_TURNS_PER_REQUEST = 80
SYNC_MARKER = "/tmp/.gorilla_sync_marker"
FILE_READ_SENTINEL = "═══GORILLA_FILE_BOUNDARY_9f8c═══"
FILE_CONTENT_SENTINEL = "═══GORILLA_CONTENT_START_9f8c═══"

DEFAULT_PREVIEW_PORT = 8080
DEFAULT_SERVER_PORT = 3000

# Speed fix: poll dev.log for this pattern instead of sleeping a fixed duration
READY_SIGNAL_CMD = (
    "timeout 25 bash -c '"
    "while ! grep -qE \"ready in|Local:|listening on\" /tmp/dev.log 2>/dev/null; "
    "do sleep 0.4; done; "
    "tail -5 /tmp/dev.log 2>/dev/null'"
)

# Keywords that indicate an edit/debug request — reviewer skipped for these
_EDIT_KEYWORDS = frozenset([
    "fix", "change", "update", "edit", "debug", "adjust", "tweak",
    "modify", "rename", "move", "delete", "remove", "add a", "add the",
])


# ---------------------------------------------------------------------------
# Path stripping helper
# ---------------------------------------------------------------------------
def _strip_app_prefix(p: str) -> str:
    p = p.strip()
    if p.startswith(APP_DIR + "/"):
        return p[len(APP_DIR) + 1:]
    app_no_slash = APP_DIR.lstrip("/") + "/"
    if p.startswith(app_no_slash):
        return p[len(app_no_slash):]
    cleaned = re.sub(r"^.*?/app/", "", p)
    if cleaned != p:
        return cleaned
    return p


# ---------------------------------------------------------------------------
# Binary path detector
# ---------------------------------------------------------------------------
def _is_binary_path(p: str) -> bool:
    BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
        ".svg", ".woff", ".woff2", ".ttf", ".eot", ".otf",
        ".mp3", ".mp4", ".wav", ".ogg", ".pdf", ".zip",
    }
    ext = os.path.splitext(p)[1].lower()
    return ext in BINARY_EXTS


# ---------------------------------------------------------------------------
# Command classifier — for UI activity cards
# ---------------------------------------------------------------------------
def classify_command(cmd: str) -> Dict[str, str]:
    c = cmd.strip()
    low = c.lower()
    m = re.match(r"cat\s+>>?\s+['\"]?([^\s'\"<]+)['\"]?\s+<<", c)
    if m:
        return {"verb": "edit", "target": m.group(1), "short": f"Edit {m.group(1)}"}
    if low.startswith("mkdir"):
        m = re.search(r"mkdir\s+(?:-p\s+)?['\"]?([^\s'\"]+)", c)
        return {"verb": "create", "target": m.group(1) if m else "", "short": f"Create dir {m.group(1) if m else ''}"}
    if low.startswith("rm"):
        m = re.search(r"rm\s+(?:-\S+\s+)*['\"]?([^\s'\"]+)", c)
        return {"verb": "delete", "target": m.group(1) if m else "", "short": f"Delete {m.group(1) if m else ''}"}
    if low.startswith("npm install") or low.startswith("npm i "):
        return {"verb": "install", "target": "", "short": "Install dependencies"}
    if low.startswith("npm run"):
        m = re.search(r"npm\s+run\s+(\S+)", c)
        return {"verb": "execute", "target": m.group(1) if m else "", "short": f"Run {m.group(1) if m else 'script'}"}
    if low.startswith("curl"):
        if "supabase.com" in c and "database/query" in c:
            return {"verb": "database", "target": "migration", "short": "Run SQL migration"}
        return {"verb": "fetch", "target": "", "short": "API call"}
    if low.startswith("cat ") or low.startswith("tail ") or low.startswith("head "):
        m = re.match(r"\S+\s+(?:-\S+\s+)*['\"]?([^\s'\"]+)", c)
        return {"verb": "read", "target": m.group(1) if m else "", "short": f"Read {m.group(1) if m else 'file'}"}
    if low.startswith("grep ") or low.startswith("find "):
        return {"verb": "scan", "target": "", "short": "Search files"}
    if low.startswith("python"):
        return {"verb": "execute", "target": "python", "short": "Run Python script"}
    if low.startswith("sed "):
        return {"verb": "edit", "target": "", "short": "Edit file"}
    first = c.split()[0] if c.split() else "run"
    return {"verb": "execute", "target": "", "short": f"Execute {first}"}


@dataclass
class SandboxSession:
    project_id: str
    sandbox: Any
    sandbox_id: str
    owner_id: str
    preview_port: int = DEFAULT_PREVIEW_PORT
    url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_bill_at: float = field(default_factory=time.time)
    total_billed_tokens: int = 0
    deps_installed: bool = False
    content_hashes: Dict[str, str] = field(default_factory=dict)
    _billing_task: Optional[asyncio.Task] = field(default=None, repr=False)
    agent: Optional[Any] = field(default=None, repr=False)
    # Speed fix: cached file tree to avoid full FS dump every agent turn
    _cached_tree: Dict[str, str] = field(default_factory=dict, repr=False)
    _tree_cached_at: float = field(default=0.0, repr=False)


class E2BSandboxManager:
    def __init__(
        self,
        db_upsert_fn: Callable,
        db_delete_fn: Callable,
        db_upsert_batch_fn: Optional[Callable],
        add_tokens_fn: Callable,
        emit_log_fn: Callable,
        emit_status_fn: Callable,
        emit_file_changed_fn: Callable,
        emit_file_deleted_fn: Callable,
        fetch_files_fn: Callable,
        list_db_paths_fn: Callable,
        progress_bus: Any = None,
    ):
        self._sessions: Dict[str, SandboxSession] = {}
        self._db_upsert = db_upsert_fn
        self._db_delete = db_delete_fn
        self._db_upsert_batch = db_upsert_batch_fn
        self._add_tokens = add_tokens_fn
        self._emit_log = emit_log_fn
        self._emit_status = emit_status_fn
        self._emit_file_changed = emit_file_changed_fn
        self._emit_file_deleted = emit_file_deleted_fn
        self._fetch_files = fetch_files_fn
        self._list_db_paths = list_db_paths_fn
        self._progress_bus = progress_bus
        self._idle_monitor_task: Optional[asyncio.Task] = None
        self._boot_locks: Dict[str, asyncio.Lock] = {}
        self._turn_locks: Dict[str, asyncio.Lock] = {}
        self._activity_counter: Dict[str, int] = {}

    # -----------------------------------------------------------
    # Emit helpers
    # -----------------------------------------------------------
    def _emit(self, project_id: str, event: Dict[str, Any]) -> None:
        if self._progress_bus:
            self._progress_bus.emit(project_id, event)

    def _next_activity_id(self, project_id: str) -> str:
        self._activity_counter[project_id] = self._activity_counter.get(project_id, 0) + 1
        return f"act_{self._activity_counter[project_id]}"

    def _emit_activity_start(self, project_id, activity_id, verb, target, short):
        self._emit(project_id, {
            "type": "activity_start", "id": activity_id,
            "verb": verb, "target": target, "short": short,
        })

    def _emit_activity_chunk(self, project_id, activity_id, stream, text):
        if text:
            self._emit(project_id, {
                "type": "activity_chunk", "id": activity_id,
                "stream": stream, "text": text,
            })

    def _emit_activity_end(self, project_id, activity_id, exit_code):
        self._emit(project_id, {
            "type": "activity_end", "id": activity_id,
            "exit_code": exit_code, "ok": exit_code == 0,
        })

    def _emit_narration(self, project_id, text):
        if text:
            self._emit(project_id, {"type": "narration", "text": text})

    # -----------------------------------------------------------
    # Port detection
    # -----------------------------------------------------------
    @staticmethod
    def _detect_preview_port(file_tree: Dict[str, str]) -> int:
        for path in ("vite.config.ts", "vite.config.js", "vite.config.mjs"):
            if path in file_tree and file_tree[path]:
                m = re.search(r"port\s*:\s*(\d+)", file_tree[path])
                if m:
                    try:
                        return int(m.group(1))
                    except ValueError:
                        pass
        pkg = file_tree.get("package.json", "")
        m = re.search(r"--port[= ]+(\d+)", pkg)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return DEFAULT_PREVIEW_PORT

    def _sandbox_url_for_port(self, sbx, port: int) -> str:
        try:
            host = sbx.get_host(port)
            return f"https://{host}"
        except Exception:
            return f"https://{sbx.sandbox_id}-{port}.e2b.dev"

    # -----------------------------------------------------------
    # Batched boot-time file upload via tar
    # Speed fix: chunk size increased from 50KB to 500KB = 10× fewer SSH round-trips
    # -----------------------------------------------------------
    @staticmethod
    def _build_tar_base64(file_tree: Dict[str, str]) -> Tuple[str, int, Dict[str, str]]:
        buf = io.BytesIO()
        hashes: Dict[str, str] = {}
        count = 0
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for path, content in file_tree.items():
                if not path or content is None:
                    continue
                if any(x in path for x in ["package-lock.json", "yarn.lock", "node_modules/", ".git/"]):
                    continue
                data = content.encode("utf-8", errors="replace")
                if len(data) > 1_000_000:
                    continue
                ti = tarfile.TarInfo(name=path)
                ti.size = len(data)
                ti.mtime = int(time.time())
                tf.addfile(ti, io.BytesIO(data))
                hashes[path] = hashlib.md5(data).hexdigest()
                count += 1
        raw = buf.getvalue()
        if len(raw) > 5 * 1024 * 1024:
            return "", 0, {}
        return base64.b64encode(raw).decode("ascii"), count, hashes

    def _upload_files_fast(self, sbx, file_tree: Dict[str, str]) -> Tuple[int, Dict[str, str]]:
        b64, count, hashes = self._build_tar_base64(file_tree)
        if not b64:
            return self._upload_files_slow(sbx, file_tree)
        try:
            sbx.commands.run("rm -f /tmp/bundle.b64 && touch /tmp/bundle.b64", timeout=5)
            # Speed fix: 500KB chunks instead of 50KB — reduces round-trips by 10×
            for i in range(0, len(b64), 500_000):
                chunk = b64[i:i + 500_000]
                meta = base64.b64encode(chunk.encode()).decode()
                sbx.commands.run(f"echo '{meta}' | base64 -d >> /tmp/bundle.b64", timeout=10)
            result = sbx.commands.run(
                f"mkdir -p {APP_DIR} && cat /tmp/bundle.b64 | base64 -d | tar -xzf - -C {APP_DIR} && rm -f /tmp/bundle.b64",
                timeout=60,
            )
            if result.exit_code != 0:
                return self._upload_files_slow(sbx, file_tree)
            return count, hashes
        except Exception as e:
            print(f"⚠️ Batched upload failed: {e}; falling back")
            return self._upload_files_slow(sbx, file_tree)

    @staticmethod
    def _upload_files_slow(sbx, file_tree: Dict[str, str]) -> Tuple[int, Dict[str, str]]:
        hashes: Dict[str, str] = {}
        count = 0
        dirs_created: Set[str] = set()
        for path, content in file_tree.items():
            if not path or content is None:
                continue
            if any(x in path for x in ["package-lock.json", "yarn.lock", "node_modules/", ".git/"]):
                continue
            full = f"{APP_DIR}/{path}"
            dirp = "/".join(full.split("/")[:-1])
            if dirp and dirp not in dirs_created:
                sbx.commands.run(f"mkdir -p '{dirp}'", timeout=5)
                dirs_created.add(dirp)
            safe = content.replace("GORILLA_EOF", "GORILLA__EOF")
            sbx.commands.run(f"cat > '{full}' << 'GORILLA_EOF'\n{safe}\nGORILLA_EOF", timeout=15)
            hashes[path] = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
            count += 1
        return count, hashes

    # -----------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------
    async def ensure_running(self, project_id: str, env_vars: Dict[str, str], owner_id: str) -> SandboxSession:
        if project_id in self._sessions:
            s = self._sessions[project_id]
            s.last_activity = time.time()
            return s
        self._boot_locks.setdefault(project_id, asyncio.Lock())
        async with self._boot_locks[project_id]:
            if project_id in self._sessions:
                s = self._sessions[project_id]
                s.last_activity = time.time()
                return s
            files = await self._fetch_files(project_id)
            return await self._do_boot(project_id, files, env_vars, owner_id)

    async def _do_boot(self, project_id, file_tree, env_vars, owner_id):
        if not Sandbox:
            raise RuntimeError("E2B SDK not installed")
        if not E2B_API_KEY:
            raise RuntimeError("E2B_API_KEY not set")

        self._emit_log(project_id, "sandbox", f"Booting sandbox ({SANDBOX_TEMPLATE})...")
        self._emit_status(project_id, "Booting Sandbox...")

        def _boot_sync():
            sbx = Sandbox(template=SANDBOX_TEMPLATE, api_key=E2B_API_KEY, timeout=3600)
            sbx.commands.run(f"mkdir -p {APP_DIR}", timeout=5)
            return sbx

        try:
            sbx = await asyncio.to_thread(_boot_sync)
        except Exception as e:
            self._emit_log(project_id, "sandbox", f"Boot failed: {e}")
            raise

        self._emit_log(project_id, "sandbox", "Uploading project files...")
        file_count, hashes = await asyncio.to_thread(self._upload_files_fast, sbx, file_tree)
        self._emit_log(project_id, "sandbox", f"Mounted {file_count} text files")

        binary_count = await self._restore_binary_files(sbx, project_id, file_tree)
        if binary_count:
            self._emit_log(project_id, "sandbox", f"Restored {binary_count} binary assets")

        if env_vars:
            try:
                env_lines = "\n".join(f"{k}={v}" for k, v in env_vars.items() if v)
                shell_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items() if v)
                await asyncio.to_thread(
                    sbx.commands.run,
                    f"cat > {APP_DIR}/.env << 'GORILLA_EOF'\n{env_lines}\nGORILLA_EOF && "
                    f"cat > {APP_DIR}/.gorilla_env << 'GORILLA_EOF'\n{shell_lines}\nGORILLA_EOF",
                )
            except Exception as e:
                self._emit_log(project_id, "sandbox", f"env write warning: {e}")

        inject_error_reporter = (
            r"grep -q '__gorilla_errors' " + APP_DIR + r"/index.html 2>/dev/null || "
            r"sed -i 's|</head>|<script>"
            r"(function(){"
            r"function send(d){try{fetch(\"/api/__gorilla_errors\",{method:\"POST\","
            r"headers:{\"Content-Type\":\"application/json\"},body:JSON.stringify(d)})}catch(e){}}"
            r"window.onerror=function(m,s,l,c,e){"
            r"send({message:m,source:s+\":\"+l,type:\"error\",stack:e&&e.stack?e.stack.slice(0,300):\"\"});"
            r"return false;};"
            r"window.addEventListener(\"unhandledrejection\",function(e){"
            r"send({message:e.reason&&e.reason.message?e.reason.message:String(e.reason),"
            r"type:\"unhandledrejection\"});});"
            r"window.addEventListener(\"error\",function(e){"
            r"if(e.target&&e.target!==window){"
            r"send({message:\"Resource failed: \"+(e.target.src||e.target.href),type:\"resource\"});}},true);"
            r"})();"
            r"<\/script><\/head>|' " + APP_DIR + r"/index.html"
        )
        try:
            await asyncio.to_thread(sbx.commands.run, inject_error_reporter, timeout=5)
            log_agent("agent", "Injected browser error reporter into index.html", project_id)
        except Exception as e:
            log_agent("agent", f"Error reporter inject skipped: {e}", project_id)

        # node_modules pre-baked in template — always treated as cached
        deps_cached = True
        preview_port = self._detect_preview_port(file_tree)

        session = SandboxSession(
            project_id=project_id, sandbox=sbx, sandbox_id=sbx.sandbox_id,
            owner_id=owner_id, preview_port=preview_port,
            url=self._sandbox_url_for_port(sbx, preview_port),
            deps_installed=deps_cached, content_hashes=hashes,
        )
        self._sessions[project_id] = session

        session._billing_task = asyncio.create_task(self._billing_loop(project_id))
        if not self._idle_monitor_task or self._idle_monitor_task.done():
            self._idle_monitor_task = asyncio.create_task(self._idle_monitor())

        self._emit_log(project_id, "sandbox", f"Sandbox ready: {session.url}")
        self._emit_status(project_id, "Session Started..")
        self._emit(project_id, {"type": "sandbox_url", "url": session.url})
        return session

    async def _restore_binary_files(self, sbx, project_id: str, file_tree: Dict[str, str]) -> int:
        count = 0
        for path, content in file_tree.items():
            if not _is_binary_path(path):
                continue
            if not content or not content.startswith("http"):
                continue
            full_path = f"{APP_DIR}/{path}"
            dirp = "/".join(full_path.split("/")[:-1])
            try:
                result = await asyncio.to_thread(
                    sbx.commands.run,
                    f"mkdir -p '{dirp}' && curl -sL --max-time 30 '{content}' -o '{full_path}' && echo OK",
                    timeout=35,
                )
                if "OK" in (result.stdout or ""):
                    count += 1
                    log_agent("agent", f"Restored binary: {path}", project_id)
                else:
                    log_agent("agent", f"Binary restore may have failed: {path}", project_id)
            except Exception as e:
                log_agent("agent", f"Failed to restore binary {path}: {e}", project_id)
        return count

    async def kill(self, project_id: str) -> None:
        session = self._sessions.pop(project_id, None)
        if not session:
            self._boot_locks.pop(project_id, None)
            self._turn_locks.pop(project_id, None)
            return
        now = time.time()
        elapsed = (now - session.last_bill_at) / 3600.0
        if elapsed > 0.01:
            prorated = int(BILLING_TOKENS_PER_HOUR * elapsed)
            if prorated > 0:
                try:
                    self._add_tokens(session.owner_id, prorated)
                except Exception as e:
                    print(f"⚠️ Billing error: {e}")
        if session._billing_task and not session._billing_task.done():
            session._billing_task.cancel()
        try:
            await asyncio.to_thread(session.sandbox.kill)
        except Exception as e:
            print(f"⚠️ Kill error: {e}")
        self._boot_locks.pop(project_id, None)
        self._turn_locks.pop(project_id, None)
        self._emit_status(project_id, "Sandbox Offline")

    def is_running(self, project_id: str) -> bool:
        return project_id in self._sessions

    def get_session(self, project_id: str) -> Optional[SandboxSession]:
        s = self._sessions.get(project_id)
        if s:
            s.last_activity = time.time()
        return s

    def get_preview_url(self, project_id: str) -> Optional[str]:
        s = self._sessions.get(project_id)
        return s.url if s else None

    # -----------------------------------------------------------
    # Console error polling
    # -----------------------------------------------------------
    async def _poll_console_errors(self, project_id: str) -> str:
        session = self._sessions.get(project_id)
        if not session:
            return ""
        try:
            result = await asyncio.to_thread(
                session.sandbox.commands.run,
                "curl -s http://localhost:3000/api/__gorilla_errors 2>/dev/null",
                timeout=5,
            )
            raw = (result.stdout or "").strip()
            if not raw or raw in ("[]", ""):
                return ""
            errors = json.loads(raw)
            if not errors:
                return ""
            lines = []
            for e in errors[:10]:
                msg = e.get("message", str(e))
                src = e.get("source", "")
                etype = e.get("type", "error")
                stack = e.get("stack", "")
                entry = f"  [{etype}] {msg}"
                if src:
                    entry += f" @ {src}"
                if stack:
                    entry += f"\n    {stack[:200]}"
                lines.append(entry)
            return "BROWSER CONSOLE ERRORS (fix these):\n" + "\n".join(lines)
        except Exception:
            return ""

    # -----------------------------------------------------------
    # Speed fix: parallel port health checks
    # Both ports are checked concurrently, then we merge the results.
    # Previously: sequential curl calls with up to 30s total wait.
    # Now: both checks fire at once, total wait = max(fe_wait, api_wait).
    # -----------------------------------------------------------
    async def _check_ports_parallel(
        self, session: SandboxSession, project_id: str
    ) -> Tuple[bool, str, bool, str]:
        """
        Returns (frontend_ok, fe_content, api_ok, api_content).
        Polls up to ~24s (12 attempts × 2s) per port, concurrently.
        """
        async def check_port(port: int) -> Tuple[bool, str]:
            for _ in range(12):
                await asyncio.sleep(2)
                try:
                    r = await asyncio.to_thread(
                        session.sandbox.commands.run,
                        f"curl -s --max-time 3 http://localhost:{port} 2>/dev/null | head -10",
                        timeout=5,
                    )
                    content = (r.stdout or "").strip()
                    if content and "Cannot GET" not in content:
                        return True, content
                    if content:
                        return False, content  # got a response, but bad
                except Exception:
                    pass
            return False, ""

        (fe_ok, fe_content), (api_ok, api_content) = await asyncio.gather(
            check_port(session.preview_port),
            check_port(DEFAULT_SERVER_PORT),
        )
        return fe_ok, fe_content, api_ok, api_content

    # -----------------------------------------------------------
    # Agent turn — mini-SWE-agent loop
    # -----------------------------------------------------------
    async def run_agent_turn(
        self, project_id, user_request, user_id, env_vars,
        chat_history=None, gorilla_proxy_url="", has_supabase=False,
        is_debug=False, error_context="", image_b64=None,
        on_assistant_message=None, agent_skills=None,
    ) -> Dict[str, Any]:
        self._turn_locks.setdefault(project_id, asyncio.Lock())
        async with self._turn_locks[project_id]:
            return await self._do_run_agent_turn(
                project_id, user_request, user_id, env_vars,
                chat_history, gorilla_proxy_url, has_supabase,
                is_debug, error_context, image_b64, on_assistant_message,
                agent_skills,
            )

    async def _do_run_agent_turn(
        self, project_id, user_request, user_id, env_vars,
        chat_history, gorilla_proxy_url, has_supabase, is_debug,
        error_context, image_b64, on_assistant_message, agent_skills=None,
    ) -> Dict[str, Any]:
        try:
            session = await self.ensure_running(project_id, env_vars, user_id)
        except Exception as e:
            self._emit_log(project_id, "system", f"Sandbox boot failed: {e}")
            self._emit_status(project_id, "Fatal Error")
            return {"ok": False, "error": str(e), "commands": [],
                    "tokens": 0, "final_message": "", "turns": 0}

        try:
            await asyncio.to_thread(session.sandbox.commands.run, f"touch {SYNC_MARKER}")
        except Exception:
            pass

        # Auto-kill ports before agent starts
        try:
            await asyncio.to_thread(
                session.sandbox.commands.run,
                "pkill -f 'vite' 2>/dev/null; pkill -f 'npm run dev' 2>/dev/null; "
                "pkill -f 'node server' 2>/dev/null; sleep 1",
                timeout=10,
            )
        except Exception:
            pass

        if session.agent is None:
            session.agent = LineageAgent(project_id)
        agent = session.agent

        all_commands: List[str] = []
        final_message = ""
        total_tokens = 0
        turn_count = 0
        previous_output: Optional[str] = None
        last_raw_output = ""

        # Speed fix: use cached file tree (TTL 30s) to avoid a full FS dump
        # at the start of every agent turn. Cache is invalidated after each sync.
        now = time.time()
        if (now - session._tree_cached_at) > 30 or not session._cached_tree:
            session._cached_tree = await self._read_tree_from_sandbox(project_id)
            session._tree_cached_at = time.time()
        tree = session._cached_tree

        for turn in range(MAX_TURNS_PER_REQUEST):
            turn_count = turn + 1
            log_agent("agent", f"Turn {turn_count}/{MAX_TURNS_PER_REQUEST}", project_id)

            result = await agent.run(
                user_request=user_request,
                file_tree=tree if turn == 0 else {},
                chat_history=chat_history if turn == 0 else None,
                gorilla_proxy_url=gorilla_proxy_url,
                has_supabase=has_supabase,
                is_debug=is_debug,
                error_context=error_context if turn == 0 else "",
                image_b64=image_b64 if turn == 0 else None,
                previous_command_output=previous_output,
                agent_skills=agent_skills if turn == 0 else None,
            )
            turn_tokens = result.get("tokens", 0) - total_tokens
            total_tokens = result.get("tokens", 0)

            if turn_tokens > 0:
                try:
                    self._add_tokens(session.owner_id, turn_tokens)
                    self._emit(project_id, {"type": "token_usage", "tokens": turn_tokens})
                except Exception:
                    pass

            # Save generated plan as .gorilla/todo.md on first turn
            if turn == 0 and hasattr(agent, '_plan_injected') and agent._plan_injected:
                try:
                    first_user_msg = agent.messages[1].get("content", "") if len(agent.messages) > 1 else ""
                    if isinstance(first_user_msg, str) and "follow it step by step" in first_user_msg:
                        plan_start = first_user_msg.find("# Task:")
                        if plan_start == -1:
                            plan_start = first_user_msg.find("- [ ]")
                        if plan_start > 0:
                            plan_content = first_user_msg[plan_start:]
                            for marker in ["\nRecent conversation:", "\nProject files"]:
                                idx = plan_content.find(marker)
                                if idx > 0:
                                    plan_content = plan_content[:idx]
                            await asyncio.to_thread(
                                session.sandbox.commands.run,
                                f"mkdir -p {APP_DIR}/.gorilla && cat > {APP_DIR}/.gorilla/todo.md << 'GORILLA_EOF'\n{plan_content.strip()}\nGORILLA_EOF",
                                timeout=5,
                            )
                            log_agent("agent", "Saved plan to .gorilla/todo.md", project_id)
                except Exception as e:
                    log_agent("agent", f"Failed to save todo.md: {e}", project_id)

            msg = result.get("message", "")
            if msg:
                final_message = msg
                self._emit_narration(project_id, msg)
                if on_assistant_message:
                    try:
                        on_assistant_message(msg)
                    except Exception:
                        pass

            commands = result.get("commands", [])

            if result.get("done", False) and not commands:
                break

            if not commands and not result.get("done", False):
                previous_output = (
                    "OBSERVATION:\n"
                    "No commands detected. Provide your next step in a "
                    "```bash``` code block, or write GORILLA_DONE if finished."
                )
                continue

            # Auto-background fix
            fixed_commands = []
            for cmd in commands:
                if re.search(r'npm run dev\s*$', cmd.strip()) and '&' not in cmd:
                    cmd = cmd.rstrip() + ' > /tmp/dev.log 2>&1 &'
                    log_agent("agent", "Auto-backgrounded npm run dev", project_id)
                elif re.search(r'npm run dev\s*>', cmd.strip()) and '&' not in cmd:
                    cmd = cmd.rstrip() + ' &'
                    log_agent("agent", "Auto-backgrounded npm run dev (had redirect)", project_id)
                fixed_commands.append(cmd)
            commands = fixed_commands

            all_commands.extend(commands)
            cmd_results = await self._execute_commands_streaming(project_id, commands)

            # Build OBSERVATION
            output_parts = []
            for r in cmd_results:
                stdout = (r.get("stdout") or "").strip()
                stderr = (r.get("stderr") or "").strip()
                exit_code = r.get("exit_code", 0)
                if stdout:
                    output_parts.append(stdout[:3000])
                if stderr:
                    output_parts.append(f"STDERR: {stderr[:1500]}")
                if exit_code != 0:
                    output_parts.append(f"[exit code: {exit_code}]")

            raw_output = "\n".join(output_parts)[:6000] if output_parts else "Your command ran successfully and did not produce any output."
            last_raw_output = raw_output

            # Speed fix: after npm run dev, poll dev.log for the ready signal
            # instead of sleeping a fixed 5s. Exits as soon as Vite prints
            # "ready in Xms" — typically 1-3s instead of always waiting 5s+.
            for cmd in commands:
                if "npm run dev" in cmd:
                    try:
                        log_agent("agent", "Waiting for Vite ready signal...", project_id)
                        ready_result = await asyncio.to_thread(
                            session.sandbox.commands.run,
                            READY_SIGNAL_CMD,
                            timeout=30,
                        )
                        ready_output = (ready_result.stdout or "").strip()
                        log_agent("agent", f"Vite ready: {ready_output[:80]}", project_id)

                        # 1. Check Vite compile errors from log
                        error_check = await asyncio.to_thread(
                            session.sandbox.commands.run,
                            "grep -i -E 'error|failed|Cannot find|could not be resolved|SyntaxError' "
                            "/tmp/dev.log 2>/dev/null | grep -v 'node_modules' | head -30",
                            timeout=10,
                        )
                        vite_errors = (error_check.stdout or "").strip()
                        if vite_errors:
                            raw_output += f"\n\nVITE COMPILE ERRORS:\n{vite_errors}"

                        # 2. Speed fix: check both ports concurrently
                        fe_ok, fe_content, api_ok, api_content = await self._check_ports_parallel(
                            session, project_id
                        )

                        if not fe_ok:
                            if fe_content:
                                raw_output += f"\n\nWARNING: Frontend returned 'Cannot GET' — check vite.config and index.html.\nResponse: {fe_content[:200]}"
                            else:
                                raw_output += "\n\nWARNING: Frontend port returned empty response — Vite may have crashed. Check /tmp/dev.log."

                        if "Cannot GET /" in api_content:
                            raw_output += "\n\nNOTE: Express on :3000 is running (no root GET / route is normal)."
                        elif not api_content:
                            raw_output += "\n\nWARNING: Express on :3000 returned nothing — server.js may have crashed. Check /tmp/dev.log."

                        # 3. Ensure Express has the console-error drain endpoint
                        inject_express = (
                            f"grep -q '__gorilla_errors' {APP_DIR}/server.js 2>/dev/null || "
                            f"cat >> {APP_DIR}/server.js << 'GORILLA_EOF'\n"
                            f"// Gorilla browser error tunnel\n"
                            f"const _gErrs = [];\n"
                            f"app.post('/api/__gorilla_errors', (req, res) => {{\n"
                            f"  if (req.body) {{ _gErrs.push(req.body); if (_gErrs.length > 50) _gErrs.shift(); }}\n"
                            f"  res.json({{ok: true}});\n"
                            f"}});\n"
                            f"app.get('/api/__gorilla_errors', (req, res) => {{\n"
                            f"  res.json(_gErrs.splice(0));\n"
                            f"}});\n"
                            f"GORILLA_EOF"
                        )
                        await asyncio.to_thread(
                            session.sandbox.commands.run, inject_express, timeout=5
                        )

                        log_agent("agent", f"Health check done. fe_ok={fe_ok} api_ok={api_ok} vite_errors={bool(vite_errors)}", project_id)
                    except Exception as e:
                        log_agent("agent", f"Health check failed: {e}", project_id)
                    break

            # Linter-in-the-loop
            for cmd in commands:
                m = re.search(r"cat\s+>\s+['\"]?(\S+\.tsx?)['\"]?\s+<<", cmd)
                if m:
                    lint_path = m.group(1)
                    try:
                        lint_result = await asyncio.to_thread(
                            session.sandbox.commands.run,
                            f"cd {APP_DIR} && npx tsc --noEmit {lint_path} 2>&1 | head -20",
                            timeout=15,
                        )
                        lint_out = (lint_result.stdout or "").strip()
                        if lint_out and ("error TS" in lint_out or "Error" in lint_out):
                            raw_output += f"\n\nLINT ERRORS in {lint_path}:\n{lint_out[:800]}"
                    except Exception:
                        pass

            # Poll browser console errors
            console_errs = await self._poll_console_errors(project_id)
            if console_errs:
                raw_output += f"\n\n{console_errs}"
                log_agent("agent", f"Console errors captured: {console_errs[:150]}", project_id)

            last_raw_output = raw_output
            previous_output = raw_output

            if result.get("done", False):
                break

        # Speed fix: reviewer only fires on substantial new builds.
        # Skipped for debug runs, short sessions, and edit/fix/tweak requests.
        # This eliminates a full LLM round-trip (~2-4s) for the common case.
        request_lower = (user_request or "").lower()
        is_edit_request = any(kw in request_lower for kw in _EDIT_KEYWORDS)
        should_review = (
            not is_debug
            and turn_count > 3
            and not is_edit_request
        )

        if should_review:
            try:
                current_tree = await self._read_tree_from_sandbox(project_id)
                # Invalidate cache since we just re-read the tree
                session._cached_tree = current_tree
                session._tree_cached_at = time.time()

                tree_summary = "\n".join(
                    f"  {p}" for p in sorted(current_tree.keys()) if not p.endswith(".b64")
                )
                review_fixes = await review_output(tree_summary, last_raw_output)
                if review_fixes:
                    self._emit_narration(project_id, "Reviewing output for issues...")
                    fix_result = await agent.run(
                        user_request=f"Code review found issues. Fix them:\n{review_fixes}",
                        file_tree={},
                        gorilla_proxy_url=gorilla_proxy_url,
                        has_supabase=has_supabase,
                        is_debug=True,
                        error_context=review_fixes,
                    )
                    total_tokens += fix_result.get("tokens", 0)
                    fix_cmds = fix_result.get("commands", [])
                    if fix_cmds:
                        all_commands.extend(fix_cmds)
                        await self._execute_commands_streaming(project_id, fix_cmds)
            except Exception as e:
                log_agent("agent", f"Reviewer error: {e}", project_id)

        # Sync to database
        self._emit_status(project_id, "Syncing to database...")
        synced, deleted = await self._sync_once(project_id)
        # Invalidate tree cache after sync so next turn reads fresh
        session._tree_cached_at = 0.0
        self._emit_log(project_id, "sync", f"Synced {synced} changed, removed {deleted} deleted")

        url = session.url
        self._emit(project_id, {"type": "sandbox_url", "url": url})

        return {
            "ok": True, "commands": all_commands,
            "tokens": total_tokens,
            "final_message": final_message or "Done.",
            "turns": turn_count, "synced_files": synced,
            "deleted_files": deleted, "preview_url": url,
        }

    # -----------------------------------------------------------
    # Streaming executor
    # -----------------------------------------------------------
    async def _execute_commands_streaming(self, project_id, commands):
        session = self._sessions.get(project_id)
        if not session:
            return [{"command": "N/A", "stdout": "", "stderr": "Sandbox not running", "exit_code": -1}]

        session.last_activity = time.time()
        results: List[Dict[str, Any]] = []

        for cmd in commands[:MAX_COMMANDS_PER_TURN]:
            if not cmd or not cmd.strip():
                continue

            classification = classify_command(cmd)
            activity_id = self._next_activity_id(project_id)
            self._emit_activity_start(
                project_id, activity_id,
                classification["verb"], classification["target"],
                classification["short"],
            )

            if cmd.startswith("cd ") or cmd.startswith("/"):
                effective = cmd
            else:
                effective = f"cd {APP_DIR} && source .gorilla_env 2>/dev/null; {cmd}"

            stdout_buf: List[str] = []
            stderr_buf: List[str] = []

            def on_stdout(line: str):
                if not line:
                    return
                stdout_buf.append(line)
                clean = line.rstrip("\n")[:400]
                if clean.strip():
                    self._emit_activity_chunk(project_id, activity_id, "stdout", clean)

            def on_stderr(line: str):
                if not line:
                    return
                stderr_buf.append(line)
                clean = line.rstrip("\n")[:400]
                if clean.strip():
                    self._emit_activity_chunk(project_id, activity_id, "stderr", clean)

            try:
                exit_code = await asyncio.to_thread(
                    self._run_command_with_streaming,
                    session.sandbox, effective, on_stdout, on_stderr,
                )
            except Exception as e:
                err = str(e)[:200]
                stderr_buf.append(err)
                self._emit_activity_chunk(project_id, activity_id, "stderr", err)
                exit_code = -1

            self._emit_activity_end(project_id, activity_id, exit_code)
            results.append({
                "command": cmd, "stdout": "".join(stdout_buf),
                "stderr": "".join(stderr_buf), "exit_code": exit_code,
            })

        try:
            await asyncio.to_thread(session.sandbox.commands.run, "sync && true")
        except Exception:
            pass
        session.last_activity = time.time()
        return results

    @staticmethod
    def _run_command_with_streaming(sandbox, cmd, on_stdout, on_stderr):
        try:
            result = sandbox.commands.run(cmd, on_stdout=on_stdout, on_stderr=on_stderr, timeout=180)
            return getattr(result, "exit_code", 0)
        except TypeError:
            pass
        try:
            result = sandbox.commands.run(cmd, timeout=180)
            if result.stdout:
                for line in result.stdout.splitlines(keepends=True):
                    on_stdout(line)
            if result.stderr:
                for line in result.stderr.splitlines(keepends=True):
                    on_stderr(line)
            return getattr(result, "exit_code", 0)
        except Exception as e:
            on_stderr(str(e))
            return -1

    # -----------------------------------------------------------
    # Batched tree read — with robust path stripping
    # -----------------------------------------------------------
    async def _read_tree_from_sandbox(self, project_id: str) -> Dict[str, str]:
        session = self._sessions.get(project_id)
        if not session:
            return {}

        dump_cmd = (
            f"find {APP_DIR} -type f "
            f"-not -path '*/node_modules/*' -not -path '*/.git/*' "
            f"-not -path '*/dist/*' -not -name 'package-lock.json' "
            f"-not -name '*.lock' -size -500k -print0 | "
            f'xargs -0 -I {{}} sh -c \''
            f'echo "{FILE_READ_SENTINEL}{{}}"; '
            f'echo "{FILE_CONTENT_SENTINEL}"; '
            f'cat "{{}}" 2>/dev/null; echo ""\''
        )

        try:
            result = await asyncio.to_thread(session.sandbox.commands.run, dump_cmd)
            output = result.stdout or ""
        except Exception:
            return await self._read_tree_slow(project_id)

        if not output.strip():
            return {}

        tree: Dict[str, str] = {}
        for chunk in output.split(FILE_READ_SENTINEL)[1:]:
            if FILE_CONTENT_SENTINEL not in chunk:
                continue
            header, _, body = chunk.partition(FILE_CONTENT_SENTINEL)
            path = _strip_app_prefix(header)
            if not path:
                continue
            content = body[:-1] if body.endswith("\n") else body
            if "\x00" in content[:1000]:
                continue
            tree[path] = content
        return tree

    async def _read_tree_slow(self, project_id: str) -> Dict[str, str]:
        session = self._sessions.get(project_id)
        if not session:
            return {}
        try:
            listing = await asyncio.to_thread(
                session.sandbox.commands.run,
                f"find {APP_DIR} -type f -not -path '*/node_modules/*' "
                f"-not -path '*/.git/*' -not -path '*/dist/*' "
                f"-not -name 'package-lock.json' -not -name '*.lock' -size -500k",
            )
        except Exception:
            return {}
        if not listing.stdout:
            return {}

        paths = [
            _strip_app_prefix(p)
            for p in listing.stdout.strip().split("\n") if p.strip()
        ]

        tree: Dict[str, str] = {}
        for rel in paths[:400]:
            if not rel:
                continue
            try:
                r = await asyncio.to_thread(
                    session.sandbox.commands.run,
                    f"cat '{APP_DIR}/{rel}' 2>/dev/null"
                )
                content = r.stdout or ""
                if "\x00" in content[:1000]:
                    continue
                tree[rel] = content
            except Exception:
                continue
        return tree

    # -----------------------------------------------------------
    # Batched sync — with robust path stripping
    # -----------------------------------------------------------
    async def _sync_once(self, project_id: str) -> Tuple[int, int]:
        session = self._sessions.get(project_id)
        if not session:
            return (0, 0)
        try:
            await asyncio.to_thread(session.sandbox.commands.run, "sync")
        except Exception:
            pass

        changed_dump_cmd = (
            f"find {APP_DIR} -type f -newer {SYNC_MARKER} "
            f"-not -path '*/node_modules/*' -not -path '*/.git/*' "
            f"-not -path '*/dist/*' -not -name 'package-lock.json' "
            f"-not -name '*.lock' -size -500k -print0 | "
            f'xargs -0 -I {{}} sh -c \''
            f'echo "{FILE_READ_SENTINEL}{{}}"; '
            f'echo "{FILE_CONTENT_SENTINEL}"; '
            f'cat "{{}}" 2>/dev/null; echo ""\''
        )

        try:
            result = await asyncio.to_thread(session.sandbox.commands.run, changed_dump_cmd)
            dump = result.stdout or ""
        except Exception as e:
            self._emit_log(project_id, "sync", f"Sync failed: {e}")
            return (0, 0)

        changed_files: Dict[str, str] = {}
        if dump.strip():
            for chunk in dump.split(FILE_READ_SENTINEL)[1:]:
                if FILE_CONTENT_SENTINEL not in chunk:
                    continue
                header, _, body = chunk.partition(FILE_CONTENT_SENTINEL)
                path = _strip_app_prefix(header)
                if not path:
                    continue
                content = body[:-1] if body.endswith("\n") else body
                if "\x00" in content[:1000]:
                    continue
                if len(content) > 500_000:
                    continue
                changed_files[path] = content

        rows: List[Dict[str, Any]] = []
        current_sandbox_paths: Set[str] = set(changed_files.keys())
        for rel, content in changed_files.items():
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
            if session.content_hashes.get(rel) == h:
                continue
            session.content_hashes[rel] = h
            rows.append({"project_id": project_id, "path": rel, "content": content})

        try:
            all_listing = await asyncio.to_thread(
                session.sandbox.commands.run,
                f"find {APP_DIR} -type f -not -path '*/node_modules/*' "
                f"-not -path '*/.git/*' -not -path '*/dist/*' "
                f"-not -name 'package-lock.json' -not -name '*.lock'",
            )
            if all_listing.stdout:
                for p in all_listing.stdout.strip().split("\n"):
                    stripped = _strip_app_prefix(p)
                    if stripped:
                        current_sandbox_paths.add(stripped)
        except Exception:
            pass

        if rows:
            try:
                if self._db_upsert_batch:
                    self._db_upsert_batch("files", rows, on_conflict="project_id,path")
                else:
                    for row in rows:
                        self._db_upsert("files", row, on_conflict="project_id,path")
                for row in rows:
                    self._emit_file_changed(project_id, row["path"])
            except Exception as e:
                self._emit_log(project_id, "sync", f"Upsert error: {e}")

        deleted_count = 0
        try:
            db_paths = set(self._list_db_paths(project_id) or [])
            to_delete = db_paths - current_sandbox_paths - {".env", ".gorilla_env"}
            for p in to_delete:
                try:
                    self._db_delete("files", {"project_id": project_id, "path": p})
                    self._emit_file_deleted(project_id, p)
                    session.content_hashes.pop(p, None)
                    deleted_count += 1
                except Exception:
                    pass
        except Exception:
            pass

        try:
            await asyncio.to_thread(session.sandbox.commands.run, f"touch {SYNC_MARKER}")
        except Exception:
            pass

        return (len(rows), deleted_count)

    # -----------------------------------------------------------
    # Dev server (for /sandbox/start endpoint, NOT used in agent loop)
    # -----------------------------------------------------------
    async def start_dev_server(self, project_id: str) -> Tuple[Optional[str], Optional[str]]:
        session = self._sessions.get(project_id)
        if not session:
            return (None, None)
        session.last_activity = time.time()

        try:
            await asyncio.to_thread(session.sandbox.commands.run,
                "pkill -f vite || true; pkill -f 'npm run dev' || true; pkill -f 'node server' || true")
            await asyncio.sleep(1)
        except Exception:
            pass

        try:
            await asyncio.to_thread(session.sandbox.commands.run,
                f"cd {APP_DIR} && nohup npm run dev > /tmp/vite.log 2>&1 &")
        except Exception:
            pass

        bound = False
        for _ in range(15):
            await asyncio.sleep(1)
            try:
                check = await asyncio.to_thread(session.sandbox.commands.run,
                    f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{session.preview_port} || echo fail")
                if "200" in (check.stdout or "") or "304" in (check.stdout or ""):
                    bound = True
                    break
            except Exception:
                pass

        self._emit(project_id, {"type": "sandbox_url", "url": session.url})
        return (session.url, None)

    # -----------------------------------------------------------
    # File write/delete for editor /save bridge
    # -----------------------------------------------------------
    async def write_file(self, project_id: str, rel_path: str, content: str) -> bool:
        session = self._sessions.get(project_id)
        if not session:
            return False
        full = f"{APP_DIR}/{rel_path}"
        dirp = "/".join(full.split("/")[:-1])
        try:
            safe = content.replace("GORILLA_EOF", "GORILLA__EOF")
            await asyncio.to_thread(session.sandbox.commands.run,
                f"mkdir -p '{dirp}' && cat > '{full}' << 'GORILLA_EOF'\n{safe}\nGORILLA_EOF")
            session.content_hashes[rel_path] = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()
            # Invalidate tree cache so the next agent turn sees the new file
            session._tree_cached_at = 0.0
            session.last_activity = time.time()
            return True
        except Exception as e:
            print(f"⚠️ write_file failed: {e}")
            return False

    async def delete_file(self, project_id: str, rel_path: str) -> bool:
        session = self._sessions.get(project_id)
        if not session:
            return False
        try:
            await asyncio.to_thread(session.sandbox.commands.run, f"rm -f '{APP_DIR}/{rel_path}'")
            session.content_hashes.pop(rel_path, None)
            # Invalidate tree cache
            session._tree_cached_at = 0.0
            session.last_activity = time.time()
            return True
        except Exception:
            return False

    # -----------------------------------------------------------
    # Billing + idle monitor
    # -----------------------------------------------------------
    async def _billing_loop(self, project_id: str) -> None:
        accumulated = 0
        try:
            while True:
                await asyncio.sleep(BILLING_TICK_S)
                session = self._sessions.get(project_id)
                if not session:
                    return
                now = time.time()
                elapsed = (now - session.last_bill_at) / 3600.0
                if elapsed <= 0:
                    continue
                prorated = int(BILLING_TOKENS_PER_HOUR * elapsed)
                if prorated <= 0:
                    continue
                accumulated += prorated
                session.total_billed_tokens += prorated
                session.last_bill_at = now
                if accumulated >= int(BILLING_TOKENS_PER_HOUR / 360):
                    try:
                        self._add_tokens(session.owner_id, accumulated)
                        accumulated = 0
                    except Exception as e:
                        print(f"⚠️ Billing error: {e}")
        except asyncio.CancelledError:
            if accumulated > 0:
                session = self._sessions.get(project_id)
                if session:
                    try:
                        self._add_tokens(session.owner_id, accumulated)
                    except Exception:
                        pass

    async def _idle_monitor(self) -> None:
        while True:
            try:
                now = time.time()
                for pid in [
                    p for p, s in list(self._sessions.items())
                    if (now - s.last_activity) > IDLE_TIMEOUT_S
                ]:
                    print(f"💤 Idle kill: {pid}")
                    try:
                        await self._sync_once(pid)
                    except Exception:
                        pass
                    await self.kill(pid)
            except Exception as e:
                print(f"Idle monitor error: {e}")
            await asyncio.sleep(30)


sandbox_manager: Optional[E2BSandboxManager] = None