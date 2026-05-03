"""
Lineage Agent v12 — Smart Model Routing (Mimo v2.5 for hard turns)
====================================================================

Architecture: mini-SWE-agent (ThoughtAction format, linear message history,
bash execution, raw OBSERVATION feedback).

New in v12:
  #14 Smart model routing — _pick_model() selects Mimo v2.5 for hard turns
       (turn 0 architecture, error recovery, debug spirals, auth/DB work)
       and Deepseek V4 Flash for mechanical turns (file writes, reads,
       boilerplate). ~80% of turns stay cheap, hard turns get intelligence.

New in v11:
  #11 Multi-file turns — up to 3 independent NEW files per bash block,
       cutting average turn count from ~25 to ~10-12.
  #12 Improved planner — groups independent files into batched steps.
  #13 Improved agent system prompt — batching rules, visual-first ordering.

All v10 improvements retained:
  #1  Prompt expander
  #2  Planner (todo.md)
  #3  Auto-kill ports (sandbox_manager)
  #4  Narration fix
  #5  Per-file specs
  #6  Linter-in-the-loop (sandbox_manager)
  #7  Template starters
  #8  Silent success
  #9  History compression
  #10 Reviewer (sandbox_manager)

Speed fixes retained:
  - Expander → planner sequential (planner gets full expanded spec)
  - Token estimation cached by message count
  - Reviewer gated on turn count + request type (sandbox_manager)
  - File tree cached on session with 30s TTL (sandbox_manager)
  - Tar upload in 500KB chunks (sandbox_manager)
  - Log-polling instead of fixed sleep after npm run dev (sandbox_manager)
  - Parallel port health checks (sandbox_manager)
"""

from __future__ import annotations

import os
import re
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("LINEAGE_MODEL", "deepseek/deepseek-v4-flash")
SMART_MODEL = os.getenv("SMART_MODEL","deepseek/deepseek-v4-pro")   # hard turns
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "xiaomi/mimo-v2.5")
VISION_MODEL = os.getenv("VISION_MODEL", "xiaomi/mimo-v2.5")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions",
).strip()
SITE_URL = os.getenv("SITE_URL", "https://gorillabuilder.dev").strip()
SITE_NAME = os.getenv("SITE_NAME", "Gorilla Builder")

MAX_CONTEXT_TOKENS = 1_000_000
CHARS_PER_TOKEN = 4

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY must be set")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_external_log_callback = None


def set_log_callback(cb):
    global _external_log_callback
    _external_log_callback = cb


def log_agent(role: str, message: str, project_id: str = "") -> None:
    prefix = f"[{project_id[:8]}]" if project_id else "[AGENT]"
    ts = time.strftime("%H:%M:%S")
    c = {
        "agent": "\033[94m",
        "llm": "\033[90m",
        "system": "\033[97m",
        "debugger": "\033[91m",
    }.get(role.lower(), "\033[94m")
    print(
        f"\033[90m{ts}\033[0m {prefix} {c}{role.upper()}\033[0m: "
        f"{message[:300]}{'...' if len(message) > 300 else ''}"
    )
    if _external_log_callback and project_id and role.lower() != "llm":
        try:
            _external_log_callback(project_id, role.lower(), message)
        except Exception:
            pass


def _render_token_limit_message() -> str:
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;padding:40px 30px;'
        'background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(30,10,50,0.8));'
        'border:1px solid rgba(217,70,239,0.3);border-radius:20px;'
        'text-align:center;max-width:400px;margin:20px auto;'
        'box-shadow:0 20px 60px rgba(0,0,0,0.5);">'
        '<h2 style="color:#fff;font-size:24px;font-weight:700;margin:0 0 12px;">'
        "Token Limit Reached</h2>"
        '<p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 28px;'
        'max-width:280px;">Upgrade to Premium for unlimited access.</p>'
        '<a href="/pricing" style="background:linear-gradient(135deg,#d946ef,#a855f7);'
        'color:white;text-decoration:none;padding:14px 32px;border-radius:12px;'
        'font-size:14px;font-weight:600;">Upgrade to Premium</a></div>'
    )


# ---------------------------------------------------------------------------
# Legacy shims (kept for app.py compatibility)
# ---------------------------------------------------------------------------
_HISTORY: Dict[str, list] = {}
HISTORY_CAP = 100


def _norm_role(r: str) -> str:
    return "user" if (r or "").strip().lower() in ("user", "you") else "assistant"


def _append_history(project_id: str, role: str, content: str) -> None:
    if not project_id or not content:
        return
    _HISTORY.setdefault(project_id, []).append(
        {"role": _norm_role(role), "content": content.strip()}
    )
    if len(_HISTORY[project_id]) > HISTORY_CAP:
        _HISTORY[project_id] = _HISTORY[project_id][-HISTORY_CAP:]


def _get_history(project_id: str, max_items: int = 20) -> list:
    return list(_HISTORY.get(project_id, []))[-max_items:]


def clear_history(project_id: str) -> None:
    _HISTORY.pop(project_id, None)


# ---------------------------------------------------------------------------
# Token substitution (for images/blobs in file tree)
# ---------------------------------------------------------------------------
class TokenSubstitution:
    THRESHOLD = 500

    def __init__(self):
        self._vault: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}
        self._n = 0

    def _mk(self) -> str:
        self._n += 1
        return f"__BLOB_{self._n:04d}__"

    @staticmethod
    def _is_b64(s: str) -> bool:
        if len(s) < 100:
            return False
        sample = s[:200].strip()
        return (
            sum(1 for c in sample if c.isalnum() or c in "+/=") / len(sample)
        ) > 0.9 and "\n" not in sample[:100]

    def compress_tree(self, tree: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for path, content in tree.items():
            if content and len(content) > self.THRESHOLD:
                if (
                    path.endswith(".b64")
                    or self._is_b64(content)
                    or (path.endswith(".json") and len(content) > 5000)
                    or (path.endswith(".svg") and len(content) > 3000)
                ):
                    h = hashlib.md5(content[:200].encode()).hexdigest()
                    if h in self._reverse:
                        out[path] = self._reverse[h]
                    else:
                        pid = self._mk()
                        self._vault[pid] = content
                        self._reverse[h] = pid
                        out[path] = pid
                    continue
            out[path] = content
        return out

    def expand(self, text: str) -> str:
        for ph, original in self._vault.items():
            if ph in text:
                text = text.replace(ph, original)
        return text


# ---------------------------------------------------------------------------
# Context management (#9 — history compression)
# ---------------------------------------------------------------------------

# Cache: [message_count, estimated_tokens]
_token_estimate_cache: List[Any] = []


def _estimate_tokens(messages: list) -> int:
    """Cached token estimator — avoids re-scanning full history every LLM call."""
    global _token_estimate_cache
    msg_count = len(messages)
    if _token_estimate_cache and _token_estimate_cache[0] == msg_count:
        return _token_estimate_cache[1]

    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c) // CHARS_PER_TOKEN
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total += len(item.get("text", "")) // CHARS_PER_TOKEN
                    elif item.get("type") == "image_url":
                        total += 1000
    _token_estimate_cache = [msg_count, total]
    return total


def _compress_history(messages: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> list:
    """
    #9 — History compression.
    Keep system prompt + first user message + last 10 messages.
    Collapse earlier observations to one-line summaries.
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    first_user = messages[1] if len(messages) > 1 else None

    keep_full = 10
    recent = messages[-keep_full:]

    middle_start = 2 if first_user else 1
    middle_end = len(messages) - keep_full
    compressed_middle = []

    for m in messages[middle_start:middle_end]:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "")[:100] for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)
        if role == "user" and content.startswith("OBSERVATION:"):
            first_line = content.split("\n")[1] if "\n" in content else content
            compressed_middle.append(
                {"role": "user", "content": f"OBSERVATION: {first_line[:120]}..."}
            )
        elif role == "assistant":
            lines = content.split("```")[0].strip().split("\n")
            summary = lines[0][:120] if lines else "..."
            compressed_middle.append({"role": "assistant", "content": summary})
        else:
            compressed_middle.append({"role": role, "content": content[:200]})

    result = []
    if sys_msg:
        result.append(sys_msg)
    if first_user and first_user not in recent:
        result.append(first_user)
    result.extend(compressed_middle)
    result.extend(recent)

    if _estimate_tokens(result) > max_tokens:
        result = [sys_msg] if sys_msg else []
        if first_user:
            result.append(first_user)
        result.extend(recent)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  (v12 — multi-file turns + visual-first ordering)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = r"""You are Gor://a, an elite autonomous software engineer. You work inside an Ubuntu sandbox with full control: write files, run commands, start servers, debug, verify.

## How this works

Each turn you provide:
1. Your reasoning (speak naturally — never say "Discussion:" or "Working on it...")
2. A single fenced bash block with the command(s) to run

Example:

Time to scaffold the core pages. I'll write the landing and dashboard together since they share no imports yet.

```bash
cat > src/pages/Landing.tsx << 'EOF'
...full file content...
EOF

cat > src/pages/Dashboard.tsx << 'EOF'
...full file content...
EOF
```

I will show you the OBSERVATION. Then you respond with your next step.

When finished and both :8080 and :3000 return HTTP 200, write GORILLA_DONE followed by a user-facing summary.

## Environment

Ubuntu 22 / Node 20 / Python 3.11 / CWD: `/home/user/app`

Binaries: node, npm, npx, git, curl, jq, unzip, tar, find, grep, sed, awk, python3

Pre-installed packages: react, react-dom, react-router-dom, vite, @vitejs/plugin-react, typescript, tailwindcss, postcss, autoprefixer, clsx, tailwind-merge, class-variance-authority, @radix-ui/*, lucide-react, @supabase/supabase-js, express, cors, body-parser, dotenv, concurrently

Dev server: `npm run dev` → Vite on :8080 (frontend) + Express on :3000 (API)

Env vars in `.gorilla_env`: $GORILLA_API_KEY, $VITE_GORILLA_AUTH_ID, $VITE_SUPABASE_URL, $VITE_SUPABASE_ANON_KEY, $SUPABASE_MGMT_TOKEN, $SUPABASE_PROJECT_REF

Layout: src/ (React), src/components/ui/ (shadcn), src/utils/auth.ts (DO NOT TOUCH), routes/ (Express), public/generated/ (AI images)

## File batching rules  ← KEY FOR SPEED

You MUST batch file writes to minimise turns:

BATCH these together (up to 3 files per bash block):
  - New files that don't yet import each other
  - Pure UI components written at the same time (Navbar + Footer + Hero)
  - Multiple new page components before wiring
  - Utility/helper files alongside a component that uses them

NEVER batch these (always solo, always their own bash block):
  - Edits to EXISTING files (App.tsx, server.js, index.css, etc.)
  - Any command that starts the dev server
  - Any verification/curl command
  - npm install
  - Database migrations

ORDERING RULE — always build in this order:
  1. Read existing files (one cat block, can read multiple files at once)
  2. index.css + design tokens (so every subsequent component looks great immediately)
  3. Shared components (Navbar, Footer, UI primitives) — batch these
  4. Page components — batch independent pages together
  5. Backend routes — batch independent route files together
  6. Wire App.tsx (solo — it imports everything)
  7. Wire server.js (solo — it imports everything)
  8. Start server + verify

VISUAL-FIRST RULE: The very first files you create must produce a stunning visual result.
Write index.css with the full design system (colors, fonts, animations) before any component.
The user sees a live preview — make it look incredible from turn 3 onward.

## Rules

1. `npm install` before importing anything not in the pre-installed list above
2. Always read an existing file before editing it
3. Wire new components (import + render) and routes (import + app.use) — don't leave orphans
4. Frontend: `@/` alias. Backend: relative paths with `.js` extension
5. Never touch vite.config.ts, .env, src/utils/auth.ts directly
6. Start server: `cd /home/user/app && npm run dev > /tmp/dev.log 2>&1 &`
7. Verify: `sleep 3 && curl -so /dev/null -w '%{http_code}' http://localhost:8080 && curl -so /dev/null -w '%{http_code}' http://localhost:3000`
8. MUST verify both ports return 200 before GORILLA_DONE
9. On failure: read /tmp/dev.log, fix, restart, verify again
10. NEVER say GORILLA_DONE before turn 5
11. ALWAYS replace boilerplate index.tsx — go all out on the design
12. Check .gorilla/todo.md every 8 turns and tick completed tasks

## Auth gateway
```tsx
import { login, logout, onAuthStateChanged } from '@/utils/auth';
useEffect(() => onAuthStateChanged(setUser), []);
<button onClick={() => login('google')}>Sign in</button>
```

## AI proxy (backend only, uses $GORILLA_API_KEY)
Base URL: {GORILLA_PROXY}
- LLM chat:    POST {GORILLA_PROXY}/api/v1/chat/completions  (omit model field)
- Image gen:   POST {GORILLA_PROXY}/api/v1/images/generations (omit model field)
- STT:         POST {GORILLA_PROXY}/api/v1/audio/transcriptions
- BG removal:  POST {GORILLA_PROXY}/api/v1/images/remove-background
- TTS:         window.speechSynthesis (frontend only)

Image generation → save as base64 to public/generated/, reference directly in src.

Critical reminders:
- API keys live in app/.env — do not look elsewhere
- Never wrap App.tsx routes in a second BrowserRouter inside main.tsx
- React errors may be silent — always check /tmp/dev.log and the console error tunnel
"""

SUPABASE_ADDON = r"""
## Supabase (active)
Client:
```ts
import { createClient } from '@supabase/supabase-js';
const supabase = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);
```
Migrations:
```bash
cat > /tmp/migration.sql << 'SQL'
CREATE TABLE IF NOT EXISTS items (...);
ALTER TABLE items ENABLE ROW LEVEL SECURITY;
CREATE POLICY "own" ON items USING (auth.uid() = user_id);
SQL
curl -sS -X POST "https://api.supabase.com/v1/projects/$SUPABASE_PROJECT_REF/database/query" \
  -H "Authorization: Bearer $SUPABASE_MGMT_TOKEN" -H "Content-Type: application/json" \
  -d "$(cat /tmp/migration.sql | jq -Rs '{query: .}')"
```
"""

DEBUG_ADDON = r"""
## Debug mode — fix the error, nothing else
Read the error. Find the file. Make the smallest fix. Restart server. Verify ports. GORILLA_DONE.
Do NOT refactor, do NOT add features. Surgical fix only.
"""

# ---------------------------------------------------------------------------
# Supabase addons for expander + planner
# ---------------------------------------------------------------------------

EXPANDER_SUPABASE_ADDON = """
## Supabase is linked and available
Plan to use Supabase for data persistence when the app clearly benefits (user-saved content, shared data, multi-session state). Specify:
- Tables, columns, and relationships
- Which tables need Row Level Security
- Whether realtime subscriptions add value
Don't force Supabase on purely client-side or stateless apps."""

PLANNER_SUPABASE_ADDON = """
## Supabase is linked
Include migration steps when the app genuinely stores user or shared data:
```bash
cat > /tmp/migration.sql << 'SQL'
CREATE TABLE IF NOT EXISTS items (...);
ALTER TABLE items ENABLE ROW LEVEL SECURITY;
CREATE POLICY "own" ON items USING (auth.uid() = user_id);
SQL
curl -sS -X POST "https://api.supabase.com/v1/projects/$SUPABASE_PROJECT_REF/database/query" \
  -H "Authorization: Bearer $SUPABASE_MGMT_TOKEN" -H "Content-Type: application/json" \
  -d "$(cat /tmp/migration.sql | jq -Rs '{query: .}')"
```"""


# ═══════════════════════════════════════════════════════════════════════════
#  #1 — Prompt Expander
# ═══════════════════════════════════════════════════════════════════════════

EXPANDER_SYSTEM = """You are a product designer for Gorilla Builder — a platform that builds REAL working SaaS apps, not landing pages or brochures.

Platform capabilities available to the developer:
- **Auth gateway** — Google & GitHub login, zero setup, returns {id, email, name, avatar}
- **AI proxy** — LLM chat, image generation, speech-to-text, background removal
- **Express backend** — full API server, can store data, proxy AI, handle webhooks
- **Supabase** (if linked) — postgres DB with RLS, realtime subscriptions

The user will give you a short app idea. Expand it into a detailed product spec (200-350 words).

A GOOD spec describes a FUNCTIONAL APP:
- What does the user DO after landing? (create, browse, interact, generate, save, share)
- What data is stored and where?
- What makes the AI features genuinely useful, not just bolted on?
- Does it need auth? (if users save anything: yes)

Include:
- App name (creative, memorable)
- Color scheme (specific hex codes, dark mode preferred)
- Typography (a distinctive Google Font — never Inter, never system-ui)
- Pages (minimum 3): what the user SEES and DOES on each
- Backend: API routes needed, what data is stored
- AI integration: specific use of LLM/image gen/STT
- Auth: which providers and why

GOOD example — "AI recipe generator":
  → Landing (hero + demo), Generate page (enter ingredients → AI recipe + food image), Saved recipes dashboard (grid, filter by cuisine), auth via Google

BAD example — "a website about coffee":
  → Hero, About, Features, Contact form ← this is a brochure, not an app

FOR MINOR TASKS OR DEBUGGING: just restate the task clearly in 1-2 sentences. Do not expand.

If an image is provided, treat it as a UI mockup — extract layout, palette, components, and flows. Incorporate them into the spec.

Output ONLY the spec. No preamble, no sign-off."""


async def expand_prompt(
    short_prompt: str,
    has_supabase: bool = False,
    image_b64: Optional[str] = None,
) -> str:
    if len(short_prompt) > 300:
        return short_prompt

    system = EXPANDER_SYSTEM
    if has_supabase:
        system += "\n" + EXPANDER_SUPABASE_ADDON

    user_content: Any
    if image_b64:
        img_url = (
            image_b64 if image_b64.startswith("data:")
            else f"data:image/jpeg;base64,{image_b64}"
        )
        user_content = [
            {"type": "text", "text": short_prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    else:
        user_content = short_prompt

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        model = VISION_MODEL if image_b64 else PLANNER_MODEL
        raw, _ = await _call_llm(messages, model=model, temperature=0.8)
        expanded = raw.strip()
        if len(expanded) > len(short_prompt) * 2:
            log_agent("agent", f"Expanded prompt: {expanded[:150]}...")
            return expanded
        return short_prompt
    except Exception as e:
        log_agent("agent", f"Expander failed ({e}), using original prompt")
        return short_prompt


# ═══════════════════════════════════════════════════════════════════════════
#  #2 + #5 + #12 — Planner (multi-file aware, visual-first ordering)
# ═══════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are a project planner for Gorilla Builder — a platform that builds full-stack SaaS apps with React + Vite + Express.

## Platform capabilities

**Auth gateway** (only if the app needs login):
```tsx
import { login, logout, onAuthStateChanged } from '@/utils/auth';
```

**AI proxy** (backend, $GORILLA_API_KEY):
- LLM: POST /api/v1/chat/completions
- Images: POST /api/v1/images/generations → save to public/generated/
- STT: POST /api/v1/audio/transcriptions
- BG removal: POST /api/v1/images/remove-background

**Express backend**: routes/ folder, mounted in server.js

## Your job

Output a markdown checklist where each item = ONE agent action.

## CRITICAL: Multi-file batching

The agent can write multiple NEW independent files in a single bash block.
Your plan MUST group independent files together into single steps to minimise total turns.

BATCH these (write together in one step):
- Multiple new page components written at the same time
- Multiple new UI components with no cross-imports
- Multiple new backend route files
- A component file + its types file

NEVER batch these (each gets its own step):
- Any edit to an existing file
- Server start command
- Verification/curl commands
- npm install
- Database migrations

## Ordering (always follow this)

1. Read existing files (cat App.tsx server.js index.css in ONE block)
2. Update index.css with full design system (solo — existing file edit)
3. Generate AI images if needed (curl, max 2-3)
4. Create shared components in ONE batch (Navbar + Footer + any UI primitives)
5. Create page components in batches (Landing + Hero in one, Dashboard + Settings in another)
6. Create backend routes in a batch (independent route files together)
7. Update App.tsx to wire all pages (solo — existing file)
8. Update server.js to mount routes (solo — existing file)
9. npm install (only if truly needed beyond pre-installed packages)
10. Start dev server + verify both ports

## Format

```
# Task: <short title>

- [ ] Read existing structure — cat src/App.tsx server.js src/index.css
- [ ] Update src/index.css — full dark design system, CSS vars, font imports, animations
- [ ] Generate hero image — AI proxy curl with descriptive prompt
- [ ] Create src/components/Navbar.tsx + src/components/Footer.tsx — dark nav with brand + auth, sticky footer
- [ ] Create src/pages/Landing.tsx + src/pages/About.tsx — hero section with CTA, about page
- [ ] Create src/pages/Dashboard.tsx + src/pages/Settings.tsx — main app grid, user settings
- [ ] Create routes/api.js + routes/auth.js — POST /api/generate, GET /api/items
- [ ] Update src/App.tsx — wire all routes, auth guard on dashboard
- [ ] Update server.js — mount routes/api.js and routes/auth.js
- [ ] Start dev server and verify both ports return 200
```

## Rules

- First item: read existing files (always)
- Last item: start server + verify
- Group independent NEW files together — this is the most important rule
- 6-14 items total (batching means fewer items than before)
- For minor tasks or debug: max 3-4 items, no batching needed
- Don't overspecify internals — the agent knows React
- Include AI image generation only when visuals genuinely add value
- Output ONLY the checklist, no preamble"""


async def generate_plan(
    expanded_prompt: str,
    file_tree_summary: str,
    has_supabase: bool = False,
    image_b64: Optional[str] = None,
) -> Optional[str]:
    system = PLANNER_SYSTEM
    if has_supabase:
        system += "\n" + PLANNER_SUPABASE_ADDON

    user_text = f"Existing files:\n{file_tree_summary}\n\nSpec:\n{expanded_prompt}"

    user_content: Any
    if image_b64:
        img_url = (
            image_b64 if image_b64.startswith("data:")
            else f"data:image/jpeg;base64,{image_b64}"
        )
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        model = VISION_MODEL if image_b64 else PLANNER_MODEL
        raw, _ = await _call_llm(messages, model=model, temperature=0.4)
        plan = raw.strip()
        if "- [ ]" in plan:
            log_agent("agent", f"Plan generated: {plan[:200]}...")
            return plan
        return None
    except Exception as e:
        log_agent("agent", f"Planner failed ({e}), agent will self-plan")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  #10 — Reviewer
# ═══════════════════════════════════════════════════════════════════════════

REVIEWER_SYSTEM = """You are a code reviewer. A developer just finished building a web app. Review the file listing and recent build output for obvious mistakes only.

Check for:
- Components created but not imported/rendered anywhere (orphaned files)
- Missing npm installs (import of a package not in node_modules)
- Express routes created but not mounted in server.js
- CSS files imported but not linked
- TypeScript errors visible in logs

Respond with ONLY one of:
- "LGTM" if everything looks correct
- A numbered list of up to 3 specific, actionable fixes (no preamble)"""


async def review_output(file_tree_summary: str, last_output: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM},
        {
            "role": "user",
            "content": f"Files:\n{file_tree_summary}\n\nRecent output:\n{last_output[:3000]}",
        },
    ]
    try:
        raw, _ = await _call_llm(messages, model=SMART_MODEL, temperature=0.2)
        review = raw.strip()
        if "LGTM" in review.upper():
            log_agent("agent", "Reviewer: LGTM")
            return None
        log_agent("agent", f"Reviewer found issues: {review[:200]}")
        return review
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Response parser — ThoughtAction format
#  Collects ALL bash blocks in a response, not just the first
# ═══════════════════════════════════════════════════════════════════════════

def _parse_response(raw: str) -> Dict[str, Any]:
    """
    Parse discussion + one or more ```bash``` blocks + optional GORILLA_DONE.

    v11 change: returns all bash blocks concatenated with a newline separator
    so the sandbox executor runs them as a single shell script. This supports
    multi-file writes where the agent emits multiple heredoc blocks.
    """
    thought = ""
    bash_blocks: List[str] = []
    done = False
    message = ""

    if not raw:
        return {"thought": "", "bash": "", "done": False, "message": ""}

    if "GORILLA_DONE" in raw:
        done = True
        parts = raw.split("GORILLA_DONE", 1)
        body = parts[0]
        message = parts[1].strip() if len(parts) > 1 else ""
        blocks = re.findall(r"```(?:bash|sh|shell)?\n(.*?)```", body, re.DOTALL)
        bash_blocks = [b.strip() for b in blocks if b.strip()]
        first_block_pos = body.find("```")
        thought = body[:first_block_pos].strip() if first_block_pos > 0 else body.strip()
        if not message and thought:
            message = thought.split("\n")[0][:300]
        return {
            "thought": thought,
            "bash": "\n\n".join(bash_blocks),
            "done": done,
            "message": message or "Done.",
        }

    blocks = re.findall(r"```(?:bash|sh|shell)?\n(.*?)```", raw, re.DOTALL)
    bash_blocks = [b.strip() for b in blocks if b.strip()]

    first_block_pos = raw.find("```")
    thought = raw[:first_block_pos].strip() if first_block_pos > 0 else raw.strip()

    if not bash_blocks:
        thought = raw.strip()

    if thought:
        sentences = re.split(r"(?<=[.!?])\s+", thought)
        message = " ".join(sentences[:3])[:300]

    return {
        "thought": thought,
        "bash": "\n\n".join(bash_blocks),
        "done": done,
        "message": message or "",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Shell safety
# ═══════════════════════════════════════════════════════════════════════════

_DANGEROUS = [
    r"\brm\s+-rf\s+/($|\s)",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r">\s*/dev/(sda|nvme|hda)",
    r"\bmkfs\b",
    r":\(\)\s*{\s*:\|:",
    r"\bdd\s+if=.*\s+of=/dev/",
]


def _is_safe(cmd: str) -> bool:
    return not any(re.search(p, cmd, re.IGNORECASE) for p in _DANGEROUS)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM call
# ═══════════════════════════════════════════════════════════════════════════

async def _call_llm(
    messages: list, model: str = MODEL, temperature: float = 0.6
) -> Tuple[str, int]:
    messages = _compress_history(messages)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 16000,
        "provider": {
            "order": ["deepseek", "xiaomi", "xai"],
            "allow_fallbacks": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    u = data.get("usage", {})
    p = u.get("prompt_tokens", 0)
    c = u.get("completion_tokens", 0)
    is_frontier = any(x in model for x in ["claude", "gpt-4", "gemini"])
    is_mimo = "mimo" in model
    if is_frontier:
        weight = p * 0.6 + c * 2.4
    elif is_mimo:
        weight = p * 0.5 + c * 2
    else:
        weight = p * 0.2 + c * 0.3
    return content, int(weight)


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Skills helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_skills_block(agent_skills: Optional[Dict[str, Any]]) -> str:
    if not agent_skills:
        return ""
    enabled = [k for k, v in agent_skills.items() if v]
    if not enabled:
        return ""
    lines = "\n".join(f"- {skill}" for skill in enabled)
    return (
        f"\n\n## User-Enabled Agent Skills\n"
        f"Respect these throughout the session:\n{lines}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  #14 — Smart model routing
# ═══════════════════════════════════════════════════════════════════════════

_HARD_OBSERVATION_SIGNALS = frozenset([
    "error ts",
    "syntaxerror",
    "cannot find",
    "could not be resolved",
    "is not defined",
    "is not a function",
    "failed to compile",
    "vite compile errors",
    "lint errors",
    "browser console errors",
    "exit code: 1",
    "exit code: -1",
    "enoent",
    "module not found",
    "cannot read propert",
    "typeerror",
    "referenceerror",
])

_HARD_DOMAIN_SIGNALS = frozenset([
    "auth", "supabase", "migration", "rls", "policy",
    "realtime", "subscription", "webhook", "oauth",
    "race condition", "async", "promise", "cors",
    "jwt", "token", "session", "cookie",
    "database", "schema", "foreign key", "join",
])


def _pick_model(
    turn: int,
    previous_output: Optional[str],
    user_request: str,
    is_debug: bool,
) -> str:
    obs = (previous_output or "").lower()
    req = (user_request or "").lower()

    if turn == 0:
        return SMART_MODEL
    if any(sig in obs for sig in _HARD_OBSERVATION_SIGNALS):
        return SMART_MODEL
    if turn > 12 and any(sig in obs for sig in _HARD_OBSERVATION_SIGNALS):
        return SMART_MODEL
    if turn <= 4 and any(sig in req for sig in _HARD_DOMAIN_SIGNALS):
        return SMART_MODEL
    if is_debug and previous_output:
        return SMART_MODEL

    return MODEL


# ═══════════════════════════════════════════════════════════════════════════
#  LineageAgent  v12
# ═══════════════════════════════════════════════════════════════════════════

class LineageAgent:
    """
    mini-SWE-agent with multi-file turn support and smart model routing.

    v12 changes:
      - _pick_model() routes each turn to Mimo v2.5 (hard) or Deepseek V4
        Flash (mechanical). ~80% of turns stay cheap and fast. Hard turns
        (architecture, error recovery, debug spirals, auth/DB) get Mimo.

    v11 changes:
      - System prompt teaches multi-file batching with clear rules
      - Planner groups independent files into single steps
      - Parser collects ALL bash blocks in a response (not just the first)
      - Visual-first ordering: index.css + design tokens written before components

    v10 speed fixes retained:
      - Expander → planner sequential (planner gets full expanded spec)
      - Token estimation cached by message count
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.total_tokens = 0
        self.token_sub = TokenSubstitution()
        self.messages: List[Dict[str, Any]] = []
        self._system_prompt_set = False
        self._plan_injected = False
        self._prompt_expanded = False

    def _ensure_system_prompt(
        self,
        gorilla_proxy_url: str,
        has_supabase: bool,
        is_debug: bool,
        agent_skills: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._system_prompt_set:
            return
        prompt = SYSTEM_PROMPT.replace(
            "{GORILLA_PROXY}",
            gorilla_proxy_url or "https://your-proxy.ngrok-free.dev",
        )
        if has_supabase:
            prompt += "\n" + SUPABASE_ADDON
        if is_debug:
            prompt += "\n" + DEBUG_ADDON

        skills_block = _build_skills_block(agent_skills)
        if skills_block:
            prompt += skills_block
            log_agent(
                "agent",
                f"Skills injected: {[k for k, v in (agent_skills or {}).items() if v]}",
                self.project_id,
            )

        self.messages = [{"role": "system", "content": prompt}]
        self._system_prompt_set = True

    def _extract_prompt_image(self, file_tree: Dict[str, str]) -> Optional[str]:
        raw = file_tree.get(".gorilla/prompt_image.b64") or file_tree.get("prompt_image.b64")
        if not raw:
            return None
        stripped = raw.strip()
        if stripped.startswith("data:"):
            return stripped
        return f"data:image/jpeg;base64,{stripped}"

    async def run(
        self,
        user_request: str,
        file_tree: Dict[str, str],
        chat_history: Optional[list] = None,
        gorilla_proxy_url: str = "",
        has_supabase: bool = False,
        is_debug: bool = False,
        error_context: str = "",
        image_b64: Optional[str] = None,
        previous_command_output: Optional[str] = None,
        agent_skills: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import asyncio

        self._ensure_system_prompt(gorilla_proxy_url, has_supabase, is_debug, agent_skills)

        compressed = self.token_sub.compress_tree(file_tree)
        clean_paths = sorted(p for p in compressed if not p.endswith(".b64"))
        tree_str = "\n".join(f"  {p}" for p in clean_paths)

        pkg_snippet = ""
        if "package.json" in compressed:
            pkg = compressed["package.json"]
            if len(pkg) < 3000:
                pkg_snippet = f"\n\npackage.json:\n{pkg}"

        prompt_image_b64: Optional[str] = None
        if not previous_command_output:
            prompt_image_b64 = self._extract_prompt_image(file_tree)
            if prompt_image_b64:
                log_agent("agent", "prompt_image.b64 found — threading to all stages", self.project_id)

        if previous_command_output:
            user_content: Any = f"OBSERVATION:\n{previous_command_output[:8000]}"
            self.messages.append({"role": "user", "content": user_content})
        else:
            effective_request = user_request
            plan_text = ""

            # Sequential: expand first, then feed expanded prompt to planner.
            # Planner needs the full expanded spec to produce good batching groups.
            if not is_debug and user_request:
                needs_expand = not self._prompt_expanded
                needs_plan = not self._plan_injected and bool(file_tree)

                if needs_expand:
                    effective_request = await expand_prompt(
                        user_request,
                        has_supabase=has_supabase,
                        image_b64=prompt_image_b64,
                    )
                    self._prompt_expanded = True

                if needs_plan:
                    plan = await generate_plan(
                        effective_request,  # expanded spec, not raw prompt
                        tree_str,
                        has_supabase=has_supabase,
                        image_b64=prompt_image_b64,
                    )
                    if plan:
                        plan_text = (
                            "\n\nHere is your plan — follow it step by step. "
                            "Respect the batching groups (files listed together on one line "
                            "must be written in a single bash block):\n" + plan
                        )
                        self._plan_injected = True

            parts = [f"Project files in /home/user/app:\n{tree_str}{pkg_snippet}"]

            if is_debug and error_context:
                parts.append(f"\nError to fix:\n{error_context}")
            elif effective_request:
                parts.append(f"\nTask:\n{effective_request}{plan_text}")

            if chat_history:
                recent = chat_history[-6:]
                history_text = "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content','')[:200]}"
                    for m in recent if m.get("content")
                )
                if history_text:
                    parts.append(f"\nRecent conversation:\n{history_text}")

            user_text = "\n".join(parts)
            first_turn_image = image_b64 or prompt_image_b64

            if first_turn_image:
                img_url = (
                    first_turn_image if first_turn_image.startswith("data:")
                    else f"data:image/jpeg;base64,{first_turn_image}"
                )
                self.messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                })
            else:
                self.messages.append({"role": "user", "content": user_text})

        first_turn_has_image = bool(
            (image_b64 or prompt_image_b64) and not previous_command_output
        )
        if first_turn_has_image:
            model = VISION_MODEL
        else:
            turn_index = max(0, len(self.messages) // 2 - 1)
            model = _pick_model(
                turn=turn_index,
                previous_output=previous_command_output,
                user_request=user_request,
                is_debug=is_debug,
            )
        log_agent("agent", f"v12 ({model.split('/')[-1]}, turn={len(self.messages) // 2})", self.project_id)

        try:
            raw, tokens = await _call_llm(self.messages, model=model, temperature=0.6)
            self.total_tokens += tokens
        except Exception as e:
            log_agent("agent", f"LLM error: {e}", self.project_id)
            return {
                "message": f"AI error: {str(e)[:150]}",
                "commands": [],
                "done": True,
                "tokens": 0,
            }

        self.messages.append({"role": "assistant", "content": raw})
        parsed = _parse_response(raw)

        if parsed["thought"]:
            log_agent("agent", f"THOUGHT: {parsed['thought'][:300]}", self.project_id)

        commands: List[str] = []
        if parsed["bash"]:
            bash_content = self.token_sub.expand(parsed["bash"])
            if _is_safe(bash_content):
                commands = [bash_content]
            else:
                log_agent("agent", "Blocked dangerous command", self.project_id)

        done = parsed["done"]
        log_agent(
            "agent",
            f"{'DONE' if done else f'{len(commands)} cmd block'}, tok={self.total_tokens}",
            self.project_id,
        )

        return {
            "message": parsed["message"],
            "commands": commands,
            "done": done,
            "tokens": self.total_tokens,
        }


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s

    def remember(self, project_id: str, role: str, text: str) -> None:
        _append_history(project_id, role, text)


__all__ = [
    "LineageAgent",
    "Agent",
    "set_log_callback",
    "log_agent",
    "_render_token_limit_message",
    "_append_history",
    "_get_history",
    "clear_history",
    "TokenSubstitution",
    "expand_prompt",
    "generate_plan",
    "review_output",
]