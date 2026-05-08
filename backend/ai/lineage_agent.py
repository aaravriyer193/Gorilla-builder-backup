"""
Lineage Agent v17 — MiMo-native (Qwen3-Coder XML), batch-limited, naturally agentic
====================================================================================

Why v17 (vs v16's MiniMax XML):
  v16 spoke MiniMax-M2.7's <minimax:tool_call> dialect. That works for
  MiniMax. But Xiaomi's MiMo family (MiMo-V2-Flash, MiMo-V2.5-Pro, MiMo-V2-Pro)
  was trained on a DIFFERENT XML format inherited from Qwen3-Coder, not on
  MiniMax's namespaced tags. Forcing MiMo through MiniMax's wrapper (or
  through OpenAI's structured `tools` API) hurts performance — the model
  emits malformed output and reliability collapses on long-horizon agentic
  tasks.

  Sources verified:
    • docs.vllm.ai/projects/recipes/en/latest/MiMo/MiMo-V2-Flash.html
        → official vLLM serve command uses --tool-call-parser qwen3_xml
        → official vLLM serve command uses --reasoning-parser qwen3
    • huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py
        → defines the canonical XML grammar: <tool_call><function=NAME>
          <parameter=KEY>VALUE</parameter></function></tool_call>
    • huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/chat_template.jinja
        → tools injected into system prompt as <tools><function><name>…</name>
          <description>…</description><parameters>…</parameters></function></tools>
    • github.com/QwenLM/Qwen3-Coder/issues/475
        → the exact instruction wording that fixed unreliable tool calls:
          "If you choose to call a function ONLY reply in the following
           format with NO suffix"
    • platform.xiaomimimo.com/docs/api/chat/openai-api
        → multi-turn requires preserving reasoning_content between turns
          to avoid performance degradation

What changed mechanically vs v16:
  ✱ XML grammar: <tool_call><function=NAME>...</function></tool_call>
    (Qwen3-Coder format, NOT <minimax:tool_call>)
  ✱ Tool defs in system prompt: <tools><function><name>...</name>...
    <parameters><parameter>...</parameter></parameters></function></tools>
    (matches the official Qwen3-Coder Jinja template format)
  ✱ Includes the canonical "ONLY reply in this format with NO suffix"
    instruction proven to fix unreliable tool calls
  ✱ Reasoning preservation: passes reasoning blocks between turns and
    feeds reasoning_content back as part of the assistant message
  ✱ ALL models on the routing path are MiMo-family — MODEL, SMART_MODEL,
    PLANNER_MODEL, VISION_MODEL all use mimo-v2 variants. One tool dialect,
    one consistent voice.
  ✱ XML repair extended for Qwen3-Coder common defects (mostly: missing
    closing tags, parameter content with stray < or > characters).

What's preserved from v14/v15/v16 (ALL features intact):
  #1  Prompt expander                      #7  Template starters
  #2  Planner (todo.md)                    #8  Silent success
  #3  Auto-kill ports                      #9  History compression
  #4  Narration fix                        #10 Reviewer
  #5  Per-file specs                       #11 Multi-file turns (batched)
  #6  Linter-in-the-loop                   #12 Smart model routing
  + Supabase / Debug / Skills addons
  + Observation noise filter
  + TokenSubstitution for blob compression
  + BATCH_LIMIT = 2 (the v16 anti-dump rule)
  + Sandbox _parsed contract preserved (sandbox_manager doesn't change)
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Config — full MiMo family. Same dialect on every routing branch.
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# All models in the MiMo family so the Qwen3-Coder XML format stays
# consistent across the entire routing graph. Mixing in MiniMax or Qwen
# bare would mean two different tool dialects, which is exactly the
# failure mode v16 was trying to fix in the OTHER direction.
MODEL         = os.getenv("LINEAGE_MODEL",   "xiaomi/mimo-v2.5")
SMART_MODEL   = os.getenv("SMART_MODEL",     "xiaomi/mimo-v2.5-pro")     # was MiniMax M2.7 in v16
PLANNER_MODEL = os.getenv("PLANNER_MODEL",   "xiaomi/mimo-v2.5")
VISION_MODEL  = os.getenv("VISION_MODEL",    "xiaomi/mimo-v2.5")

OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions",
).strip()
SITE_URL  = os.getenv("SITE_URL",  "https://gorillabuilder.dev").strip()
SITE_NAME = os.getenv("SITE_NAME", "Gorilla Builder")

MAX_CONTEXT_TOKENS = 230_000
CHARS_PER_TOKEN    = 4

# Hard cap on writes per turn — preserved from v16. Single most important
# behavioral lever in the whole agent. MiMo is a strong long-horizon model
# but it still benefits enormously from observing intermediate results
# rather than dumping a full codebase in one shot.
BATCH_LIMIT = 2

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
        "agent":    "\033[94m",
        "llm":      "\033[90m",
        "system":   "\033[97m",
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
# Token substitution
# ---------------------------------------------------------------------------
class TokenSubstitution:
    THRESHOLD = 500

    def __init__(self):
        self._vault:   Dict[str, str] = {}
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
                    or (path.endswith(".svg")  and len(content) > 3000)
                ):
                    h = hashlib.md5(content[:200].encode()).hexdigest()
                    if h in self._reverse:
                        out[path] = self._reverse[h]
                    else:
                        pid = self._mk()
                        self._vault[pid]   = content
                        self._reverse[h]   = pid
                        out[path]          = pid
                    continue
            out[path] = content
        return out

    def expand(self, text: str) -> str:
        for ph, original in self._vault.items():
            if ph in text:
                text = text.replace(ph, original)
        return text


# ═══════════════════════════════════════════════════════════════════════════
# Tool definitions — these are rendered into the system prompt as a <tools>
# XML block in the EXACT format from Qwen3-Coder's official Jinja chat
# template. Do NOT pass these via OpenRouter's `tools` API parameter — that
# triggers a different code path inside MiMo's chat_template that treats
# them as Hermes-style JSON, which the model is not trained to emit
# reliably (verified against Qwen3-Coder issue #475 and the vLLM recipe).
# ═══════════════════════════════════════════════════════════════════════════

AGENT_TOOL_DEFS = [
    {
        "name": "write_file",
        "description": (
            "Write the full content of a single file to /home/user/app. "
            "Overwrites the file if it exists. Call this twice in the SAME tool_call "
            "block to write two files in parallel (max 2 per turn — see batching rules)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from /home/user/app, e.g. src/components/Navbar.tsx",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content. No line numbers, no diff markers — just the raw file.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional one-line note about what this file does.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a bash command in the sandbox at /home/user/app. "
            "Use AFTER any write_file calls in the same turn (or alone for read-only "
            "operations like cat, ls, curl). Prefer one &&-chained command over multiple "
            "separate calls. NEVER use heredocs to write files — use write_file for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional one-line description.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "mark_done",
        "description": (
            "Signal the task is complete. Only call this AFTER both ports have been "
            "verified to return 200 and the build genuinely matches the spec. "
            "Provide a short, user-facing summary of what was built or fixed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "1-3 sentences of what shipped, written for the user.",
                },
            },
            "required": ["summary"],
        },
    },
]


def _format_tools_for_prompt() -> str:
    """
    Render AGENT_TOOL_DEFS in Qwen3-Coder's exact training format.

    The structure is:
        # Tools
        You have access to the following functions:
        <tools>
        <function>
          <name>...</name>
          <description>...</description>
          <parameters>
            <parameter>
              <name>...</name>
              <type>...</type>
              <description>...</description>
              <required>true|false</required>
            </parameter>
            ...
          </parameters>
        </function>
        ...
        </tools>

    Verified against:
      huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/chat_template.jinja
      gist.github.com/mostlygeek/6fe263bad8026dca73cb6f5470dfdb0d
    """
    parts = [
        "# Tools",
        "",
        "You have access to the following functions:",
        "",
        "<tools>",
    ]
    for tool in AGENT_TOOL_DEFS:
        params      = tool.get("parameters", {})
        properties  = params.get("properties", {}) or {}
        required    = set(params.get("required", []) or [])

        parts.append("<function>")
        parts.append(f"<name>{tool['name']}</name>")
        parts.append(f"<description>{tool['description']}</description>")
        parts.append("<parameters>")

        for pname, pinfo in properties.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            parts.append("<parameter>")
            parts.append(f"<name>{pname}</name>")
            parts.append(f"<type>{ptype}</type>")
            parts.append(f"<description>{pdesc}</description>")
            parts.append(f"<required>{'true' if pname in required else 'false'}</required>")
            parts.append("</parameter>")

        parts.append("</parameters>")
        parts.append("</function>")

    parts.append("</tools>")

    # The exact instruction wording from QwenLM/Qwen3-Coder issue #475 that
    # fixed the missing-tag bug. This is verbatim from the upstream patch
    # that proved most reliable across community tests. We adapt it slightly
    # to mention the BATCH_LIMIT.
    parts.extend([
        "",
        "If you choose to call a function ONLY reply in the following format with NO suffix:",
        "",
        "<tool_call>",
        "<function=example_function_name>",
        "<parameter=example_parameter_1>",
        "value_1",
        "</parameter>",
        "<parameter=example_parameter_2>",
        "This is the value for the second parameter that can span multiple lines",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "",
        "Reminder:",
        "- Function calls MUST follow the specified format above",
        "- Required parameters MUST be specified",
        "- Only call functions that are listed above",
        f"- At MOST {BATCH_LIMIT} <function=...> blocks per <tool_call> wrapper (the batch limit)",
        "- You may provide brief reasoning in plain text BEFORE the <tool_call>, but NOT after",
        "- After </tool_call>, stop generating — the results come back next turn",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context management — history compression (XML-aware)
# ---------------------------------------------------------------------------
_token_estimate_cache: List[Any] = []


def _estimate_tokens(messages: list) -> int:
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
        # Reasoning tokens (MiMo-specific) — we keep them in context so the
        # model maintains its train of thought across turns. Each reasoning
        # block is roughly bounded so this doesn't explode.
        rd = m.get("reasoning_content", "")
        if rd:
            total += len(str(rd)) // CHARS_PER_TOKEN
    _token_estimate_cache = [msg_count, total]
    return total


def _compress_history(messages: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> list:
    """
    Keep system + first user + last 10 messages. Collapse middle assistant
    messages to "[wrote: a.tsx, b.tsx]" summaries, middle observations to
    first line. Strip reasoning_content from compressed messages.
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    sys_msg    = messages[0] if messages and messages[0].get("role") == "system" else None
    first_user = None
    for m in messages[1:]:
        if m.get("role") == "user":
            first_user = m
            break

    keep_full    = 10
    recent       = messages[-keep_full:]
    middle_start = 2 if first_user else 1
    middle_end   = len(messages) - keep_full
    compressed   = []

    for m in messages[middle_start:middle_end]:
        role    = m.get("role", "")
        content = str(m.get("content", ""))

        if role == "assistant":
            # Pull out write_file paths and run_bash commands for a one-liner
            paths = re.findall(
                r"<function=write_file>[\s\S]*?<parameter=path>\s*([^\s<][^<]*?)\s*</parameter>",
                content,
            )
            cmds = re.findall(
                r"<function=run_bash>[\s\S]*?<parameter=command>\s*([^<]+?)\s*</parameter>",
                content,
            )
            summary_parts = []
            if paths:
                summary_parts.append(f"wrote: {', '.join(p.strip() for p in paths[:3])}")
            if cmds:
                first_cmd = cmds[0].strip().split("\n")[0][:80]
                summary_parts.append(f"ran: {first_cmd}")
            if "<function=mark_done>" in content:
                summary_parts.append("DONE")
            if summary_parts:
                compressed.append({"role": "assistant", "content": "[" + " | ".join(summary_parts) + "]"})
            else:
                compressed.append({"role": "assistant", "content": content[:120]})

        elif role == "user":
            if content.startswith("OBSERVATION:") or content.startswith("[tool_result]"):
                first_line = content.split("\n", 2)[1] if "\n" in content else content
                compressed.append({"role": "user", "content": f"OBSERVATION: {first_line[:120]}..."})
            else:
                compressed.append({"role": role, "content": content[:200]})
        else:
            compressed.append({"role": role, "content": content[:200]})

    result = []
    if sys_msg:    result.append(sys_msg)
    if first_user and first_user not in recent:
        result.append(first_user)
    result.extend(compressed)
    result.extend(recent)

    if _estimate_tokens(result) > max_tokens:
        result = [sys_msg] if sys_msg else []
        if first_user:
            result.append(first_user)
        result.extend(recent)

    return result


# ---------------------------------------------------------------------------
# Observation noise filter
# ---------------------------------------------------------------------------
_VITE_NOISE_RE = re.compile(
    r"""(
        ^\s*(WARNING|warn)\b(?!.*\berror\b)     |
        node_modules/.*warning                  |
        Browserslist:.*outdated                 |
        @vitejs/plugin-react.*preamble          |
        vite\s+v\d                              |
        VITE\s+v\d                              |
        ^\s*→\s+Local:                          |
        ^\s*➜\s+Local:                          |
        ^\s*ready\s+in\s+\d+ms                  |
        ^\s*hmr\s                               |
        \[vite\]\s+(page reload|hot updated|connected)  |
        ^\s*(\d+\s+)?modules?\s+transformed     |
        eslint.*warning                         |
        ^\s*\d+\s+warning
    )""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

_FATAL_SIGNALS = frozenset([
    "error ts", "syntaxerror", "cannot find module", "could not be resolved",
    "failed to compile", "exit code: 1", "exit code: -1", "enoent",
    "typeerror", "referenceerror", "error:", "[error]",
])


def _filter_observation(raw: str) -> str:
    if not raw:
        return raw
    lines       = raw.splitlines()
    kept        = []
    noise_count = 0
    for line in lines:
        low = line.lower()
        if any(sig in low for sig in _FATAL_SIGNALS):
            kept.append(line)
            continue
        if _VITE_NOISE_RE.search(line):
            noise_count += 1
            continue
        kept.append(line)
    result = "\n".join(kept).strip()
    if noise_count > 5 and len(result) < 100:
        result = (result + f"\n[{noise_count} Vite/lint warnings suppressed — no errors]").strip()
    return result or raw


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  v17
#
#  Inspired by:
#  • Cline's explore-implement-verify loop (cline.ghost.io/system-prompt-advanced)
#  • Aider's full-file-write-then-targeted-edit philosophy
#  • OpenHands's "thorough, methodical, quality over speed" framing
#  • Qwen3-Coder's "ONLY reply in this format with NO suffix" pattern
#  • Original v14/v15/v16 Gorilla domain knowledge (Vite + Express + AI proxy)
#
#  The Qwen3-Coder tools block is appended at runtime by
#  _format_tools_for_prompt() so MiMo sees its tools in the exact format
#  it was trained on.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_BODY = r"""You are Gorilla, a senior full-stack engineer. You share one workspace with the user: an Ubuntu sandbox running React + Vite on port 8080 and Express on port 3000. Your job is to build real, working SaaS apps — not mockups — and stay with the work until it's genuinely done.

# How you work

You work in small, observable steps. Each turn you do ONE of three things:

1. **Explore** — read files, list directories, check logs. Use this when you need information before deciding what to build or fix.
2. **Implement (small batch)** — write up to 2 files, then optionally run a quick command to install/restart. After this you stop and look at what happened.
3. **Verify or finish** — run the verification command, read the output, and either fix or call mark_done.

**The hard rule: at most 2 write_file calls per turn.** This is non-negotiable. If you're building something that needs 10 files, that's 5 turns — and that's good, because between each turn you get to see the dev server's hot-reload output and catch problems early. Bad codebases come from agents that try to write everything at once and ship a tangled mess. You write a focused pair of files, observe, then continue with the next pair informed by what you just saw.

This is closer to how a human engineer works: you don't write the whole app then hit save once. You write Navbar.tsx + index.css together, alt-tab to the browser, see it render, then move on to Footer.tsx + Hero.tsx. The two-file batch is your unit of progress.

**Order of operations within a build:**

Turn 1: Explore — read existing src/App.tsx, server.js, src/index.css to understand the starting point.
Turn 2: Foundation — write src/index.css (full design system) + maybe src/App.tsx skeleton.
Turn 3-N: Components — Navbar + Footer in one turn, then two pages in the next, etc.
Turn N+1: Backend — one or two route files at a time.
Final turn: run dev server, curl both ports, mark_done.

# Environment

- Ubuntu 22 / Node 20 / Python 3.11 — working directory `/home/user/app`
- Dev server: `npm run dev` starts Vite on :8080 and Express on :3000 concurrently.
- Vite has HMR — frontend file writes hot-reload; no restart needed for `src/` changes.
- Pre-installed: react, react-dom, react-router-dom, vite, @vitejs/plugin-react, typescript, tailwindcss, postcss, autoprefixer, clsx, tailwind-merge, class-variance-authority, @radix-ui/*, lucide-react, express, cors, body-parser, dotenv, concurrently
- Source layout: `src/` (React), `src/components/ui/` (shadcn), `routes/` (Express), `public/generated/` (AI images)
- Import alias `@/` → `src/`. Backend files use relative imports with `.js` extensions.
- Files you must not modify: `vite.config.ts`, `.env`, `src/utils/auth.ts`

# Auth

```tsx
import { login, logout, onAuthStateChanged } from '@/utils/auth';
useEffect(() => onAuthStateChanged(setUser), []);
<button onClick={() => login('google')}>Sign in</button>
```

# AI proxy

Base URL: `{GORILLA_PROXY}` — pass `$GORILLA_API_KEY` as the Authorization Bearer token.

- LLM chat:     `POST {GORILLA_PROXY}/api/v1/chat/completions`  (omit the model field)
- Image gen:    `POST {GORILLA_PROXY}/api/v1/images/generations` → save base64 to `public/generated/`
- STT:          `POST {GORILLA_PROXY}/api/v1/audio/transcriptions`
- BG removal:   `POST {GORILLA_PROXY}/api/v1/images/remove-background`

# Engineering judgment

You bring a senior engineer's judgment to each decision. When the spec is open, you choose conservatively and in sympathy with what's already in the codebase. You prefer established patterns and keep edits tightly scoped.

**Greenfield build order** (one batch per turn):
1. `src/index.css` — complete design system (custom properties, dark palette, typography, Google Font)
2. Shared components — Navbar + Footer (one turn) → primitives (next turn if needed)
3. Page components — two independent pages per turn until done
4. Backend routes — one or two route files per turn
5. `src/App.tsx` — wire all routes, auth guards (solo write, paired with one other small edit if any)
6. `server.js` — mount all routes (solo write, or paired with the last route file)
7. run_bash: npm install (if needed) → start → verify both 200s

This order makes the live preview look good from the very first hot-reload, and the small batches mean you catch each new component's import errors immediately instead of debugging a wall of failures at the end.

# Frontend quality

Interfaces feel rich and domain-appropriate. A SaaS dashboard is quiet and work-focused; a game can be expressive. Use lucide-react icons, keep border-radius ≤ 8px, build tooltips for icon-only buttons, no decorative gradient orbs, ensure text fits all viewports.

# When something goes wrong

Read first. One run_bash call:
```
cat /tmp/dev.log | tail -60
```
Then make the smallest fix that addresses the root cause. Don't refactor on the way to a fix. If a component is missing an import, add the import — don't rewrite the component.

# Verification before done

```
curl -so /dev/null -w '%{http_code}' http://localhost:8080 && \
curl -so /dev/null -w '%{http_code}' http://localhost:3000
```

Both must return 200 before you call mark_done.

DELETE the dev.log each time you read it to prevent cache.

# Autonomy

Stay with the work until the task is handled end to end. Work through blockers rather than stopping and asking — unless something is genuinely impossible without user input.

# Tool call mechanics

You have three tools: write_file, run_bash, mark_done. The tool definitions and the exact XML format you must use are documented in the # Tools section below this one.

The mental model: every turn ends with a single <tool_call>...</tool_call> block. Inside that block you place 1 or 2 <function=name>...</function> sub-blocks (the batch limit). Before the <tool_call> you can write a brief plain-text thought (1-3 sentences) explaining what you're doing. After the </tool_call> you stop — the next turn starts after the sandbox executes your calls and shows you the result.

Two practical examples of well-formed turns:

Single file write:
```
I'll set up the design system foundation first — this is the file every component will pull from.

<tool_call>
<function=write_file>
<parameter=path>
src/index.css
</parameter>
<parameter=content>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk');
:root { --bg: #0a0a0f; --fg: #f5f5f7; }
body { background: var(--bg); color: var(--fg); font-family: 'Space Grotesk', sans-serif; }
</parameter>
<parameter=reason>
design system foundation
</parameter>
</function>
</tool_call>
```

Two files in one batch:
```
Building the shared chrome — Navbar and Footer can ship together since they don't depend on each other.

<tool_call>
<function=write_file>
<parameter=path>
src/components/Navbar.tsx
</parameter>
<parameter=content>
...full file content...
</parameter>
</function>
<function=write_file>
<parameter=path>
src/components/Footer.tsx
</parameter>
<parameter=content>
...full file content...
</parameter>
</function>
</tool_call>
```

Read a file (no batch limit, run_bash is one call):
```
Need to see the existing routing before I touch anything.

<tool_call>
<function=run_bash>
<parameter=command>
cat src/App.tsx
</parameter>
</function>
</tool_call>
```

Verify and finish (two separate turns):
```
Both files are written. Time to verify the servers come up clean.

<tool_call>
<function=run_bash>
<parameter=command>
curl -so /dev/null -w '%{http_code}\n' http://localhost:8080 && curl -so /dev/null -w '%{http_code}\n' http://localhost:3000
</parameter>
</function>
</tool_call>
```

(then next turn, after seeing 200 200:)
```
Both ports are healthy. Shipping it.

<tool_call>
<function=mark_done>
<parameter=summary>
Built the dashboard with Navbar, three pages, and the items API route. Both servers are healthy.
</parameter>
</function>
</tool_call>
```
"""


# ---------------------------------------------------------------------------
# Conditional addons
# ---------------------------------------------------------------------------

SUPABASE_ADDON = r"""
# Supabase

Active. Env vars: `$VITE_SUPABASE_URL`, `$VITE_SUPABASE_ANON_KEY`, `$SUPABASE_MGMT_TOKEN`, `$SUPABASE_PROJECT_REF`. Package `@supabase/supabase-js` is installed.

```ts
import { createClient } from '@supabase/supabase-js';
const supabase = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);
```

Run migrations via run_bash:
```
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
# Debug mode

You are fixing a specific bug. Turn 1: run_bash to read the error log. Turn 2 (or later): write_file with the smallest possible fix. Then verify. Do not refactor. Do not add features. The 2-file batch limit still applies — most bugs need only one file changed anyway.
"""

EXPANDER_SUPABASE_ADDON = """
Supabase is available for data persistence. Plan to use it when users need to save data across sessions or share data between users. Specify tables, columns, and which need Row Level Security. Don't force it on purely client-side apps."""

PLANNER_SUPABASE_ADDON = """
Supabase is available. Include migrations when the app genuinely stores persistent data."""


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Expander
# ═══════════════════════════════════════════════════════════════════════════

def _build_expander_system(gorilla_proxy_url: str) -> str:
    proxy = gorilla_proxy_url or "{GORILLA_PROXY}"
    return f"""You are a product designer for Gorilla Builder — a platform that builds real working SaaS apps.

The developer's sandbox has access to these capabilities — spec features that use them:

**Auth gateway** (zero-setup login):
```tsx
import {{ login, logout, onAuthStateChanged }} from '@/utils/auth';
<button onClick={{() => login('google')}}>Sign in with Google</button>
```
Use auth whenever the app saves per-user data or has a dashboard the user returns to.

**AI proxy** (base URL: `{proxy}`, auth via `$GORILLA_API_KEY`):
- LLM chat:       `POST {proxy}/api/v1/chat/completions`  — omit model field
- Image gen:      `POST {proxy}/api/v1/images/generations` — returns base64, save to `public/generated/`
- Speech-to-text: `POST {proxy}/api/v1/audio/transcriptions`
- BG removal:     `POST {proxy}/api/v1/images/remove-background`

Use these for features that genuinely benefit from AI — not as decorations.

**Express backend**: full API server at port 3000, routes in `routes/`, can store data, proxy AI calls, handle logic.

Take the user's short idea and expand it into a concrete product spec (200–350 words).

A good spec describes a FUNCTIONAL APP — what does the user actually do? What gets created, saved, generated, or shared? Which AI capability makes the core feature work?

Include:
- App name (creative, memorable)
- Color scheme (specific hex codes, dark mode preferred)
- Typography (a distinctive Google Font — not Inter, not system-ui)
- 3+ pages: what the user sees and does on each
- Backend: API routes, what data is stored
- AI integration: which proxy endpoint, what it does, how the result is shown
- Auth: which provider and why (only if the app genuinely needs it)

For minor tasks or bug fixes: restate the task in 1–2 sentences. No expansion needed.

If an image is attached, treat it as a UI mockup — extract layout, palette, and flows.

Output only the spec. No preamble."""


async def expand_prompt(
    short_prompt: str,
    has_supabase: bool = False,
    image_b64: Optional[str] = None,
    gorilla_proxy_url: str = "",
) -> str:
    if len(short_prompt) > 300:
        return short_prompt

    system = _build_expander_system(gorilla_proxy_url)
    if has_supabase:
        system += "\n" + EXPANDER_SUPABASE_ADDON

    user_content: Any
    if image_b64:
        img_url = (
            image_b64 if image_b64.startswith("data:")
            else f"data:image/jpeg;base64,{image_b64}"
        )
        user_content = [
            {"type": "text",      "text": short_prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    else:
        user_content = short_prompt

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
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
#  Planner
# ═══════════════════════════════════════════════════════════════════════════

def _build_planner_system(gorilla_proxy_url: str) -> str:
    return f"""You are a project planner for Gorilla Builder — React + Vite + Express full-stack apps.

The agent uses three tools: write_file (max 2 per turn), run_bash, mark_done.

Produce a markdown checklist where each item = ONE turn of the agent (one tool_call block, max 2 invokes).

**Hard rule: at most 2 file writes per checklist item.** If you'd want 5 components in one step, split into 3 steps (2+2+1). This keeps the agent honest about observing intermediate results.

**Step order:**
1. Explore — run_bash: cat src/App.tsx server.js src/index.css
2. write_file: src/index.css (one big design system file — solo)
3. write_file (batch of 2): Navbar.tsx + Footer.tsx
4. write_file (batch of 2): page A + page B
5. write_file (batch of 2): page C + maybe a primitive
6. write_file (batch of 2): backend route file 1 + route file 2
7. write_file (batch of 2): src/App.tsx + server.js (the wire-up)
8. run_bash: npm install (only if a package isn't pre-installed)
9. run_bash: start dev server + verify both ports 200

**Format:**
```
# Task: <short title>

- [ ] Read structure — run_bash: cat src/App.tsx server.js src/index.css
- [ ] write_file: src/index.css — dark design system, CSS vars, Google Font
- [ ] write_file (batch): Navbar.tsx + Footer.tsx
- [ ] write_file (batch): Landing.tsx + Dashboard.tsx
- [ ] write_file (batch): About.tsx + components/Card.tsx
- [ ] write_file (batch): routes/api.js + routes/items.js
- [ ] write_file (batch): src/App.tsx + server.js (wire everything)
- [ ] run_bash: start server + verify 200 200
```

Rules: 6–12 items. Every batch item names exactly 2 files. First step explores. Last step verifies. For debug/minor tasks: 2-3 items. Output only the checklist."""


async def generate_plan(
    expanded_prompt: str,
    file_tree_summary: str,
    has_supabase: bool = False,
    image_b64: Optional[str] = None,
    gorilla_proxy_url: str = "",
) -> Optional[str]:
    system = _build_planner_system(gorilla_proxy_url)
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
            {"type": "text",      "text": user_text},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
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
#  Reviewer
# ═══════════════════════════════════════════════════════════════════════════

REVIEWER_SYSTEM = """You are a code reviewer. A developer just finished building a web app. Review the file listing and recent build output for obvious mistakes only.

Check for: components created but not imported anywhere, missing npm installs, Express routes not mounted in server.js, TypeScript errors in logs.

Respond with ONLY:
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
#  XML repair — handles common Qwen3-Coder output defects
#
#  MiMo (and Qwen3-Coder generally) sometimes produces malformed XML when:
#    1. Tool call gets cut off in the middle of a parameter (no closing tag)
#    2. Missing </tool_call> when generation hits max_tokens mid-block
#    3. Stray whitespace/newlines around parameter content (harmless,
#       but we normalize so the regex parser is happier)
# ═══════════════════════════════════════════════════════════════════════════

def _repair_qwen3_xml(text: str) -> str:
    """
    Repair common Qwen3-Coder XML defects so the parser succeeds even when
    the model output isn't perfectly formed. Safe on already-valid XML.
    """
    # Auto-close <tool_call> if the model started one but didn't finish
    if text.count("<tool_call>") > text.count("</tool_call>"):
        text = text + "\n</tool_call>"

    # Auto-close trailing <function=...> if missing </function>
    open_funcs  = len(re.findall(r"<function=[^>]+>", text))
    close_funcs = text.count("</function>")
    if open_funcs > close_funcs:
        diff = open_funcs - close_funcs
        # Insert closes BEFORE </tool_call> if present
        if "</tool_call>" in text:
            text = text.replace("</tool_call>", "</function>" * diff + "</tool_call>", 1)
        else:
            text = text + ("</function>" * diff)

    # Auto-close trailing <parameter=...> if missing </parameter>
    open_params  = len(re.findall(r"<parameter=[^>]+>", text))
    close_params = text.count("</parameter>")
    if open_params > close_params:
        diff = open_params - close_params
        # Insert closes BEFORE </function> (the immediate parent)
        # Walk back from the end and inject closes where needed
        # Simple heuristic: stick them right before the first </function>
        if "</function>" in text:
            text = text.replace("</function>", "</parameter>" * diff + "</function>", 1)
        else:
            text = text + ("</parameter>" * diff)

    return text


# ═══════════════════════════════════════════════════════════════════════════
#  XML tool-call parser — Qwen3-Coder / MiMo native format
#
#  Parses:
#    <tool_call>
#      <function=write_file>
#        <parameter=path>
#        src/App.tsx
#        </parameter>
#        <parameter=content>
#        ...full content...
#        </parameter>
#      </function>
#      <function=run_bash>
#        <parameter=command>
#        npm install
#        </parameter>
#      </function>
#    </tool_call>
#
#  Falls back to ```bash blocks for non-MiMo turns. Enforces BATCH_LIMIT
#  by truncating extras with a logged warning.
# ═══════════════════════════════════════════════════════════════════════════

# DOTALL via [\s\S] so parameter values can contain anything — code, JSON,
# base64, multi-line strings.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>")
_FUNCTION_RE        = re.compile(r"<function=([^>\s]+)>([\s\S]*?)</function>")
_PARAMETER_RE       = re.compile(r"<parameter=([^>\s]+)>([\s\S]*?)</parameter>")


def _strip_param_value(raw: str) -> str:
    """
    Qwen3-Coder convention: parameter values often have a leading and
    trailing newline (the format shows them on their own lines). Strip
    those without touching internal whitespace, since file contents need
    their formatting preserved exactly.
    """
    if not raw:
        return raw
    # Strip exactly ONE leading newline if present (the formatting one)
    if raw.startswith("\r\n"):
        raw = raw[2:]
    elif raw.startswith("\n"):
        raw = raw[1:]
    # Strip exactly ONE trailing newline if present
    if raw.endswith("\r\n"):
        raw = raw[:-2]
    elif raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _parse_xml_functions(block_inner: str) -> List[Dict[str, Any]]:
    """Extract the function-call list from inside a single <tool_call> block."""
    functions: List[Dict[str, Any]] = []
    for m in _FUNCTION_RE.finditer(block_inner):
        name   = m.group(1).strip()
        body   = m.group(2)
        params: Dict[str, str] = {}
        for pm in _PARAMETER_RE.finditer(body):
            pname = pm.group(1).strip()
            pval  = _strip_param_value(pm.group(2))
            params[pname] = pval
        functions.append({"name": name, "params": params})
    return functions


def _parse_response(raw_text: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "thought":      str,    # text before the tool_call block
        "write_files":  [{"path", "content", "reason"}],  # max BATCH_LIMIT
        "bash":         str,    # merged bash from all run_bash calls
        "done":         bool,
        "message":      str,    # user-facing summary
        "extra_writes_dropped": int,  # how many writes we cut for safety
      }
    """
    result = {
        "thought":              "",
        "write_files":          [],
        "bash":                 "",
        "done":                 False,
        "message":              "",
        "extra_writes_dropped": 0,
    }

    if not raw_text:
        return result

    repaired = _repair_qwen3_xml(raw_text)
    blocks   = _TOOL_CALL_BLOCK_RE.findall(repaired)

    # ── Path 1: native Qwen3-Coder XML ─────────────────────────────────────
    if blocks:
        first_pos = repaired.find("<tool_call>")
        result["thought"] = repaired[:first_pos].strip() if first_pos > 0 else ""

        bash_parts:    List[str] = []
        all_functions: List[Dict[str, Any]] = []
        for block_inner in blocks:
            all_functions.extend(_parse_xml_functions(block_inner))

        for fn in all_functions:
            n = fn["name"]
            p = fn["params"]

            if n == "write_file":
                # Defense-in-depth batch limit. Even if the model ignores
                # the system prompt instruction, the agent CANNOT ship
                # more than BATCH_LIMIT files per turn.
                if len(result["write_files"]) < BATCH_LIMIT:
                    result["write_files"].append({
                        "path":    p.get("path", "").strip(),
                        "content": p.get("content", ""),
                        "reason":  p.get("reason", "").strip(),
                    })
                else:
                    result["extra_writes_dropped"] += 1

            elif n == "run_bash":
                cmd = p.get("command", "").strip()
                if cmd:
                    bash_parts.append(cmd)

            elif n == "mark_done":
                result["done"]    = True
                result["message"] = p.get("summary", "Done.").strip()

        result["bash"] = "\n\n".join(bash_parts)

        if not result["message"]:
            if result["thought"]:
                result["message"] = result["thought"].split("\n")[0][:300]
            else:
                paths = [w["path"] for w in result["write_files"]]
                if paths:
                    result["message"] = f"Working on {', '.join(paths)}"
                elif result["bash"]:
                    result["message"] = ""

        return result

    # ── Path 2: legacy fallback (vision turn etc.) ─────────────────────────
    # GORILLA_DONE marker (v14 compat)
    if "GORILLA_DONE" in raw_text:
        result["done"] = True
        parts   = raw_text.split("GORILLA_DONE", 1)
        body    = parts[0]
        summary = parts[1].strip() if len(parts) > 1 else ""
        bash_blocks = re.findall(r"```(?:bash|sh|shell)?\n([\s\S]*?)```", body)
        result["bash"]    = "\n\n".join(b.strip() for b in bash_blocks if b.strip())
        result["thought"] = body[:body.find("```")].strip() if "```" in body else body.strip()
        result["message"] = summary or result["thought"].split("\n")[0][:300] or "Done."
        return result

    # Plain bash blocks
    bash_blocks = re.findall(r"```(?:bash|sh|shell)?\n([\s\S]*?)```", raw_text)
    if bash_blocks:
        result["bash"]    = "\n\n".join(b.strip() for b in bash_blocks if b.strip())
        first_pos         = raw_text.find("```")
        result["thought"] = raw_text[:first_pos].strip() if first_pos > 0 else ""
    else:
        result["thought"] = raw_text.strip()

    if result["thought"]:
        sentences = re.split(r"(?<=[.!?])\s+", result["thought"])
        result["message"] = " ".join(sentences[:3])[:300]

    return result


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
#  LLM call — XML-native, NO `tools` parameter on the API
#
#  CRITICAL: we do NOT pass `tools` or `tool_choice` to OpenRouter. MiMo's
#  chat_template uses Hermes-style JSON when those parameters are present,
#  and MiMo (like Qwen3-Coder) is much less reliable in that mode. Tools
#  live in the system prompt as the Qwen3-Coder XML <tools> block, exactly
#  as the model was trained to receive them.
#
#  Reasoning preservation: per Xiaomi's official docs, multi-turn tool
#  calls degrade if reasoning_content isn't preserved between turns. We
#  pass reasoning.exclude=false and feed the reasoning back into the
#  conversation history.
# ═══════════════════════════════════════════════════════════════════════════

async def _call_llm(
    messages:    list,
    model:       str   = MODEL,
    temperature: float = 0.6,
) -> Tuple[str, int]:
    messages = _compress_history(messages)

    # Strip non-API fields (reasoning_content) from outgoing messages — they
    # were stored locally for context but the API doesn't accept them as
    # input. We move them into the content prefix instead.
    api_messages = []
    for m in messages:
        api_msg = {"role": m["role"], "content": m.get("content", "")}
        # Don't merge reasoning into API content — already in conversation
        # naturally via the assistant text (we keep things simple).
        api_messages.append(api_msg)

    payload: Dict[str, Any] = {
        "model":       model,
        "messages":    api_messages,
        "max_tokens":  16000,
        "temperature": temperature,
        "provider": {
            "order":           ["xiaomi", "fireworks", "alibaba", "novita"],
            "allow_fallbacks": False,
        },
    }

    # Preserve reasoning between turns for MiMo models — Xiaomi's docs are
    # explicit that performance degrades without this. Same flag for any
    # Qwen-family model. No-op on non-reasoning models.
    if any(x in model.lower() for x in ["mimo", "qwen", "minimax"]):
        payload["reasoning"] = {"exclude": False}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  SITE_URL,
        "X-Title":       SITE_NAME,
    }

    data = None
    for attempt in range(5):
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if resp.status_code == 429:
                wait = 0.5 * (attempt + 1)
                log_agent("agent", f"429 — waiting {wait}s (attempt {attempt + 1}/5)")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break

    if data is None:
        raise RuntimeError("LLM call failed after 5 attempts (persistent 429)")

    choice  = data["choices"][0]
    msg     = choice["message"]
    content = msg.get("content") or ""

    # MiMo (and Qwen3) often return reasoning as a separate field. If the
    # tool_call landed only in `reasoning_content`, we pull it forward into
    # `content` so the parser sees it. This is the documented multi-turn
    # behavior — see platform.xiaomimimo.com/docs/api/chat/openai-api.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if reasoning and "<tool_call>" not in content and "<tool_call>" in str(reasoning):
        content = str(reasoning) + "\n\n" + content
    elif reasoning and "<tool_call>" not in content and not content.strip():
        # Sometimes content is empty and the tool_call is *only* in reasoning
        content = str(reasoning)

    u = data.get("usage", {})
    p = u.get("prompt_tokens", 0)
    c = u.get("completion_tokens", 0)
    is_frontier = any(x in model for x in ["claude", "gpt-4", "gemini"])
    is_mimo     = "mimo" in model.lower()
    if is_frontier:
        weight = p * 0.6 + c * 2.4
    elif is_mimo:
        weight = p * 0.8 + c * 2.8
    else:
        weight = p * 0.3 + c * 1.2

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
    return f"\n\n# User preferences\n{lines}"


# ═══════════════════════════════════════════════════════════════════════════
#  Smart model routing
#
#  Both MODEL and SMART_MODEL are MiMo in v17, so this is mostly a no-op
#  switch — but the routing scaffolding is preserved so a future swap
#  (e.g. Claude Sonnet for hard cases) is a one-env-var change.
# ═══════════════════════════════════════════════════════════════════════════

_HARD_OBSERVATION_SIGNALS = frozenset([
    "error ts", "syntaxerror", "cannot find", "could not be resolved",
    "is not defined", "is not a function", "failed to compile",
    "exit code: 1", "exit code: -1", "enoent", "module not found",
    "cannot read propert", "typeerror", "referenceerror",
])

_HARD_DOMAIN_SIGNALS = frozenset([
    "auth", "supabase", "migration", "rls", "policy", "realtime",
    "subscription", "webhook", "oauth", "race condition", "async",
    "promise", "cors", "jwt", "token", "session", "cookie",
    "database", "schema", "foreign key", "join",
])


def _pick_model(
    turn: int,
    previous_output: Optional[str],
    user_request: str,
    is_debug: bool,
) -> str:
    obs = (previous_output or "").lower()
    req = (user_request    or "").lower()

    if turn == 0:
        return SMART_MODEL
    if any(sig in obs for sig in _HARD_OBSERVATION_SIGNALS):
        return SMART_MODEL
    if turn <= 4 and any(sig in req for sig in _HARD_DOMAIN_SIGNALS):
        return SMART_MODEL
    if is_debug and previous_output:
        return SMART_MODEL

    return MODEL


# ═══════════════════════════════════════════════════════════════════════════
#  LineageAgent  v17
#
#  Conversation shape:
#    [system]    full prompt + Qwen3-Coder <tools> block + addons (immutable)
#    [user]      initial task w/ file tree, plan, optional image
#    [assistant] thought + <tool_call>...</tool_call>
#    [user]      OBSERVATION: <bash output OR "Files written: a.tsx, b.tsx">
#    [assistant] thought + next tool_call
#    ...
#    [assistant] <function=mark_done>...</function>
#
#  Observations come back as plain user messages (NOT tool-role messages),
#  because Qwen3-Coder/MiMo were trained to read them in the user channel.
# ═══════════════════════════════════════════════════════════════════════════

class LineageAgent:
    """
    Two-phase, batch-limited XML agent for MiMo-family models.
    Sandbox contract is unchanged from v15/v16 — _parsed dict on every
    result, sandbox_manager calls record_tool_results() after executing.
    """

    def __init__(self, project_id: str):
        self.project_id         = project_id
        self.total_tokens       = 0
        self.token_sub          = TokenSubstitution()
        self.messages:          List[Dict[str, Any]] = []
        self._system_prompt_set = False
        self._plan_injected     = False
        self._prompt_expanded   = False
        self._gorilla_proxy_url: str  = ""
        self._has_supabase:      bool = False

    def _ensure_system_prompt(
        self,
        gorilla_proxy_url: str,
        has_supabase: bool,
        is_debug: bool,
        agent_skills: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._system_prompt_set:
            return

        # 1. Domain prompt body (Gorilla / Vite / Express / AI proxy)
        prompt = SYSTEM_PROMPT_BODY.replace(
            "{GORILLA_PROXY}",
            gorilla_proxy_url or "https://your-proxy.ngrok-free.dev",
        )

        # 2. Conditional addons (Supabase / Debug)
        if has_supabase:
            prompt += "\n" + SUPABASE_ADDON
        if is_debug:
            prompt += "\n" + DEBUG_ADDON

        # 3. Tools block — appended LAST so the JSON schema sits closest to
        #    where the model starts generating. Qwen3-Coder's training data
        #    has tools at the end of the system prompt (verified against
        #    the official chat_template.jinja).
        prompt += "\n\n" + _format_tools_for_prompt()

        # 4. User skills/preferences
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
        return stripped if stripped.startswith("data:") else f"data:image/jpeg;base64,{stripped}"

    def _append_observation(
        self,
        write_paths:      List[str],
        bash_observation: str,
    ) -> None:
        """
        After the sandbox executes a turn's tool calls, this gets called
        with the results. We render them as a single OBSERVATION user
        message — that's what MiMo/Qwen3-Coder were trained to read.
        """
        parts: List[str] = []
        if write_paths:
            parts.append("Files written: " + ", ".join(write_paths))
        if bash_observation:
            filtered = _filter_observation(bash_observation)[:8000]
            parts.append("Bash output:\n" + filtered)
        if not parts:
            parts.append("(no output)")

        self.messages.append({
            "role":    "user",
            "content": "OBSERVATION:\n" + "\n\n".join(parts),
        })

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

        if gorilla_proxy_url:
            self._gorilla_proxy_url = gorilla_proxy_url
        self._has_supabase = has_supabase

        self._ensure_system_prompt(gorilla_proxy_url, has_supabase, is_debug, agent_skills)

        compressed  = self.token_sub.compress_tree(file_tree)
        clean_paths = sorted(p for p in compressed if not p.endswith(".b64"))
        tree_str    = "\n".join(f"  {p}" for p in clean_paths)

        pkg_snippet = ""
        if "package.json" in compressed:
            pkg = compressed["package.json"]
            if len(pkg) < 3000:
                pkg_snippet = f"\n\npackage.json:\n{pkg}"

        prompt_image_b64: Optional[str] = None
        if not previous_command_output:
            prompt_image_b64 = self._extract_prompt_image(file_tree)
            if prompt_image_b64:
                log_agent("agent", "prompt_image.b64 found", self.project_id)

        # ── Build the new user message ──────────────────────────────────────
        if previous_command_output:
            # Subsequent turn: the sandbox's output becomes the OBSERVATION.
            # (record_tool_results may have already appended this; this
            # branch handles the legacy path where the sandbox passes
            # previous_command_output directly.)
            filtered = _filter_observation(previous_command_output)
            self.messages.append({"role": "user", "content": f"OBSERVATION:\n{filtered[:8000]}"})
        else:
            effective_request = user_request
            plan_text         = ""

            if not is_debug and user_request:
                if not self._prompt_expanded:
                    effective_request = await expand_prompt(
                        user_request,
                        has_supabase=has_supabase,
                        image_b64=prompt_image_b64,
                        gorilla_proxy_url=self._gorilla_proxy_url,
                    )
                    self._prompt_expanded = True

                if not self._plan_injected and bool(file_tree):
                    plan = await generate_plan(
                        effective_request,
                        tree_str,
                        has_supabase=has_supabase,
                        image_b64=prompt_image_b64,
                        gorilla_proxy_url=self._gorilla_proxy_url,
                    )
                    if plan:
                        plan_text = (
                            "\n\nHere's a plan — follow it step by step. "
                            "Each item = one turn. Remember the 2-file batch limit:\n"
                            + plan
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

            user_text        = "\n".join(parts)
            first_turn_image = image_b64 or prompt_image_b64

            if first_turn_image:
                img_url = (
                    first_turn_image if first_turn_image.startswith("data:")
                    else f"data:image/jpeg;base64,{first_turn_image}"
                )
                self.messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": user_text},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                })
            else:
                self.messages.append({"role": "user", "content": user_text})

        # ── Model routing ───────────────────────────────────────────────────
        first_turn_has_image = bool(
            (image_b64 or prompt_image_b64) and not previous_command_output
        )

        if first_turn_has_image:
            # Vision turn: VISION_MODEL handles the image. With v17 this is
            # also a MiMo variant, so it speaks the same XML format — no
            # legacy fallback needed in the common case.
            model = VISION_MODEL
        else:
            turn_index = max(0, len(self.messages) // 2 - 1)
            model = _pick_model(
                turn=turn_index,
                previous_output=previous_command_output,
                user_request=user_request,
                is_debug=is_debug,
            )

        log_agent(
            "agent",
            f"v17 model={model.split('/')[-1]} step={len(self.messages) // 2} batch_limit={BATCH_LIMIT}",
            self.project_id,
        )

        try:
            raw_text, tokens = await _call_llm(
                self.messages,
                model=model,
                temperature=0.6,
            )
            self.total_tokens += tokens
        except Exception as e:
            log_agent("agent", f"LLM error: {e}", self.project_id)
            return {
                "message":     f"AI error: {str(e)[:150]}",
                "write_files": [],
                "commands":    [],
                "done":        True,
                "tokens":      0,
            }

        # ── Record assistant message verbatim (XML and all) ────────────────
        # MiMo wants to see its own tool_call XML in the conversation
        # history — that's how it knows what it already did and avoids
        # re-doing the same work next turn.
        self.messages.append({"role": "assistant", "content": raw_text or ""})

        # ── Parse the XML ───────────────────────────────────────────────────
        parsed = _parse_response(raw_text)

        if parsed["thought"]:
            log_agent("agent", f"THOUGHT: {parsed['thought'][:300]}", self.project_id)

        if parsed["extra_writes_dropped"] > 0:
            log_agent(
                "agent",
                f"⚠ Dropped {parsed['extra_writes_dropped']} extra writes (batch limit = {BATCH_LIMIT})",
                self.project_id,
            )

        # ── Safety-check bash ───────────────────────────────────────────────
        safe_bash = ""
        if parsed["bash"]:
            bash_content = self.token_sub.expand(parsed["bash"])
            if _is_safe(bash_content):
                safe_bash = bash_content
            else:
                log_agent("agent", "Blocked dangerous command", self.project_id)

        # ── Expand blob tokens in file contents ─────────────────────────────
        for wf in parsed["write_files"]:
            wf["content"] = self.token_sub.expand(wf["content"])

        done   = parsed["done"]
        writes = len(parsed["write_files"])
        log_agent(
            "agent",
            f"{'DONE' if done else f'writes={writes} bash={bool(safe_bash)}'} tok={self.total_tokens}",
            self.project_id,
        )

        return {
            "message":     parsed["message"],
            "write_files": parsed["write_files"],     # max BATCH_LIMIT entries
            "commands":    [safe_bash] if safe_bash else [],
            "done":        done,
            "tokens":      self.total_tokens,
            "_parsed":     parsed,
        }

    def record_tool_results(
        self,
        parsed:            Dict[str, Any],
        write_observation: str = "",
        bash_observation:  str = "",
    ) -> None:
        """
        Sandbox calls this AFTER executing the tool calls from the previous
        turn, so the next LLM call has proper context.

        sandbox_manager:
            result = await agent.run(...)
            # ... execute result["write_files"] and result["commands"] ...
            agent.record_tool_results(
                result["_parsed"],
                bash_observation=shell_stdout,
            )
        """
        write_paths = [wf.get("path", "") for wf in parsed.get("write_files", []) if wf.get("path")]
        if write_observation and not bash_observation:
            bash_observation = write_observation
        self._append_observation(write_paths, bash_observation)


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
    "_filter_observation",
    "_parse_response",
    "_repair_qwen3_xml",
    "AGENT_TOOL_DEFS",
    "BATCH_LIMIT",
]