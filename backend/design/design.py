"""
Gorilla Design — Core Engine
==============================
Everything in one file. No separate agent/figma/tokens/serve modules.

Core concept:
  1. LLM generates a complete Figma JSON design in one shot from a brief
  2. Edits are surgical search/replace ops on the JSON tree
  3. Images embed as base64 fills on specific nodes
  4. Export to HTML, Figma JSON download, or hosted static page

No boilerplate frames. No hardcoded structures. The LLM designs everything.
"""

from __future__ import annotations

import os
import re
import json
import copy
import uuid
import httpx
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
DESIGN_MODEL = os.getenv("DESIGN_MODEL", "deepseek/deepseek-v4-pro")
SITE_URL = os.getenv("SITE_URL", "https://gorillabuilder.dev")


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

async def _llm(system: str, user: str, max_tokens: int = 8000) -> str:
    """Call the design LLM. Returns raw text."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": DESIGN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": "Gorilla Design",
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    if not content:
        # Some models return content in a different field when streaming leaks through
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content") or msg.get("text") or ""
            if content:
                break
    if not content:
        raise RuntimeError(f"LLM returned empty content. Model: {DESIGN_MODEL}. Response: {str(data)[:300]}")
    return content.strip()


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling fences and truncation."""
    raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Recover truncated JSON
        depth = 0
        last_valid = 0
        for i, ch in enumerate(raw):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_valid = i + 1
        if last_valid > 0:
            try:
                return json.loads(raw[:last_valid])
            except Exception:
                pass
        raise RuntimeError("LLM returned invalid JSON. Try rephrasing your brief.")


# ---------------------------------------------------------------------------
# IMAGE PLACEHOLDER PROCESSING
# ---------------------------------------------------------------------------
# LLM can put <image prompt="dark mountain hero"/> in any node's
# "characters" field or as "_image_prompt" property.
# After generation, we scan the tree, collect all placeholders,
# generate images in parallel, and embed them as fills.

import re as _re
import asyncio as _asyncio

def _collect_image_placeholders(node: dict, results: list) -> None:
    """Recursively find all nodes with image placeholders."""
    # Check for <image prompt="..."/> in characters
    chars = node.get("characters", "")
    if chars:
        match = _re.search(r'<image\s+prompt=["\'](.*?)["\'"]\s*/>', chars, _re.IGNORECASE)
        if match:
            results.append({"node_id": node["id"], "prompt": match.group(1), "node": node})

    # Check for explicit _image_prompt property
    if node.get("_image_prompt"):
        results.append({"node_id": node["id"], "prompt": node["_image_prompt"], "node": node})

    for child in node.get("children", []):
        _collect_image_placeholders(child, results)


async def _generate_image_for_placeholder(item: dict, tree: dict) -> dict:
    """Generate one image and embed it. Returns updated tree."""
    node_id = item["node_id"]
    prompt = item["prompt"]
    print(f"🖼️ Auto image gen: node={node_id} prompt={prompt[:60]}")
    updated, success = await generate_image_fill(node_id, prompt, tree)
    if success:
        # Also clear the placeholder text
        def clear_placeholder(node):
            if node.get("id") == node_id:
                if node.get("characters", "").startswith("<image"):
                    node["characters"] = ""
                node.pop("_image_prompt", None)
                return True
            for child in node.get("children", []):
                if clear_placeholder(child):
                    return True
            return False
        clear_placeholder(updated)
        return updated
    return tree


async def process_image_placeholders(tree: dict) -> dict:
    """
    Scan tree for <image prompt="..."/> placeholders and generate all images.
    Runs generations in parallel.
    """
    placeholders = []
    _collect_image_placeholders(tree, placeholders)

    if not placeholders:
        return tree

    print(f"🖼️ Found {len(placeholders)} image placeholders, generating...")

    # Generate all in parallel
    tasks = [_generate_image_for_placeholder(item, tree) for item in placeholders]
    results = await _asyncio.gather(*tasks, return_exceptions=True)

    # Apply the last successful result (each generation modifies the full tree)
    for result in results:
        if isinstance(result, dict):
            tree = result

    return tree


# ---------------------------------------------------------------------------
# 1. GENERATE — full design from brief
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """You are an elite UI designer. Generate a Figma JSON design from a brief.

OUTPUT: Single valid JSON object only. No markdown fences. No explanation.

STRUCTURE (follow exactly):
{"id":"frame:0","name":"Name","type":"FRAME","width":1440,"height":900,"fills":[{"type":"SOLID","color":{"r":0.05,"g":0.05,"b":0.08,"a":1},"opacity":1}],"_gorilla_tokens":{"colors":{"background":"#0D0D14","surface":"#161620","accent":"#hex","text-primary":"#FAFAFA","text-secondary":"#888899","border":"#1E1E2E"},"typography":{"fontFamily":"Body Font, sans-serif","displayFont":"Display Font, sans-serif"}},"children":[{"id":"nav:0","name":"Navbar","type":"FRAME","absoluteBoundingBox":{"x":0,"y":0,"width":1440,"height":72},"fills":[{"type":"SOLID","color":{"r":0.08,"g":0.08,"b":0.1,"a":1},"opacity":1}],"children":[{"id":"brand:0","name":"Brand","type":"TEXT","absoluteBoundingBox":{"x":80,"y":20,"width":200,"height":32},"characters":"Brand Name","fills":[{"type":"SOLID","color":{"r":1,"g":1,"b":1,"a":1},"opacity":1}],"style":{"fontFamily":"Display Font","fontWeight":600,"fontSize":22,"letterSpacing":-0.3}}]}]}

RULES:
- color.r/g/b/a = floats 0-1 only, NEVER hex inside color objects
- absoluteBoundingBox on every node
- TEXT nodes need "characters" with real brand-appropriate content
- 4-5 sections max (navbar, hero, one feature section, footer minimum)
- Keep children arrays small — max 4-5 children per section
- Pick fonts matching the brand. Never Inter, Roboto, Arial
- Dark theme unless brief says light
- _gorilla_tokens hex are CSS refs only
- Output ONLY the JSON, nothing else
- For hero backgrounds, product shots, or any node needing a real photo/illustration,
  add "_image_prompt": "detailed description" to that FRAME node.
  The system will auto-generate and embed the image after design creation.
- Use _image_prompt for: hero sections, feature images, backgrounds, product cards"""


async def generate_design(brief: str) -> dict:
    """
    Generate a complete Figma JSON design from a brief.
    This is the main entry point — the LLM designs everything.
    """
    raw = await _llm(GENERATE_SYSTEM, f"Design brief: {brief}", max_tokens=12000)
    result = _parse_json(raw)

    # Ensure required fields
    if "id" not in result:
        result["id"] = "frame:0"
    if "type" not in result:
        result["type"] = "FRAME"
    if "width" not in result:
        result["width"] = 1440
    if "height" not in result:
        result["height"] = 900
    if "_gorilla_tokens" not in result:
        result["_gorilla_tokens"] = {}

    # Process any <image prompt="..."/> placeholders
    result = await process_image_placeholders(result)

    return result


# ---------------------------------------------------------------------------
# 2. EDIT — surgical search/replace on existing design
# ---------------------------------------------------------------------------

EDIT_SYSTEM = """You are a precise JSON editor for Figma design files.

Given a Figma JSON tree summary and an edit instruction, output ONLY a JSON object with ops:

{
  "narration": "Brief description of what changed.",
  "ops": [
    {"type": "replace_text",     "node_id": "text:0",  "value": "New text"},
    {"type": "replace_fill",     "node_id": "hero:0",  "fills": [{"type":"SOLID","color":{"r":0.1,"g":0.1,"b":0.2,"a":1},"opacity":1}]},
    {"type": "replace_property", "node_id": "btn:0",   "key": "cornerRadius", "value": 24},
    {"type": "replace_token",    "path": "colors.accent", "value": "#FF6B35"},
    {"type": "replace_style",    "node_id": "headline:0", "key": "fontSize", "value": 96},
    {"type": "add_node",         "parent_id": "frame:0", "node": {"id":"new-section:0","name":"New Section","type":"FRAME","absoluteBoundingBox":{"x":0,"y":800,"width":1440,"height":400},"fills":[{"type":"SOLID","color":{"r":0.08,"g":0.08,"b":0.1,"a":1},"opacity":1}],"children":[]}},
    {"type": "delete_node",      "node_id": "old:0"}
  ]
}

Op types:
- replace_text: change TEXT node characters field
- replace_fill: change fills array on any node (use r/g/b floats 0-1 in color objects)
- replace_property: change any node property (cornerRadius, opacity, width, height, x, y)
- replace_token: change a _gorilla_tokens value by dot-path (colors.accent, typography.fontFamily)
- replace_style: change a text style property (fontSize, fontWeight, letterSpacing, fontFamily)
- add_node: add a new node to a parent (provide complete node with id, type, absoluteBoundingBox, fills, children)
- delete_node: remove a node by id

Rules:
- Output ONLY the JSON, no markdown fences
- Be surgical — minimum ops to achieve the instruction
- For add_node: generate complete valid nodes with all required fields
- For adding sections: add to the top-level frame (id usually "frame:0")
- node_id must exist in the tree for replace/delete ops
- For darker/lighter: adjust r/g/b values
- When asked to add sections, USE add_node — do not say it's impossible
- For image nodes use: {"type": "replace_text", "node_id": "...", "value": "<image prompt=\"description\"/>"}
  The system will auto-generate the image after applying ops"""


def _tree_summary(node: dict, depth: int = 0, max_depth: int = 4) -> str:
    """Compact summary of node tree for LLM context."""
    if depth > max_depth:
        return ""
    indent = "  " * depth
    t = node.get("type", "?")
    name = node.get("name", "?")
    nid = node.get("id", "?")
    chars = f' "{node["characters"][:30]}"' if t == "TEXT" and node.get("characters") else ""
    bb = node.get("absoluteBoundingBox", {})
    size = f' [{bb.get("width",0):.0f}x{bb.get("height",0):.0f} @ {bb.get("x",0):.0f},{bb.get("y",0):.0f}]' if bb else ""
    lines = [f"{indent}{t} '{name}' id:{nid}{size}{chars}"]
    for child in node.get("children", []):
        s = _tree_summary(child, depth + 1, max_depth)
        if s:
            lines.append(s)
    return "\n".join(lines)


def _apply_ops(tree: dict, ops: list) -> dict:
    """Apply edit ops to a Figma JSON tree. Returns modified copy."""
    tree = copy.deepcopy(tree)

    def find_node(node: dict, nid: str) -> Optional[dict]:
        if node.get("id") == nid:
            return node
        for child in node.get("children", []):
            found = find_node(child, nid)
            if found:
                return found
        return None

    def set_path(obj: dict, path: str, value: Any) -> None:
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj.setdefault(p, {})
        obj[parts[-1]] = value

    for op in ops:
        otype = op.get("type", "")
        nid = op.get("node_id", "")

        if otype == "replace_token":
            tokens = tree.setdefault("_gorilla_tokens", {})
            set_path(tokens, op.get("path", ""), op.get("value"))

        elif otype == "replace_text":
            node = find_node(tree, nid)
            if node:
                node["characters"] = op.get("value", "")

        elif otype == "replace_fill":
            node = find_node(tree, nid)
            if node:
                node["fills"] = op.get("fills", [])

        elif otype == "replace_property":
            node = find_node(tree, nid)
            if node:
                node[op.get("key", "")] = op.get("value")

        elif otype == "replace_style":
            node = find_node(tree, nid)
            if node:
                node.setdefault("style", {})[op.get("key", "")] = op.get("value")

        elif otype == "add_node":
            parent_id = op.get("parent_id", "")
            new_node = op.get("node", {})
            parent = find_node(tree, parent_id)
            if parent is not None:
                parent.setdefault("children", []).append(new_node)
                # Expand frame height to fit new node if needed
                new_bb = new_node.get("absoluteBoundingBox", {})
                if new_bb:
                    needed = new_bb.get("y", 0) + new_bb.get("height", 0)
                    if needed > tree.get("height", 0):
                        tree["height"] = needed + 40

        elif otype == "delete_node":
            def remove_node(node, target_id):
                children = node.get("children", [])
                for i, child in enumerate(children):
                    if child.get("id") == target_id:
                        children.pop(i)
                        return True
                    if remove_node(child, target_id):
                        return True
                return False
            remove_node(tree, nid)

    return tree


async def edit_design(tree: dict, instruction: str) -> Tuple[dict, str]:
    """
    Edit a design via surgical search/replace ops.
    Returns (updated_tree, narration).
    """
    summary = _tree_summary(tree)
    tokens_str = json.dumps(tree.get("_gorilla_tokens", {}), indent=2)

    user_msg = (
        f"Current tokens:\n{tokens_str}\n\n"
        f"Node tree:\n{summary}\n\n"
        f"Instruction: {instruction}"
    )

    raw = await _llm(EDIT_SYSTEM, user_msg, max_tokens=2000)
    result = _parse_json(raw)

    narration = result.get("narration", "Updated.")
    ops = result.get("ops", [])
    updated = _apply_ops(tree, ops)

    # Process any image placeholders added by ops
    updated = await process_image_placeholders(updated)

    return updated, narration


# ---------------------------------------------------------------------------
# 3. IMAGE GENERATION — embed as fill on a node
# ---------------------------------------------------------------------------

async def generate_image_fill(
    node_id: str,
    prompt: str,
    tree: dict,
    user_api_key: str = "",
    proxy_base: str = "",
) -> Tuple[dict, bool]:
    """
    Generate an image via OpenRouter directly and embed as IMAGE fill.
    Bypasses the Gorilla proxy entirely — no auth issues.
    Returns (updated_tree, success).
    """
    import base64 as _b64

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("Image gen: OPENROUTER_API_KEY not set")
        return tree, False

    try:
        # Use a valid OpenRouter image model
        image_model = os.getenv("IMAGE_MODEL", "black-forest-labs/flux.2-klein-4b")
        payload = {
            "model": image_model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("SITE_URL", "https://gorillabuilder.dev"),
            "X-Title": "Gorilla Design",
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )

        if resp.status_code != 200:
            print(f"Image gen OpenRouter {resp.status_code}: {resp.text[:300]}")
            return tree, False

        result = resp.json()
        b64 = None

        # Extract image from response
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            # Some models return images array
            for img in msg.get("images", []):
                url = img.get("image_url", {}).get("url") or img.get("url", "")
                if url:
                    b64 = url
                    break
            # Some models return content as data URI directly
            if not b64 and msg.get("content"):
                c = msg["content"]
                if c.startswith("data:image"):
                    b64 = c

        if not b64:
            print(f"No image in response: {str(result)[:300]}")
            return tree, False

        # If URL, fetch and convert to base64
        if b64.startswith("http"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                img_resp = await client.get(b64)
            if img_resp.status_code != 200:
                print(f"Failed to fetch image URL {img_resp.status_code}")
                return tree, False
            b64 = f"data:image/jpeg;base64,{_b64.b64encode(img_resp.content).decode()}"
        elif not b64.startswith("data:"):
            b64 = f"data:image/jpeg;base64,{b64}"

        # Find node and embed
        tree = copy.deepcopy(tree)

        def find_and_fill(node: dict, nid: str) -> bool:
            if node.get("id") == nid:
                node["fills"] = [{
                    "type": "IMAGE",
                    "scaleMode": "FILL",
                    "imageRef": b64,
                    "opacity": 1,
                    "visible": True,
                }]
                return True
            for child in node.get("children", []):
                if find_and_fill(child, nid):
                    return True
            return False

        success = find_and_fill(tree, node_id)
        return tree, success

    except Exception:
        return tree, False


# ---------------------------------------------------------------------------
# 4. EXPORT — HTML renderer
# ---------------------------------------------------------------------------

def design_to_html(tree: dict) -> str:
    """Render a Figma JSON design tree to a self-contained static HTML page."""
    tokens = tree.get("_gorilla_tokens", {})
    colors = tokens.get("colors", {})
    typography = tokens.get("typography", {})
    font_family = typography.get("fontFamily", "DM Sans, sans-serif")
    display_font = typography.get("displayFont", font_family)
    name = tree.get("name", "Design")
    width = tree.get("width", 1440)
    height = tree.get("height", 900)

    # Google Fonts
    fonts = set()
    for f in [font_family, display_font]:
        base = f.split(",")[0].strip()
        if base.lower() not in ("system-ui", "sans-serif", "serif", "monospace"):
            fonts.add(base.replace(" ", "+"))

    font_links = "\n".join(
        f'<link href="https://fonts.googleapis.com/css2?family={f}:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
        for f in fonts
    )

    css_vars = "\n".join(f"  --color-{k}: {v};" for k, v in colors.items())

    # Get frame origin from first child or default to 0,0
    frame_bb = tree.get("absoluteBoundingBox", {})
    frame_x = frame_bb.get("x", 0)
    frame_y = frame_bb.get("y", 0)

    body_parts: List[str] = []
    _render_node(tree.get("children", []), body_parts, frame_x, frame_y)

    bg_color = _fill_color(tree.get("fills", [])[0]) if tree.get("fills") else colors.get("background", "#0D0D14")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
{font_links}
<style>
:root {{
{css_vars}
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: {font_family};
  background: {bg_color};
  color: {colors.get("text-primary", "#FAFAFA")};
  overflow-x: hidden;
  min-height: 100vh;
}}
.gd-frame {{
  position: relative;
  width: {width}px;
  min-height: {height}px;
  margin: 0 auto;
  overflow: hidden;
  background: {bg_color};
}}
.gd-node {{ position: absolute; box-sizing: border-box; }}
@media (max-width: {width}px) {{
  .gd-frame {{ width: 100%; transform-origin: top left; }}
}}
</style>
</head>
<body>
<div class="gd-frame">
{"".join(body_parts)}
</div>
<script>
// Scale frame to fit viewport
function scaleFrame() {{
  const frame = document.querySelector('.gd-frame');
  const vw = window.innerWidth;
  const fw = {width};
  if (vw < fw) {{
    const scale = vw / fw;
    frame.style.transform = 'scale(' + scale + ')';
    frame.style.marginBottom = '-' + (fw * (1 - scale)) + 'px';
  }}
}}
scaleFrame();
window.addEventListener('resize', scaleFrame);
</script>
</body>
</html>"""


def _render_node(nodes: list, parts: List[str], parent_x: float, parent_y: float) -> None:
    for node in nodes:
        bb = node.get("absoluteBoundingBox", {})
        x = bb.get("x", 0) - parent_x
        y = bb.get("y", 0) - parent_y
        w = bb.get("width", 0)
        h = bb.get("height", 0)

        style = f"left:{x:.1f}px;top:{y:.1f}px;width:{w:.1f}px;height:{h:.1f}px;"

        # Fill
        fills = node.get("fills", [])
        fill = next((f for f in fills if f.get("visible", True) and f.get("type") in ("SOLID","IMAGE")), None)
        if fill:
            if fill.get("type") == "SOLID":
                c = fill.get("color", {})
                r = int(c.get("r", 0) * 255)
                g = int(c.get("g", 0) * 255)
                b = int(c.get("b", 0) * 255)
                a = fill.get("opacity", c.get("a", 1))
                style += f"background:rgba({r},{g},{b},{a:.2f});"
            elif fill.get("type") == "IMAGE" and fill.get("imageRef", "").startswith("data:"):
                style += f"background-image:url({fill['imageRef']});background-size:cover;background-position:center;"

        if node.get("cornerRadius"):
            style += f"border-radius:{node['cornerRadius']}px;"
        if node.get("opacity") is not None and node["opacity"] < 1:
            style += f"opacity:{node['opacity']};"

        strokes = node.get("strokes", [])
        sw = node.get("strokeWeight", 0)
        if strokes and sw:
            sc = _fill_color(strokes[0])
            style += f"border:{sw}px solid {sc};"

        if node.get("type") == "TEXT":
            ts = node.get("style", {})
            fs = ts.get("fontSize", 16)
            fw = ts.get("fontWeight", 400)
            ff = ts.get("fontFamily", "inherit")
            ls = ts.get("letterSpacing", 0)
            lh = ts.get("lineHeightPx", fs * 1.4)
            ta = ts.get("textAlignHorizontal", "LEFT").lower()
            # Text color from fills, fallback to white for dark designs
            text_color = _fill_color(fills[0]) if fills else "rgba(250,250,250,1)"
            style += (
                f"font-size:{fs}px;font-weight:{fw};font-family:'{ff}',sans-serif;"
                f"letter-spacing:{ls}px;line-height:{lh:.0f}px;text-align:{ta};"
                f"color:{text_color};overflow:hidden;white-space:pre-wrap;"
            )
            text = node.get("characters", "")
            parts.append(f'<div class="gd-node" style="{style}">{text}</div>')
        else:
            parts.append(f'<div class="gd-node" style="{style}">')
            _render_node(node.get("children", []), parts, bb.get("x", 0), bb.get("y", 0))
            parts.append("</div>")


def _fill_color(fill: dict) -> str:
    if fill.get("type") == "SOLID":
        c = fill.get("color", {})
        r, g, b = int(c.get("r", 0) * 255), int(c.get("g", 0) * 255), int(c.get("b", 0) * 255)
        a = fill.get("opacity", c.get("a", 1))
        return f"rgba({r},{g},{b},{a:.2f})"
    return "transparent"


# ---------------------------------------------------------------------------
# 5. HOST — static page
# ---------------------------------------------------------------------------

def generate_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "design"
    return f"{slug}-{str(uuid.uuid4())[:8]}"


def render_hosted_page(tree: dict, design_id: str = "", site_url: str = "") -> str:
    html = design_to_html(tree)
    badge = ""
    if design_id:
        badge = f"""<a href="{site_url}/design/editor/{design_id}"
style="position:fixed;bottom:20px;right:20px;background:#0066ff;color:#fff;
padding:8px 16px;border-radius:8px;font-family:DM Sans,sans-serif;font-size:13px;
text-decoration:none;font-weight:500;box-shadow:0 4px 16px rgba(0,102,255,0.4);z-index:9999;">
Edit in Gorilla Design</a>"""
    return html.replace("</body>", f"{badge}\n</body>")