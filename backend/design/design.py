"""
Gorilla Design — HTML-First Engine
====================================
Flow:
  1. LLM generates pure HTML (no JS) — natural for LLMs, readable, editable
  2. HTML → Figma JSON via CSS parsing + layout calculation
  3. Images: <img data-generate="prompt"> auto-generates via OpenRouter
  4. Figma export: proper Figma plugin clipboard format (paste directly into Figma)
  5. Serve: original HTML served directly, no re-rendering needed
"""

from __future__ import annotations

import os
import re
import json
import copy
import uuid
import base64
import httpx
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DESIGN_MODEL = os.getenv("DESIGN_MODEL", "deepseek/deepseek-v4-pro")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/flux.2-klein-4b")
SITE_URL = os.getenv("SITE_URL", "https://gorillabuilder.dev")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

async def _llm(system: str, user: str, max_tokens: int = 80000) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": "Gorilla Design",
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:400]}")
    content = resp.json()["choices"][0]["message"].get("content", "")
    if not content:
        raise RuntimeError("LLM returned empty response")
    return content.strip()


# ---------------------------------------------------------------------------
# 1. GENERATE — LLM produces pure HTML
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """You are an elite UI designer. Generate a complete, stunning single-page HTML design.

OUTPUT: Pure HTML only — no JavaScript, no explanations, no markdown fences.
Start your response with <!DOCTYPE html> and end with </html>.

MANDATORY STRUCTURE — you MUST include ALL of these sections:
1. Navbar (height: 80px)
2. Hero section with large headline and CTA (height: 700px)
3. Features or Products section with 3+ cards (height: 500px)
4. About or Story section (height: 400px)
5. Testimonials or Stats section (height: 350px)
6. Footer (height: 200px)
TOTAL HEIGHT: at least 2230px. More sections = better design.

POSITIONING RULES:
- Every element: position:absolute with exact left/top/width/height in px
- Body has position:relative, min-height matches total content height
- Sections stack vertically — each section's top = previous section's top + height
- Section divs: position:absolute; left:0; width:1440px; height:Npx; top:Npx
- Child elements inside sections: position:absolute relative to their parent section

STYLE RULES:
- Inline all CSS in <style> tag — use @import for Google Fonts at the very top
- Use CSS variables in :root for all colors
- NEVER use Inter, Roboto, Arial, system-ui — pick distinctive Google Fonts
- Dark themes strongly preferred

IMAGE RULES — IMPORTANT:
- For ANY photo, illustration, or background image use EXACTLY:
  <img data-generate="your detailed prompt" style="position:absolute;left:0px;top:0px;width:1440px;height:700px;object-fit:cover;">
- Use data-generate for: hero backgrounds, product photos, portraits, team photos
- NEVER use picsum, unsplash, placeholder.com or any external image URLs
- AI images will be auto-generated and embedded

OTHER RULES:
- Add data-node-id to every section: data-node-id="navbar", "hero", "features", "footer" etc
- Real content only — no Lorem Ipsum, use actual copy matching the brief
- Output ONLY the HTML, nothing else before or after"""

EDIT_SYSTEM = """You are an elite HTML editor. Given existing HTML and an edit instruction, output the complete updated HTML.

OUTPUT: Complete HTML only — no explanations, no markdown fences.
Start with <!DOCTYPE html> and end with </html>.

RULES:
- Return the FULL updated HTML, not just the changed parts
- Preserve all existing data-node-id attributes
- For new image placeholders use: <img data-generate="prompt" style="...">
- Keep all CSS variables and Google Font imports
- Output ONLY the HTML"""


async def generate_design(brief: str) -> Dict[str, Any]:
    """
    Generate a complete design from a brief.
    Returns dict with html, figma_json, tokens, name.
    """
    raw_html = await _llm(GENERATE_SYSTEM, f"Design brief: {brief}", max_tokens=80000)

    # Clean up any accidental markdown fences
    raw_html = re.sub(r'^```(?:html)?\n?', '', raw_html.strip())
    raw_html = re.sub(r'\n?```$', '', raw_html.strip())

    if not raw_html.strip().startswith('<!'):
        # LLM added preamble — find the HTML
        idx = raw_html.find('<!DOCTYPE')
        if idx == -1:
            idx = raw_html.find('<html')
        if idx >= 0:
            raw_html = raw_html[idx:]

    # Extract name from title tag
    title_match = re.search(r'<title>(.*?)</title>', raw_html, re.IGNORECASE)
    name = title_match.group(1).strip() if title_match else brief[:40].strip().title()

    # Extract tokens from CSS variables
    tokens = _extract_tokens_from_html(raw_html)

    # Process image placeholders
    raw_html = await _process_html_image_placeholders(raw_html)

    # Convert to Figma JSON
    figma_json = html_to_figma(raw_html, name, tokens)

    return {
        "html": raw_html,
        "figma_json": figma_json,
        "tokens": tokens,
        "name": name,
    }


async def edit_design(html: str, instruction: str) -> Dict[str, Any]:
    """
    Edit existing HTML design.
    Returns dict with html, figma_json, tokens, narration.
    """
    user_msg = f"Current HTML:\n{html[:6000]}\n\nInstruction: {instruction}"
    raw_html = await _llm(EDIT_SYSTEM, user_msg, max_tokens=80000)

    raw_html = re.sub(r'^```(?:html)?\n?', '', raw_html.strip())
    raw_html = re.sub(r'\n?```$', '', raw_html.strip())

    idx = raw_html.find('<!DOCTYPE')
    if idx == -1:
        idx = raw_html.find('<html')
    if idx >= 0:
        raw_html = raw_html[idx:]

    # Process any new image placeholders
    raw_html = await _process_html_image_placeholders(raw_html)

    title_match = re.search(r'<title>(.*?)</title>', raw_html, re.IGNORECASE)
    name = title_match.group(1).strip() if title_match else "Design"

    tokens = _extract_tokens_from_html(raw_html)
    figma_json = html_to_figma(raw_html, name, tokens)

    return {
        "html": raw_html,
        "figma_json": figma_json,
        "tokens": tokens,
        "narration": f"Updated: {instruction[:100]}",
    }


# ---------------------------------------------------------------------------
# 2. IMAGE GENERATION — process data-generate placeholders
# ---------------------------------------------------------------------------

async def _generate_one_image(prompt: str) -> Optional[str]:
    """Generate one image, return as base64 data URL or None."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        payload = {
            "model": os.getenv("IMAGE_MODEL", "black-forest-labs/flux.2-pro"),
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": "Gorilla Design",
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"Image gen failed {resp.status_code}: {resp.text[:200]}")
            return None

        result = resp.json()
        choices = result.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})

        # Extract URL from response
        url = None
        for img in msg.get("images", []):
            url = img.get("image_url", {}).get("url") or img.get("url", "")
            if url:
                break
        if not url and msg.get("content", "").startswith("data:image"):
            return msg["content"]

        if not url:
            return None

        # If URL, fetch and convert to base64
        if url.startswith("http"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                img_resp = await client.get(url)
            if img_resp.status_code == 200:
                b64 = base64.b64encode(img_resp.content).decode()
                return f"data:image/jpeg;base64,{b64}"
            return None

        return url if url.startswith("data:") else f"data:image/jpeg;base64,{url}"

    except Exception as e:
        print(f"Image gen error: {e}")
        return None


async def _process_html_image_placeholders(html: str) -> str:
    """Find all <img data-generate="..."> tags and replace with real images."""
    pattern = re.compile(r'<img([^>]*?)data-generate="([^"]+)"([^>]*?)>', re.IGNORECASE)
    matches = list(pattern.finditer(html))
    if not matches:
        return html

    print(f"🖼️ Found {len(matches)} image placeholders, generating in parallel...")

    # Generate all in parallel
    tasks = [_generate_one_image(m.group(2)) for m in matches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Replace in reverse order to preserve positions
    for match, result in zip(reversed(matches), reversed(results)):
        if isinstance(result, str) and result.startswith("data:"):
            before = match.group(1)
            after = match.group(3)
            replacement = f'<img{before}src="{result}"{after}>'
            html = html[:match.start()] + replacement + html[match.end():]
        # If failed, leave as placeholder with a grey background
        else:
            before = match.group(1)
            after = match.group(3)
            style_add = 'style="background:#1a1a2e;display:block;"'
            replacement = f'<img{before}{style_add}{after}>'
            html = html[:match.start()] + replacement + html[match.end():]

    return html


# ---------------------------------------------------------------------------
# 3. HTML → FIGMA JSON
# ---------------------------------------------------------------------------

def html_to_figma(html: str, name: str = "Design", tokens: dict = None) -> dict:
    """
    Convert HTML to Figma JSON format.
    Uses CSS parsing to extract layout and styling.
    """
    tokens = tokens or {}

    # Extract CSS
    css_text = ""
    style_match = re.search(r'<style[^>]*>(.*?)</style>', html, re.DOTALL | re.IGNORECASE)
    if style_match:
        css_text = style_match.group(1)

    # Parse CSS rules
    css_rules = _parse_css(css_text)

    # Build Figma node tree from HTML structure
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
    body_html = body_match.group(1) if body_match else html

    # Get background color
    bg_color = tokens.get("colors", {}).get("background", "#0D0D14")
    bg_fills = [_hex_to_fill(bg_color)]

    # Build children from major sections
    children = _parse_html_to_nodes(body_html, css_rules, 0, 0)

    # Calculate total height
    total_height = 900
    for child in children:
        bb = child.get("absoluteBoundingBox", {})
        bottom = (bb.get("y") or 0) + (bb.get("height") or 0)
        if bottom > total_height:
            total_height = bottom + 40

    figma_json = {
        "id": "frame:0",
        "name": name,
        "type": "FRAME",
        "width": 1440,
        "height": total_height,
        "fills": bg_fills,
        "_gorilla_tokens": tokens,
        "absoluteBoundingBox": {"x": 0, "y": 0, "width": 1440, "height": total_height},
        "children": children,
    }

    return figma_json


def _parse_css(css_text: str) -> Dict[str, Dict[str, str]]:
    """Parse CSS into a dict of selector -> properties."""
    rules = {}
    # Remove comments
    css_text = re.sub(r'/\*.*?\*/', '', css_text, flags=re.DOTALL)
    # Parse :root variables
    root_match = re.search(r':root\s*\{([^}]+)\}', css_text)
    root_vars = {}
    if root_match:
        for line in root_match.group(1).split(';'):
            m = re.match(r'\s*(--[\w-]+)\s*:\s*(.+)', line.strip())
            if m:
                root_vars[m.group(1).strip()] = m.group(2).strip()

    # Parse all rules
    for match in re.finditer(r'([^{]+)\{([^}]+)\}', css_text):
        selector = match.group(1).strip()
        props_text = match.group(2)
        if selector.startswith('@') or selector == ':root':
            continue
        props = {}
        for prop in props_text.split(';'):
            prop = prop.strip()
            if ':' in prop:
                k, _, v = prop.partition(':')
                k = k.strip()
                v = v.strip()
                # Resolve CSS variables
                v = re.sub(r'var\((--[\w-]+)(?:,([^)]+))?\)',
                           lambda m: root_vars.get(m.group(1), m.group(2) or '').strip(), v)
                props[k] = v
        if props:
            rules[selector] = props
    return rules


def _parse_html_to_nodes(html: str, css_rules: dict, offset_x: int, offset_y: int) -> List[dict]:
    """Parse HTML elements into Figma nodes using inline styles."""
    nodes = []
    node_id_counter = [0]

    def make_id(prefix="node"):
        node_id_counter[0] += 1
        return f"{prefix}:{node_id_counter[0]}"

    def parse_color(color_str: str) -> Optional[dict]:
        if not color_str or color_str in ('transparent', 'none', 'inherit'):
            return None
        color_str = color_str.strip()
        # hex
        hex_match = re.match(r'#([0-9a-fA-F]{3,8})', color_str)
        if hex_match:
            h = hex_match.group(1)
            if len(h) == 3:
                h = h[0]*2 + h[1]*2 + h[2]*2
            if len(h) >= 6:
                r = int(h[0:2], 16) / 255
                g = int(h[2:4], 16) / 255
                b = int(h[4:6], 16) / 255
                a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
                return {"r": r, "g": g, "b": b, "a": a}
        # rgb/rgba
        rgb_match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)', color_str)
        if rgb_match:
            return {
                "r": int(rgb_match.group(1)) / 255,
                "g": int(rgb_match.group(2)) / 255,
                "b": int(rgb_match.group(3)) / 255,
                "a": float(rgb_match.group(4)) if rgb_match.group(4) else 1.0,
            }
        return None

    def px(val: str, default: float = 0) -> float:
        if not val:
            return default
        val = val.strip()
        m = re.match(r'([\d.]+)px', val)
        if m:
            return float(m.group(1))
        m = re.match(r'([\d.]+)', val)
        if m:
            return float(m.group(1))
        return default

    def style_to_node(tag: str, attrs: dict, inner_text: str, children_nodes: list,
                       x: float, y: float, w: float, h: float) -> Optional[dict]:
        style_str = attrs.get('style', '')
        style = {}
        for part in style_str.split(';'):
            if ':' in part:
                k, _, v = part.partition(':')
                style[k.strip().lower()] = v.strip()

        node_id = attrs.get('data-node-id', make_id(tag))
        name = attrs.get('data-node-id', tag).replace('-', ' ').title()

        # Position from style
        nx = px(style.get('left'), x)
        ny = px(style.get('top'), y)
        nw = px(style.get('width'), w)
        nh = px(style.get('height'), h)

        fills = []
        bg = style.get('background') or style.get('background-color', '')
        color = parse_color(bg)
        if color:
            fills = [{"type": "SOLID", "color": color, "opacity": color.get('a', 1)}]

        # Check for background-image (already embedded base64)
        bg_img = style.get('background-image', '')
        if 'url(' in bg_img:
            url_match = re.search(r'url\(["\']?(data:[^"\']+)["\']?\)', bg_img)
            if url_match:
                fills = [{"type": "IMAGE", "scaleMode": "FILL", "imageRef": url_match.group(1), "opacity": 1}]

        # Text node
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'a', 'label', 'li') or (not children_nodes and inner_text.strip()):
            text_color = parse_color(style.get('color', ''))
            text_fills = [{"type": "SOLID", "color": text_color, "opacity": 1}] if text_color else [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1, "a": 1}, "opacity": 1}]
            font_size = px(style.get('font-size'), 16)
            font_weight_str = style.get('font-weight', '400')
            try:
                font_weight = int(font_weight_str) if font_weight_str.isdigit() else (700 if font_weight_str in ('bold','bolder') else 400)
            except:
                font_weight = 400

            return {
                "id": node_id,
                "name": name,
                "type": "TEXT",
                "absoluteBoundingBox": {"x": nx, "y": ny, "width": nw or 200, "height": nh or font_size * 1.4},
                "characters": inner_text.strip(),
                "fills": text_fills,
                "style": {
                    "fontFamily": style.get('font-family', 'DM Sans').split(',')[0].strip().strip('"\''),
                    "fontWeight": font_weight,
                    "fontSize": font_size,
                    "letterSpacing": px(style.get('letter-spacing'), 0),
                    "lineHeightPx": px(style.get('line-height'), font_size * 1.4),
                    "textAlignHorizontal": style.get('text-align', 'LEFT').upper(),
                },
            }

        # Image node
        if tag == 'img':
            src = attrs.get('src', '')
            img_fills = []
            if src.startswith('data:'):
                img_fills = [{"type": "IMAGE", "scaleMode": "FILL", "imageRef": src, "opacity": 1}]
            return {
                "id": node_id,
                "name": name or "Image",
                "type": "FRAME",
                "absoluteBoundingBox": {"x": nx, "y": ny, "width": nw or 400, "height": nh or 300},
                "fills": img_fills,
                "children": [],
            }

        # Frame/container node
        node = {
            "id": node_id,
            "name": name,
            "type": "FRAME",
            "absoluteBoundingBox": {"x": nx, "y": ny, "width": nw or 1440, "height": nh or 100},
            "fills": fills,
            "children": children_nodes,
        }

        # Corner radius
        br = style.get('border-radius', '')
        if br:
            node["cornerRadius"] = px(br, 0)

        # Stroke
        border = style.get('border', '')
        if border:
            bw_match = re.search(r'([\d.]+)px', border)
            bc_match = re.search(r'(#[0-9a-fA-F]+|rgba?\([^)]+\))', border)
            if bw_match:
                node["strokeWeight"] = float(bw_match.group(1))
                bc = parse_color(bc_match.group(1)) if bc_match else {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1}
                node["strokes"] = [{"type": "SOLID", "color": bc, "opacity": 1}]

        return node

    # Simple HTML element parser
    class NodeParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.nodes = []
            self.text_buffer = []
            self.y_cursor = offset_y

        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'head', 'meta', 'link', 'title'):
                self.stack.append({'tag': '__skip__', 'attrs': {}, 'children': [], 'text': ''})
                return
            attr_dict = dict(attrs)
            self.stack.append({'tag': tag, 'attrs': attr_dict, 'children': [], 'text': ''})

        def handle_endtag(self, tag):
            if not self.stack:
                return
            top = self.stack.pop()
            if top['tag'] == '__skip__':
                return

            # Build the node
            style_str = top['attrs'].get('style', '')
            style = {}
            for part in style_str.split(';'):
                if ':' in part:
                    k, _, v = part.partition(':')
                    style[k.strip().lower()] = v.strip()

            # Determine position and size
            nx = px(style.get('left'), offset_x)
            ny = px(style.get('top'), self.y_cursor)
            nw = px(style.get('width'), 1440)
            nh = px(style.get('height'), 100)

            node = style_to_node(
                top['tag'], top['attrs'], top['text'],
                top['children'], nx, ny, nw, nh
            )

            if node:
                # Update y_cursor for flow layout (sections stacking vertically)
                bb = node.get('absoluteBoundingBox', {})
                bottom = bb.get('y', 0) + bb.get('height', 0)
                if bottom > self.y_cursor:
                    self.y_cursor = bottom

                if self.stack:
                    self.stack[-1]['children'].append(node)
                else:
                    self.nodes.append(node)

        def handle_data(self, data):
            if self.stack and data.strip():
                self.stack[-1]['text'] += data

    parser = NodeParser()
    try:
        parser.feed(html)
    except Exception as e:
        print(f"HTML parse warning: {e}")

    return parser.nodes or []


def _hex_to_fill(hex_color: str) -> dict:
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        r = int(h[0:2], 16) / 255
        g = int(h[2:4], 16) / 255
        b = int(h[4:6], 16) / 255
    except:
        r, g, b = 0.05, 0.05, 0.08
    return {"type": "SOLID", "color": {"r": r, "g": g, "b": b, "a": 1}, "opacity": 1}


def _extract_tokens_from_html(html: str) -> dict:
    """Extract design tokens from CSS variables in HTML."""
    tokens = {"colors": {}, "typography": {}}

    # Extract :root variables
    root_match = re.search(r':root\s*\{([^}]+)\}', html, re.DOTALL)
    if root_match:
        for line in root_match.group(1).split(';'):
            m = re.match(r'\s*(--[\w-]+)\s*:\s*(.+)', line.strip())
            if m:
                key = m.group(1).strip().lstrip('-').lstrip('-')
                val = m.group(2).strip()
                if re.match(r'#[0-9a-fA-F]', val) or val.startswith('rgb'):
                    tokens["colors"][key] = val

    # Extract font imports
    font_imports = re.findall(r"@import url\('https://fonts.googleapis.com/css2\?family=([^&']+)", html)
    if font_imports:
        font = font_imports[0].replace('+', ' ').split(':')[0]
        tokens["typography"]["fontFamily"] = f"{font}, sans-serif"

    # Extract body font
    body_match = re.search(r'body\s*\{[^}]*font-family\s*:\s*([^;}]+)', html)
    if body_match:
        tokens["typography"]["fontFamily"] = body_match.group(1).strip().strip('"\'')

    return tokens


# ---------------------------------------------------------------------------
# 4. FIGMA CLIPBOARD FORMAT — paste directly into Figma
# ---------------------------------------------------------------------------

def to_figma_clipboard(figma_json: dict) -> str:
    """
    Convert our design to the exact HTML string Figma expects in clipboard.
    
    Figma reads clipboard as text/html and looks for:
    <meta charset="utf-8"><span data-metadata="<!--(figma)BASE64_JSON(figma)-->"></span>
    
    The JSON inside is base64-encoded and contains the node tree.
    This is the ONLY format Figma accepts for paste — raw JSON doesn't work.
    """
    def _convert_node(node: dict, parent_x: float = 0, parent_y: float = 0) -> dict:
        bb = node.get("absoluteBoundingBox", {})
        x = round((bb.get("x") or 0) - parent_x, 2)
        y = round((bb.get("y") or 0) - parent_y, 2)
        w = round(bb.get("width") or 100, 2)
        h = round(bb.get("height") or 40, 2)
        node_type = node.get("type", "FRAME")
        nid = "I" + str(uuid.uuid4().int)[:18]

        def _fill(f):
            if f.get("type") == "SOLID":
                c = f.get("color") or {}
                return {
                    "blendMode": "NORMAL",
                    "type": "SOLID",
                    "color": {
                        "r": round(float(c.get("r", 0)), 4),
                        "g": round(float(c.get("g", 0)), 4),
                        "b": round(float(c.get("b", 0)), 4),
                        "a": round(float(c.get("a", 1)), 4),
                    },
                    "opacity": round(float(f.get("opacity", 1)), 4),
                }
            return None

        fills = [_fill(f) for f in (node.get("fills") or []) if _fill(f)]

        base = {
            "id": nid,
            "name": node.get("name", "Node"),
            "type": node_type,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "visible": True,
            "locked": False,
            "opacity": round(float(node.get("opacity", 1)), 4),
            "blendMode": "PASS_THROUGH",
            "isMask": False,
            "effects": [],
            "fills": fills,
            "strokes": [],
            "strokeWeight": node.get("strokeWeight", 0),
            "strokeAlign": "INSIDE",
            "strokeCap": "NONE",
            "strokeJoin": "MITER",
            "dashPattern": [],
            "exportSettings": [],
        }

        if node.get("strokes"):
            for s in node["strokes"]:
                sf = _fill(s)
                if sf:
                    base["strokes"].append(sf)

        if node_type == "TEXT":
            st = node.get("style") or {}
            fs = st.get("fontSize", 16)
            fw = st.get("fontWeight", 400)
            ff = st.get("fontFamily", "DM Sans")
            lh = st.get("lineHeightPx", fs * 1.4)
            base.update({
                "characters": node.get("characters", ""),
                "style": {
                    "fontFamily": ff,
                    "fontPostScriptName": ff.replace(" ", "") + "-Regular",
                    "paragraphSpacing": 0,
                    "paragraphIndent": 0,
                    "listSpacing": 0,
                    "italic": False,
                    "fontWeight": fw,
                    "fontSize": fs,
                    "textAlignHorizontal": st.get("textAlignHorizontal", "LEFT"),
                    "textAlignVertical": "TOP",
                    "textAutoResize": "HEIGHT",
                    "textDecoration": "NONE",
                    "textDecorationColor": {"r": 0, "g": 0, "b": 0, "a": 1},
                    "letterSpacing": st.get("letterSpacing", 0),
                    "lineHeightPx": round(lh, 2),
                    "lineHeightPercent": 100,
                    "lineHeightPercentFontSize": round(lh / fs * 100, 2) if fs else 140,
                    "lineHeightUnit": "PIXELS",
                    "textCase": "ORIGINAL",
                    "textDecorationStyle": "SOLID",
                    "fills": fills,
                },
                "layoutVersion": 4,
                "characterStyleOverrides": [],
                "styleOverrideTable": {},
                "lineIndentations": [],
                "lineTypes": [],
            })
        else:
            cr = node.get("cornerRadius", 0)
            base.update({
                "cornerRadius": cr,
                "rectangleCornerRadii": [cr, cr, cr, cr],
                "clipsContent": True,
                "background": fills,
                "backgroundColor": fills[0]["color"] if fills else {"r": 0, "g": 0, "b": 0, "a": 0},
                "layoutMode": "NONE",
                "layoutWrap": "NO_WRAP",
                "counterAxisSizingMode": "FIXED",
                "primaryAxisSizingMode": "FIXED",
                "primaryAxisAlignItems": "MIN",
                "counterAxisAlignItems": "MIN",
                "paddingLeft": 0,
                "paddingRight": 0,
                "paddingTop": 0,
                "paddingBottom": 0,
                "itemSpacing": 0,
                "overflowDirection": "NONE",
                "numberOfFixedChildren": 0,
                "overlayPositionType": "CENTER",
                "overlayBackground": {"type": "NONE"},
                "overlayBackgroundInteraction": "NONE",
                "constraints": {"vertical": "TOP", "horizontal": "LEFT"},
            })
            abs_x = bb.get("x") or 0
            abs_y = bb.get("y") or 0
            base["children"] = [
                _convert_node(child, abs_x, abs_y)
                for child in (node.get("children") or [])
            ]

        return base

    # Convert tree
    bb = figma_json.get("absoluteBoundingBox", {})
    top_node = _convert_node(figma_json, bb.get("x", 0), bb.get("y", 0))
    top_node["x"] = 0
    top_node["y"] = 0

    payload = {
        "fileKey": "",
        "pasteID": uuid.uuid4().int % (10 ** 15),
        "dataType": "scene",
        "nodes": [top_node],
    }

    # Encode as base64 and wrap in Figma's clipboard HTML format
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_b64 = base64.b64encode(payload_json.encode()).decode()
    
    clipboard_html = f'<meta charset="utf-8"><span data-metadata="<!--(figma){payload_b64}(figma)-->"></span>'
    return clipboard_html


# ---------------------------------------------------------------------------
# 5. SERVE — just return the HTML directly
# ---------------------------------------------------------------------------

def get_hosted_html(html: str, design_id: str = "", site_url: str = "") -> str:
    """Return the HTML with an optional edit badge."""
    if not design_id:
        return html
    badge = f"""<a href="{site_url}/design/editor/{design_id}"
style="position:fixed;bottom:20px;right:20px;background:#0066ff;color:#fff;
padding:8px 16px;border-radius:8px;font-family:DM Sans,sans-serif;font-size:13px;
text-decoration:none;font-weight:500;box-shadow:0 4px 16px rgba(0,102,255,0.4);z-index:9999;">
✦ Edit in Gorilla Design</a>"""
    return html.replace('</body>', f'{badge}\n</body>')


def generate_slug(name: str) -> str:
    import re as _re
    slug = _re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'design'
    return f"{slug}-{str(uuid.uuid4())[:8]}"


# ---------------------------------------------------------------------------
# Legacy compat — these are called by old routes
# ---------------------------------------------------------------------------

def design_to_html(tree: dict) -> str:
    """Compat: if we have stored HTML use it, else render from JSON."""
    # If stored as raw HTML in tree (new format)
    if tree.get("_raw_html"):
        return tree["_raw_html"]
    # Fallback: basic render from figma json
    return _figma_json_to_html_fallback(tree)


def _figma_json_to_html_fallback(tree: dict) -> str:
    """Minimal fallback renderer from Figma JSON."""
    tokens = tree.get("_gorilla_tokens", {})
    colors = tokens.get("colors", {})
    bg = colors.get("background", "#0D0D14")
    name = tree.get("name", "Design")
    w = tree.get("width", 1440)
    h = tree.get("height", 900)

    def node_html(node, px_off, py_off):
        bb = node.get("absoluteBoundingBox", {})
        x = (bb.get("x") or 0) - px_off
        y = (bb.get("y") or 0) - py_off
        nw = bb.get("width") or 0
        nh = bb.get("height") or 0
        style = f"left:{x}px;top:{y}px;width:{nw}px;height:{nh}px;"
        fills = node.get("fills", [])
        nt = node.get("type", "")
        if nt != "TEXT":
            fill = next((f for f in fills if f.get("visible", True) and f.get("type") == "SOLID"), None)
            if fill:
                c = fill.get("color", {})
                style += f"background:rgba({int(c.get('r',0)*255)},{int(c.get('g',0)*255)},{int(c.get('b',0)*255)},{fill.get('opacity',1)});"
            if node.get("cornerRadius"):
                style += f"border-radius:{node['cornerRadius']}px;"
        if nt == "TEXT":
            st = node.get("style", {})
            color = fills[0].get("color", {}) if fills else {}
            style += f"background:transparent;font-size:{st.get('fontSize',16)}px;font-weight:{st.get('fontWeight',400)};color:rgba({int(color.get('r',1)*255)},{int(color.get('g',1)*255)},{int(color.get('b',1)*255)},1);font-family:'{st.get('fontFamily','DM Sans')}',sans-serif;overflow:hidden;white-space:pre-wrap;"
            chars = node.get("characters", "")
            return f'<div class="gd-node" style="{style}">{chars}</div>'
        children_html = "".join(node_html(c, bb.get("x",0), bb.get("y",0)) for c in node.get("children", []))
        return f'<div class="gd-node" style="{style}">{children_html}</div>'

    body = "".join(node_html(c, 0, 0) for c in tree.get("children", []))
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{name}</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:{bg};}}
.gd-frame{{position:relative;width:{w}px;min-height:{h}px;background:{bg};margin:0 auto;overflow:hidden;}}
.gd-node{{position:absolute;box-sizing:border-box;}}</style></head>
<body><div class="gd-frame">{body}</div></body></html>"""