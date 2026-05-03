"""
Gorilla Design — Static Page Hosting
======================================
Converts a design (Figma JSON + tokens) to a hosted static HTML page
served at /design/hosted/{slug}
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from .figma import figma_to_html
from .tokens import tokens_to_css_vars


def generate_slug(name: str) -> str:
    """Generate a URL-safe slug from a design name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        slug = "design"
    # Append short UUID for uniqueness
    short_id = str(uuid.uuid4())[:8]
    return f"{slug}-{short_id}"


def render_hosted_page(
    node_tree: dict,
    tokens: dict,
    design_name: str,
    include_edit_link: bool = False,
    design_id: Optional[str] = None,
) -> str:
    """
    Render a complete self-contained HTML page from a design.
    This is what gets stored in designs.hosted_html and served publicly.
    """
    # Generate the base HTML from figma renderer
    html = figma_to_html(node_tree, tokens)

    # Inject CSS vars from tokens
    css_vars = tokens_to_css_vars(tokens)

    # Optionally inject an edit link badge
    edit_badge = ""
    if include_edit_link and design_id:
        edit_badge = f"""
<a href="/design/editor/{design_id}"
   style="position:fixed;bottom:20px;right:20px;
          background:#0066ff;color:#fff;
          padding:8px 16px;border-radius:8px;
          font-family:DM Sans,sans-serif;font-size:13px;
          text-decoration:none;font-weight:500;
          box-shadow:0 4px 16px rgba(0,102,255,0.4);
          z-index:9999;">
  Edit in Gorilla Design
</a>"""

    # Inject tokens CSS vars and edit badge into the HTML
    html = html.replace(
        "</style>",
        f"\n/* Gorilla Design tokens */\n{css_vars}\n</style>",
    )
    html = html.replace("</body>", f"{edit_badge}\n</body>")

    return html


def get_hosted_url(slug: str, base_url: str = "") -> str:
    """Get the public URL for a hosted design page."""
    base = base_url.rstrip("/") or ""
    return f"{base}/design/hosted/{slug}"