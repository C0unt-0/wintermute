"""
theme.py — Wintermute TUI design system.
Color tokens and master Textual CSS stylesheet.
"""

# ── Color tokens ──
BG = "#0a0e14"
BG_PANEL = "#0d1117"
BG_CARD = "#131820"
BG_HOVER = "#1a2030"
BORDER = "#1e2a3a"
BORDER_ACTIVE = "#00e5ff"

TEXT = "#c5cdd8"
TEXT_MUTED = "#5a6577"
TEXT_BRIGHT = "#e8ecf1"

CYAN = "#00e5ff"
GREEN = "#00ff9f"
RED = "#ff3366"
AMBER = "#ffb300"
PURPLE = "#b388ff"

SAFE = GREEN
MALICIOUS = RED
ATTACKER = RED
DEFENDER = CYAN

# ── Master TCSS stylesheet ──
STYLESHEET = f"""
Screen {{
    background: {BG};
}}

.panel {{
    background: {BG_CARD};
    border: solid {BORDER};
    padding: 1 2;
}}

TabbedContent {{
    background: {BG};
}}

TabPane {{
    padding: 1;
}}

Tabs {{
    background: {BG_PANEL};
    border-bottom: solid {BORDER};
}}

Tab {{
    color: {TEXT_MUTED};
    background: {BG_PANEL};
    padding: 1 2;
}}

Tab.-active {{
    color: {TEXT_BRIGHT};
    border-bottom: tall {CYAN};
}}

Tab:hover {{
    color: {TEXT_BRIGHT};
}}

Footer {{
    background: {BG_PANEL};
    color: {TEXT_MUTED};
    border-top: solid {BORDER};
}}

Header {{
    background: {BG_PANEL};
    color: {TEXT_BRIGHT};
}}

DataTable {{
    background: {BG_CARD};
    border: solid {BORDER};
}}

DataTable > .datatable--header {{
    color: {TEXT_MUTED};
    text-style: bold;
    background: {BG_CARD};
}}

DataTable > .datatable--cursor {{
    background: {BG_HOVER};
    color: {TEXT_BRIGHT};
}}

RichLog {{
    background: {BG_CARD};
    border: solid {BORDER};
    padding: 0 1;
}}

Sparkline {{
    margin: 0 1;
}}

ProgressBar {{
    padding: 0 1;
}}

Bar > .bar--bar {{
    color: {CYAN};
    background: {BORDER};
}}

Bar > .bar--complete {{
    color: {GREEN};
}}

Input {{
    background: {BG_CARD};
    border: solid {BORDER};
    color: {TEXT_BRIGHT};
}}

Input:focus {{
    border: solid {BORDER_ACTIVE};
}}

Button {{
    background: {CYAN};
    color: {BG};
    border: none;
    text-style: bold;
}}

Button:hover {{
    background: {GREEN};
}}

.row {{
    layout: horizontal;
    height: auto;
}}

.flex-1 {{
    width: 1fr;
}}
"""
