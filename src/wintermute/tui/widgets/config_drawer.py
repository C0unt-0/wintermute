"""config_drawer.py — Reusable collapsible side panel with typed form fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field as dc_field

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Select, Static, Switch

from wintermute.tui import theme


@dataclass
class FieldDef:
    """Definition of a single configuration field."""

    name: str
    label: str
    default: str
    field_type: str = "str"  # "str", "int", "float", "select", "switch"
    options: list[str] = dc_field(default_factory=list)


class ConfigDrawer(Vertical):
    """Collapsible side panel containing form fields for configuring operations."""

    DEFAULT_CSS = f"""
    ConfigDrawer {{
        display: none;
        width: 36;
        background: {theme.BG_PANEL};
        border-left: solid {theme.BORDER};
        padding: 1 2;
    }}

    ConfigDrawer.visible {{
        display: block;
    }}

    ConfigDrawer .drawer-title {{
        color: {theme.TEXT_BRIGHT};
        text-style: bold;
        padding: 0 0 1 0;
    }}

    ConfigDrawer Label {{
        color: {theme.TEXT_MUTED};
        padding: 1 0 0 0;
    }}

    ConfigDrawer.locked .drawer-title {{
        color: {theme.AMBER};
    }}
    """

    locked: reactive[bool] = reactive(False)

    def __init__(
        self,
        fields: Sequence[FieldDef] | None = None,
        title: str = "Configuration",
        start_label: str = "Start",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._fields: list[FieldDef] = list(fields) if fields else []
        self._title = title
        self._start_label = start_label

    def compose(self):
        """Yield title, form fields, and start button."""
        yield Static(self._title, classes="drawer-title")
        yield from self._build_fields()
        yield Button(self._start_label, id="drawer-start")

    def _build_fields(self):
        """Yield Label + widget pairs for each field definition."""
        for f in self._fields:
            yield Label(f.label)
            if f.field_type == "select":
                options = [(opt, opt) for opt in f.options]
                yield Select(
                    options,
                    value=f.default,
                    allow_blank=False,
                    id=f"cfg-{f.name}",
                )
            elif f.field_type == "switch":
                yield Switch(
                    value=f.default.lower() in ("on", "true", "1", "yes"),
                    id=f"cfg-{f.name}",
                )
            else:
                # str, int, float — all use Input
                yield Input(
                    value=f.default,
                    placeholder=f.label,
                    id=f"cfg-{f.name}",
                )

    def get_values(self) -> dict[str, str]:
        """Return current form values. Falls back to defaults if not mounted."""
        values: dict[str, str] = {}
        for f in self._fields:
            widget_id = f"cfg-{f.name}"
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                # Not mounted yet — use default
                values[f.name] = f.default
                continue

            if isinstance(widget, Switch):
                values[f.name] = "on" if widget.value else "off"
            elif isinstance(widget, Select):
                values[f.name] = (
                    str(widget.value) if widget.value is not Select.BLANK else f.default
                )
            elif isinstance(widget, Input):
                values[f.name] = widget.value
            else:
                values[f.name] = f.default
        return values

    def toggle(self) -> None:
        """Show or hide the drawer by toggling the 'visible' CSS class.

        No-op if locked.
        """
        if self.locked:
            return
        self.toggle_class("visible")

    def lock(self) -> None:
        """Disable all form widgets and the start button."""
        self.locked = True
        self.add_class("locked")
        self._set_form_disabled(True)

    def unlock(self) -> None:
        """Re-enable all form widgets and the start button."""
        self.locked = False
        self.remove_class("locked")
        self._set_form_disabled(False)

    def _set_form_disabled(self, disabled: bool) -> None:
        """Toggle disabled state on all form inputs and button."""
        for f in self._fields:
            try:
                widget = self.query_one(f"#cfg-{f.name}")
                widget.disabled = disabled
            except Exception:
                pass
        try:
            self.query_one("#drawer-start", Button).disabled = disabled
        except Exception:
            pass

    def set_fields(self, fields: Sequence[FieldDef]) -> None:
        """Replace fields and rebuild the form (for polymorphic forms)."""
        self._fields = list(fields)
        # Remove all children and recompose
        self.remove_children()
        self.mount(Static(self._title, classes="drawer-title"))
        for widget in self._build_fields():
            self.mount(widget)
        self.mount(Button(self._start_label, id="drawer-start"))
