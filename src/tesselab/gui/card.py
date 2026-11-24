"""
An analysis in the window: a card with a header (label, comment, key results, status dot, buttons) and one group per
node instance. The views are the card's sections (heading with status, their plots in a grid, their plot settings);
the readers and compute nodes follow in a folded "Details" area (heading with status and key results; opened, their
parameters), or without any view, open as the card's groups. A view switched off in the graph is not shown. Nodes are
added and connected in the graph window (graph_view.py), through this card's methods.

How a change flows: every edit goes through AnalysisCard.edit(), which keeps an undo step (the analysis content
before), marks the card unsaved and emits changed; the window then asks the runner for the new state, and when
results arrive it calls update_results(), which updates each group (status, summary, messages) and draws the plots
of open groups whose result (fingerprint) or plot settings changed. Interactions (cursors, ranges, picks) write into
parameters through set_param(), the same way as typing does, so they are undoable and saved.

An analysis can be open in several tabs: its cards show the same in-memory analysis and share one undo history and
one saved state (Shared); an edit in one card appears in all of them. Which groups are open is each card's own.

With a shared graph file (graph.py) an edit that changes the graph reaches the other open analyses using it
(Window.graph_changed). An undo step remembers the graph before and after its edit: undoing it puts the graph back
only if no other analysis changed the graph since; otherwise only this analysis's own values go back.
"""
import json
import os
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import params as P
from tesselab.analysis import CycleError, Source
from tesselab.engine import Key, node_kind
from tesselab.graph import clone, is_var
from tesselab.gui import theme
from tesselab.gui.forms import ParamForm
from tesselab.gui.plots import CursorItem, PickHandler, PlotView, RangeItem
from tesselab.gui.widgets import CardTop, ElidedLabel, PlotTile, ReflowGrid, message_box
from tesselab.node import Crosshair, Cursor, Pick, RangeSelect, Toggle
from tesselab.plotspec import note_only
from tesselab.signature import bindings


def format_value(value, error=None, unit="", display=None, digits=4) -> str:
    from tesselab import units
    if value is None:
        return "–"
    shown = units.to_display(value, unit, display) if display else value
    text = f"{shown:.{digits}g}"
    if error is not None:
        shown_error = units.to_display(error, unit, display) if display else error
        text += f" ± {shown_error:.2g}"
    u = display or unit
    return f"{text} {u}".strip()


def find_values(outputs, name):
    """The values of a variable or coordinate `name` in any of the outputs (for snapping cursors)."""
    for dataset in (outputs or {}).values():
        if name in dataset.variables:
            return np.asarray(dataset[name].values, dtype=float).ravel()
    return None


class Shared:
    """What the cards of one analysis share (it can be open in several tabs): the undo history, the content at the
    last save, and the list of its cards."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.undo_steps, self.redo_steps, self.coalesce = [], [], None
        self.saved_content = analysis.state()  # what the analysis was at the last save (Analysis.state)
        self.cards = []


class NodeGroup(QtWidgets.QWidget):
    """One node instance in the card."""

    def __init__(self, card, name):
        super().__init__()
        self.card, self.name = card, name
        self.tiles, self.interactions = {}, []
        self.cls, self.problem = None, None
        try:
            self.cls = card.engine.node_class(card.analysis, name)
        except Exception as error:
            self.problem = str(error)
        self.kind = node_kind(self.cls) if self.cls is not None else "compute"
        # a reader or compute node has no plots: opening its group shows its parameters (no Settings button)
        self.has_plots = bool(self.cls is not None and self.cls.plots)
        self._drawn = None
        self.parts = []       # the key results in the header (kept while a new result is computed)
        self._running = False
        self._status_shown = self._messages_shown = None  # what the status dot and the messages show now
        self.result = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        heading = QtWidgets.QHBoxLayout()
        heading.setSpacing(4)
        # chevron, status dot, name: the dot right in front of the name, as in the card's header; chevron and name
        # open and close the group, the dot shows what went wrong
        self.chevron = QtWidgets.QToolButton(objectName="icon", autoRaise=True)
        self.chevron.setIconSize(QtCore.QSize(14, 14))
        self.chevron.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chevron.clicked.connect(lambda: self.card.set_open(self.name, not self.is_open))
        heading.addWidget(self.chevron)
        self.status = QtWidgets.QToolButton(objectName="icon", autoRaise=True)
        self.status.setIconSize(QtCore.QSize(12, 12))
        self.status.clicked.connect(self._show_status)
        heading.addWidget(self.status)
        self.toggle = QtWidgets.QPushButton(objectName="group")
        self.toggle.setFlat(True)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(lambda: self.card.set_open(self.name, not self.is_open))
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        heading.addWidget(self.toggle)
        self.summary = ElidedLabel(objectName="keyresult")
        heading.addWidget(self.summary, 1)
        self.run_button = None
        if self.cls is not None and self.cls.run_on == "button":
            self.run_button = QtWidgets.QPushButton("Run", objectName="primary",
                                                    toolTip="This node runs only when asked (it takes a while).")
            self.run_button.clicked.connect(lambda: self.card.run_node(self.name))
            heading.addWidget(self.run_button)
        self.action_buttons = {}  # action id: button (busy while a job runs)
        for action in (self.cls.actions if self.cls is not None else []):
            button = QtWidgets.QPushButton(action.label, toolTip=action.help)
            button.clicked.connect(lambda checked=False, a=action: self.card.run_action(self.name, a))
            heading.addWidget(button)
            self.action_buttons[action.id] = button
        self.settings_button = QtWidgets.QPushButton("Settings", checkable=True)
        self.settings_button.toggled.connect(self._settings_toggled)
        self.settings_button.setVisible(self.has_plots and bool(self.cls.params or self.cls.inputs))
        heading.addWidget(self.settings_button)
        more = QtWidgets.QToolButton(text="⋯", objectName="icon", autoRaise=True, toolTip="More")
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(more)
        menu.addAction("Rename…", self._rename)
        menu.addAction("Move up", lambda: self.card.move(self.name, -1))
        menu.addAction("Move down", lambda: self.card.move(self.name, 1))
        menu.addSeparator()
        menu.addAction("Remove", lambda: self.card.remove(self.name))
        more.setMenu(menu)
        heading.addWidget(more)
        layout.addLayout(heading)
        self.messages = QtWidgets.QLabel(objectName="note", wordWrap=True)
        self.messages.setVisible(False)
        self.messages.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.messages)
        self.settings_box = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout(self.settings_box)
        self.settings_layout.setContentsMargins(26, 2, 4, 6)
        self.settings_box.setVisible(False)
        layout.addWidget(self.settings_box)
        self.form = None
        self.input_widgets = {}
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body.setVisible(False)
        layout.addWidget(self.body)
        self.grid = None
        self.update_heading()

    # --- heading ------------------------------------------------------------------------------------------------
    @property
    def entry(self):
        return self.card.analysis.nodes[self.name]

    @property
    def is_open(self):
        return self.name in self.card.open_groups

    def title(self):
        node_name = self.cls.name if self.cls is not None else self.entry["node"]
        label = self.entry.get("label")
        same = [n for n, e in self.card.analysis.nodes.items() if e["node"] == self.entry["node"]]
        if label:
            return f"{node_name}: {label}"
        if len(same) > 1:
            return f"{node_name} #{same.index(self.name) + 1}"
        return node_name

    def update_heading(self):
        open_ = self.name in self.card.open_groups
        self.chevron.setIcon(theme.icon("chevron_down" if open_ else "chevron_right"))
        self.toggle.setText(self.title())
        self.toggle.setToolTip((self.cls.help if self.cls else self.problem or "") + f"\n(instance '{self.name}')")

    def set_open(self, open_):
        self.update_heading()
        if not self.has_plots:  # a reader or compute node: its parameters
            self._settings_toggled(open_ and self.cls is not None)
            return
        self.body.setVisible(open_)
        if open_ and self.cls is not None:
            self._make_plots()
            self.draw()

    def _settings_toggled(self, on):
        if on and self.form is None and self.cls is not None:
            self._make_settings()
        self.settings_box.setVisible(on)

    def _rename(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename", "Label of this node (empty: its name):",
                                                  text=self.entry.get("label", ""))
        if ok:
            self.card.edit(lambda: self.entry.__setitem__("label", text.strip()) if text.strip()
                           else self.entry.pop("label", None))
            self.update_heading()

    def _show_status(self):
        result = self.result
        lines = []
        if self.problem:
            lines.append(self.problem)
        if result is not None:
            lines.append(f"{result.status}" + (f", {result.seconds * 1000:.0f} ms" if result.seconds else "")
                         + (f" (from the {result.cached} cache)" if result.cached else ""))
            for d in result.diagnostics:
                lines.append(f"{d['level']}: {d['message']}")
                if d.get("details"):
                    lines.append(d["details"])
        if self.name in self.card.runner.is_running(self.card.analysis):
            lines.insert(0, "running …")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(self.title())
        box.setText("\n".join(lines[:4]) or "No result yet.")
        if len(lines) > 4:
            box.setDetailedText("\n".join(lines))
        box.exec()

    # --- settings: inputs and parameters --------------------------------------------------------------------------
    def _make_settings(self):
        card = self.card
        for input_name in self.cls.inputs:  # where the inputs come from: changed in the graph window
            label = QtWidgets.QLabel(objectName="hint", wordWrap=True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setToolTip("Connect inputs in the graph window (the graph button, Ctrl+G).")
            self.input_widgets[input_name] = label
            self.settings_layout.addWidget(label)
        if self.cls.params:
            self.form = ParamForm([p for p in self.cls.params if not p.redraw or not self._in_plot_settings(p.name)],
                                  self.entry.get("params", {}), card.analysis.vars, card.project_root,
                                  advanced=card.window.advanced)
            card.connect_form(self.form, self.name)
            self.settings_layout.addWidget(self.form)
        self.update_settings()

    def _in_plot_settings(self, name):
        return any(name in plot.settings for plot in self.cls.plots)

    def update_settings(self, only_if_changed=False):
        """The parameter form and the input lines show the current values. only_if_changed: when results arrive
        (nothing was edited): fields whose values are the same as shown last are left alone."""
        if self.form is not None:
            self.card.update_form(self.form, self.name, only_if_changed)
        for input_name, label in self.input_widgets.items():
            current = self.entry.get("inputs", {}).get(input_name)
            sources = [Source.parse(s) for s in (current if isinstance(current, list) else [current])] if current else []
            text = ", ".join(self.card.source_text(s) for s in sources) or "not connected"
            label.setText(f"{input_name} ← {text}")

    # --- plots ------------------------------------------------------------------------------------------------------
    def _make_plots(self):
        if self.grid is not None or self.cls is None:
            return
        tiles = []
        for plot in self.cls.plots:
            settings = []
            for name in plot.settings:
                p = P.find(self.cls.params, name)
                if p is None:
                    continue
                form = ParamForm([p], self.entry.get("params", {}), self.card.analysis.vars, self.card.project_root)
                form.changed.connect(lambda n, v: self.card.set_param(self.name, n, v))
                form.var_changed.connect(self.card.set_var)
                settings.append(form)
            if plot.widget is not None:
                widget = plot.widget()
                view = PlotView()  # a level 2 plot still gets a tile; the widget goes where the view would be
                tile = PlotTile(view, plot.title, plot.help, settings, plot.span, self.card.analysis.label)
                tile.layout().replaceWidget(view, widget)
                view.setVisible(False)
                tile.widget = widget
            else:
                view = PlotView(rows=plot.rows)
                tile = PlotTile(view, plot.title, plot.help, settings, plot.span, self.card.analysis.label)
                tile.widget = None
            tile.settings_forms = settings
            self.tiles[plot.id] = tile
            tiles.append(tile)
        self.grid = ReflowGrid(tiles)
        self.body_layout.addWidget(self.grid)
        self.card.renumber()
        self.card.link_group(self)

    def plot_count(self):
        return len(self.cls.plots) if self.cls is not None else 0

    def number_plots(self, first):
        if self.cls is None:
            return
        for i, plot in enumerate(self.cls.plots):
            if plot.id in self.tiles:
                self.tiles[plot.id].set_number(first + i)

    def show_result(self, result, running):
        """New results (or none yet): status, summary, messages; the plots if the group is open."""
        self.result = result
        self._running = running
        pending = result is None and running and self.shows_result()  # computing: what is shown stays (below)
        level = "running" if running else (result.level() if result else ("error" if self.problem else "waiting"))
        if result is not None and result.status == "waiting" and not running:
            level = "waiting"
        tip = {"running": "running …", "waiting": "waiting: press Run" if self.run_button else "no result yet",
               None: "fine", "info": "fine", "warning": "warnings: click for details", "error": "error: click for details"}
        timing = f" ({result.seconds * 1000:.0f} ms)" if result and result.seconds else ""
        shown = (level, tip.get(level, level) + timing, theme.current()["dark"])
        if shown != self._status_shown:  # (an icon is made anew each time: every group, at every result)
            self._status_shown = shown
            self.status.setIcon(theme.dot_icon(level))
            self.status.setToolTip(shown[1])
        if pending:
            return self.update_pending()
        self.parts = []
        if result is not None and self.cls is not None:
            for r in self.cls.key_results:
                value, error = result.key_results.get(r.name, (None, None))
                if value is not None:
                    self.parts.append(f"{r.label} = {format_value(value, error, r.unit, r.display, r.digits)}")
        if "   ".join(self.parts) != self.summary.text():
            self.summary.setText("   ".join(self.parts))
        messages = []
        if self.problem:
            messages.append(("error", self.problem))
        if result is not None:
            for d in result.diagnostics:
                if not d.get("plots") or not self.is_open:
                    if d.get("code") in self.card.analysis.doc.get("view", {}).get("muted", []):
                        continue
                    messages.append((d["level"], d["message"]))
        if messages != self._messages_shown:
            self._messages_shown = messages
            if messages:
                error = any(level == "error" for level, _ in messages)
                self.messages.setObjectName("error" if error else "note")
                self.messages.setStyleSheet("")
                self.messages.setText("\n".join(m for _, m in messages))
            self.messages.setVisible(bool(messages))
        if self.form is not None or self.input_widgets:
            self.update_settings(only_if_changed=True)
        if self.is_open:  # not isVisible(): a group made a moment ago is shown only later (it stayed "Computing")
            self.draw()

    def shows_result(self) -> bool:
        """The plots show a result (not the note "Computing …" or an error)."""
        return self._drawn is not None and self._drawn[0] is not None

    def update_pending(self):
        """A new result is on its way (λ dragged, a field edited): the status dot says so, and the plots, messages and
        header stay as they are until it arrives, as in the old viewer. Building them anew for a note and again for the
        result made dragging λ crawl (about 20 plots built per step, 2026-11-24). The cursors follow the parameters."""
        if self.form is not None or self.input_widgets:
            self.update_settings(only_if_changed=True)
        if self.is_open and self.cls is not None and self.grid is not None:
            values, _ = self.card.engine.resolve_params(self.card.analysis, self.name)
            self._update_cursors(values)

    def draw(self, force=False):
        """Draw the plots for the current result (only if it or the plot settings changed)."""
        if self.cls is None or self.grid is None:
            return
        result = self.result
        values, _ = self.card.engine.resolve_params(self.card.analysis, self.name)
        if result is None and self._running and self.shows_result() and not force:
            return self._update_cursors(values)  # a result is on its way: the plots stay (see update_pending)
        state = (result.fingerprint if result else None, result.status if result else None,
                 json.dumps({p.name: values.get(p.name) for p in self.cls.params if p.redraw}, default=str),
                 theme.current()["dark"])
        if state == self._drawn and not force:
            self._update_cursors(values)
            return
        self._drawn = state
        diagnostics = result.diagnostics if result else []
        for plot in self.cls.plots:
            tile = self.tiles[plot.id]
            for form in tile.settings_forms:
                form.set_values(self.entry.get("params", {}), self.card.analysis.vars)
            plot_messages = [d for d in diagnostics if plot.id in d.get("plots", [])]
            tile.set_message("\n".join(d["message"] for d in plot_messages),
                             any(d["level"] == "error" for d in plot_messages))
            if result is None:
                spec = note_only("Computing …")
            elif not result.ok:
                spec = note_only("No result: " + (result.error or ("waiting for Run" if result.status == "waiting"
                                                                  else "an input is missing")))
            else:
                missing = [n for n in plot.needs if not any(n in ds.variables for ds in result.outputs.values())]
                waiting = self.card.engine.missing_inputs(self.card.analysis, self.name, result)
                if waiting:
                    spec = note_only("No " + ", ".join(waiting) + " yet (see Details: where it comes from has no "
                                     "result)")
                elif missing:
                    spec = note_only("Needs " + ", ".join(missing))
                else:
                    ctx = self.card.engine.draw_context(self.card.analysis, self.name, result,
                                                        self.card.window.others(self.card, self.entry["node"])
                                                        if plot.items == "several" else ())
                    if tile.widget is not None:
                        try:
                            tile.widget.update(ctx)
                        except Exception as error:
                            tile.set_message(f"The plot failed: {error}", True)
                        continue
                    kept = tile.kept
                    if not force and kept is not None and kept[0] == theme.current()["dark"] and ctx.same_as(kept[1]):
                        spec = kept[2]  # nothing this plot read has changed (dragging λ leaves the spectra alone)
                    else:
                        try:
                            spec = plot.draw(ctx)
                            snapshot = ctx.snapshot()
                            tile.kept = (theme.current()["dark"], snapshot, spec) if snapshot is not None else None
                        except Exception as error:
                            traceback.print_exc()
                            spec = note_only(f"The plot failed: {type(error).__name__}: {error}")
                            tile.kept = None
            if tile.widget is None:
                try:
                    tile.view.set_spec(spec)
                except Exception as error:
                    traceback.print_exc()
                    tile.view.set_spec(note_only(f"The plot cannot be drawn: {error}"))
        if not self.interactions:
            self._make_interactions(values)
        self._update_cursors(values)

    def _make_interactions(self, values):
        outputs = self.result.outputs if self.result else {}
        for interaction in self.cls.interactions:
            for plot_id in interaction.plots:
                tile = self.tiles.get(plot_id)
                if tile is None or tile.widget is not None:
                    continue
                view = tile.view
                if isinstance(interaction, Cursor):
                    snap = find_values(outputs, interaction.snap) if interaction.snap else None
                    item = CursorItem(view, interaction.panel, interaction.axis, values.get(interaction.param),
                                      lambda v, final, i=interaction: self._cursor_moved(i, v, final), snap,
                                      interaction.label)
                    self.interactions.append((interaction, interaction.param, item))
                elif isinstance(interaction, Crosshair):
                    for axis, param in (("x", interaction.x), ("y", interaction.y)):
                        snap_name = interaction.snap_x if axis == "x" else interaction.snap_y
                        snap = find_values(outputs, snap_name) if snap_name else None
                        item = CursorItem(view, interaction.panel, axis, values.get(param),
                                          lambda v, final, p=param: self.card.set_param(
                                              self.name, p, v, final, coalesce=f"cursor:{self.name}:{p}"), snap)
                        self.interactions.append((interaction, param, item))
                elif isinstance(interaction, RangeSelect):
                    item = RangeItem(view, interaction.panel, values.get(interaction.param),
                                     lambda v, final, p=interaction.param: self.card.set_param(self.name, p, v, final),
                                     interaction.axis)
                    self.interactions.append((interaction, interaction.param, item))
                elif isinstance(interaction, Pick):
                    PickHandler(view, interaction.panel, lambda x, y, p=interaction.param: self._pick(p, x, y, True),
                                lambda x, y, p=interaction.param: self._pick(p, x, y, False))
                elif isinstance(interaction, Toggle):
                    view.point_clicked.connect(lambda panel, key, i=interaction: self._toggle(i, panel, key))

    def _cursor_moved(self, interaction, value, final):
        self.card.set_param(self.name, interaction.param, value, final,
                            coalesce=f"cursor:{self.name}:{interaction.param}")

    def _update_cursors(self, values):
        outputs = self.result.outputs if self.result and self.result.ok else {}
        fresh = getattr(self, "_snapped_for", None) is not self.result  # the values to snap to: once per result
        self._snapped_for = self.result
        for interaction, param, item in self.interactions:
            if fresh and hasattr(item, "snap") and isinstance(interaction, Cursor) and interaction.snap:
                snap = find_values(outputs, interaction.snap)
                if snap is not None:
                    item.snap = snap
            item.set_value(values.get(param))

    def _toggle(self, interaction, panel, key):
        """A click on a point with a key: the key into the parameter, or out of it again."""
        if interaction.panel is not None and panel != interaction.panel:
            return
        current = list(self.card.param_value(self.name, interaction.param) or [])
        self.card.set_param(self.name, interaction.param,
                            [k for k in current if k != key] if key in current else sorted(current + [key]))

    def _pick(self, param, x, y, add):
        current = self.card.param_value(self.name, param) or []
        points = [list(p) for p in current]
        if add:
            points.append([x, y])
        elif points:
            distances = [abs(px - x) + abs(py - y) for px, py in points]
            del points[int(np.argmin(distances))]
        self.card.set_param(self.name, param, points)


class AnalysisCard(QtWidgets.QFrame):
    changed = QtCore.Signal(object)                    # card: its analysis content changed (recompute)
    saved_changed = QtCore.Signal(object)              # card: unsaved / saved
    cursor_moved = QtCore.Signal(object, str, object, bool)  # card, link key, value, final
    fold_changed = QtCore.Signal(object, str, bool)    # card, node id, open
    close_requested = QtCore.Signal(object)
    _job_done = QtCore.Signal(str, object, object)     # node, action, future: a job action finished (worker thread)
    link_toggled = QtCore.Signal(object, bool)

    def __init__(self, window, analysis):
        super().__init__(objectName="card")
        self.window, self.analysis = window, analysis
        self.engine, self.runner = window.engine, window.runner
        self.project_root = window.project.root if window.project else None
        self.shared = window.shared_of(analysis)
        self.shared.cards.append(self)
        self._waiting_drag = None  # the newest step of a dragged cursor that waits for the one being computed
        self.open_groups = list(analysis.doc["view"].get("open", []))  # which groups are open: this card's own
        self.groups: dict[str, NodeGroup] = {}
        self._job_done.connect(self._finish_job, Qt.ConnectionType.QueuedConnection)
        self.setMinimumWidth(460)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        # the header stays in view when the tab is scrolled down (place_header): a widget of its own outside the
        # layout, which only keeps its place free (header_space)
        self.header = CardTop(self)
        header = self._make_header()
        header.setContentsMargins(0, 0, 0, 0)
        self.header.setLayout(header)
        self.header.ensurePolished()  # the fonts of the style sheet decide its height
        self.header_space = QtWidgets.QWidget()
        self.header_space.setFixedHeight(header.sizeHint().height())
        layout.addWidget(self.header_space)
        self.comment = QtWidgets.QLabel(objectName="comment", wordWrap=True)
        layout.addWidget(self.comment)
        self.notice = QtWidgets.QLabel(objectName="error", wordWrap=True)
        self.notice.setVisible(False)
        layout.addWidget(self.notice)
        self.groups_layout = QtWidgets.QVBoxLayout()
        self.groups_layout.setSpacing(2)
        layout.addLayout(self.groups_layout)
        # Details: the readers and compute nodes, below the views, folded at first
        self.details_header = QtWidgets.QWidget()
        details_row = QtWidgets.QHBoxLayout(self.details_header)
        details_row.setContentsMargins(0, 8, 0, 0)
        details_row.setSpacing(4)
        self.details_chevron = QtWidgets.QToolButton(objectName="icon", autoRaise=True)
        self.details_chevron.setIconSize(QtCore.QSize(14, 14))
        self.details_chevron.clicked.connect(self.toggle_details)
        details_row.addWidget(self.details_chevron)
        self.details_button = QtWidgets.QPushButton("Details", objectName="group", flat=True,
                                                    toolTip="The nodes that read and compute: their status, key "
                                                            "results and parameters")
        self.details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_button.clicked.connect(self.toggle_details)
        details_row.addWidget(self.details_button)
        self.details_hint = QtWidgets.QLabel("inputs and computations", objectName="hint")
        details_row.addWidget(self.details_hint, 1)
        layout.addWidget(self.details_header)
        self.details_box = QtWidgets.QWidget()
        self.details_layout = QtWidgets.QVBoxLayout(self.details_box)
        self.details_layout.setContentsMargins(14, 0, 0, 0)
        self.details_layout.setSpacing(2)
        layout.addWidget(self.details_box)
        self.empty = QtWidgets.QWidget()  # no nodes yet: they are added in the graph window
        empty_row = QtWidgets.QHBoxLayout(self.empty)
        empty_row.setContentsMargins(0, 6, 0, 6)
        empty_row.addWidget(QtWidgets.QLabel("No nodes yet: add them in the graph.", objectName="hint"))
        open_graph = QtWidgets.QPushButton("Open the graph", objectName="add")
        open_graph.clicked.connect(self._show_graph)
        empty_row.addWidget(open_graph)
        empty_row.addStretch()
        layout.addWidget(self.empty)
        layout.addStretch()
        self.rebuild()
        self.header.raise_()  # above the plots that scroll below it
        if analysis.read_only:
            self.show_notice("This analysis was written by a newer Tesselab: it is shown, but cannot be saved.")

    def place_header(self):
        """The header at the top of the card; when the tab is scrolled down, the card's top with the header stays at
        the top of the visible area (until the card's end), so that it stays clear which analysis the plots belong
        to. Called when the card's size changes and when its tab is scrolled."""
        space = self.header_space.geometry()  # the header's place in the layout
        area = self.window.scroll_area_of(self)
        inside = area is not None and area.viewport().isAncestorOf(self)
        scrolled = max(0, -self.mapTo(area.viewport(), QtCore.QPoint(0, 0)).y()) if inside else 0
        below = 5 if scrolled else 0  # a little room below the pinned header, before what scrolls by
        height = space.bottom() + 1 + below
        edge = max(0, min(scrolled, self.height() - height))
        self.header.pin(scrolled > 0)
        self.header.layout().setContentsMargins(space.x(), space.y(), self.width() - 1 - space.right(), below)
        self.header.setGeometry(0, edge, self.width(), height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.place_header()

    # --- header -------------------------------------------------------------------------------------------------
    def _make_header(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.status = QtWidgets.QToolButton(objectName="icon", autoRaise=True)  # in front, as in the groups
        self.status.setIconSize(QtCore.QSize(14, 14))
        self.status.clicked.connect(self._show_status)
        row.addWidget(self.status)
        self.label = QtWidgets.QLabel(objectName="label")
        self.label.mouseDoubleClickEvent = lambda event: self._rename()
        row.addWidget(self.label)
        row.addSpacing(10)
        # the key results of the compute nodes (they sit folded in Details): what the analysis is for, in view
        self.results_label = ElidedLabel(objectName="keyresult")
        row.addWidget(self.results_label, 1)
        # undo and redo here, not in the rail: they act on this analysis, plainly, and not on whichever card is active
        # (the user, 2026-11-23); usable only when there is a step (update_header)
        self.undo_button = theme.icon_button("undo", "Undo in this analysis (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = theme.icon_button("redo", "Redo in this analysis (Ctrl+Shift+Z or Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo)
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        row.addSpacing(8)
        for name, tip, slot in (("graph", "Show the graph of this analysis (Ctrl+G)", self._show_graph),
                                ("save", "Save (Ctrl+S)", self.save),
                                ("export", "Export PDF and JSON of every plot", self.export)):
            button = theme.icon_button(name, tip)
            button.clicked.connect(slot)
            row.addWidget(button)
            if name == "save":
                self.save_button = button
        self.link_button = theme.icon_button("link", "Link with the other linked analyses of this tab: they share "
                                                     "zoom, cursors and which groups are open.", checkable=True)
        self.link_button.toggled.connect(lambda on: self.link_toggled.emit(self, on))
        row.addWidget(self.link_button)
        more = QtWidgets.QToolButton(text="⋯", objectName="icon", autoRaise=True, toolTip="More")
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(more)
        menu.addAction("Rename…", self._rename)
        menu.addAction("Edit comment…", self._edit_comment)
        menu.addAction("Open the analysis file", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(self.analysis.path)) if self.analysis.path else None)
        menu.addAction("Open the graph file", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(self.analysis.graph.path)) if self.analysis.graph else None)
        more.setMenu(menu)
        row.addWidget(more)
        close = theme.icon_button("close", "Close this analysis")
        close.clicked.connect(lambda: self.close_requested.emit(self))
        row.addWidget(close)
        return row

    def update_header(self, check_unsaved=True):
        self.label.setText(self.analysis.label)
        comment = self.analysis.doc.get("comment", "")
        self.comment.setText(f"“{comment}”" if comment else "")
        self.comment.setVisible(bool(comment))
        self.undo_button.setEnabled(bool(self.undo_steps))
        self.redo_button.setEnabled(bool(self.redo_steps))
        if not check_unsaved and getattr(self, "_shown_unsaved", None) is not None:
            return
        unsaved = self.is_unsaved()
        if unsaved != getattr(self, "_shown_unsaved", None):  # the save icon: orange with unsaved changes
            self._shown_unsaved = unsaved
            self.save_button.setProperty("unsaved", unsaved)
            self.save_button.setIcon(theme.icon("save", "#ffffff" if unsaved else None))
            self.save_button.setToolTip("Save (Ctrl+S): there are unsaved changes" if unsaved else "Save (Ctrl+S)")
            self.save_button.style().unpolish(self.save_button)
            self.save_button.style().polish(self.save_button)

    def show_notice(self, text):
        self.notice.setText(text)
        self.notice.setVisible(bool(text))

    @property
    def undo_steps(self):
        return self.shared.undo_steps

    @property
    def redo_steps(self):
        return self.shared.redo_steps

    @property
    def saved_content(self):
        return self.shared.saved_content

    @saved_content.setter
    def saved_content(self, content):
        self.shared.saved_content = content

    def others(self):
        """The other cards of the same analysis (in other tabs)."""
        return [c for c in self.shared.cards if c is not self]

    def is_unsaved(self):
        graph = self.analysis.graph
        return self.analysis.state() != self.shared.saved_content or (graph is not None and graph.is_dirty())

    def set_active(self, active):
        """The active card (Ctrl+S, Ctrl+Z, ... act on it): its border in the accent color."""
        if self.property("active") != active:
            self.setProperty("active", active)
            self.header.active = active
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
            self.header.update()

    def _rename(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename analysis", "Label:", text=self.analysis.label)
        if ok and text.strip():
            self.edit(lambda: self.analysis.doc.__setitem__("label", text.strip()))

    def _edit_comment(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Comment", "Comment (shown under the label):",
                                                           self.analysis.doc.get("comment", ""))
        if ok:
            self.edit(lambda: self.analysis.doc.__setitem__("comment", text.strip()))

    def _show_status(self):
        results = self.runner.results(self.analysis)
        lines = []
        total = 0.0
        for name in self.analysis.doc["view"]["order"]:
            result = results.get(name)
            group = self.groups.get(name)
            title = group.title() if group else name
            if result is None:
                lines.append(f"{title}: running …")
                continue
            total += result.seconds
            loading = f", + {result.load_seconds * 1000:.0f} ms loading the module into a new worker" \
                if result.load_seconds > 0.002 else ""
            lines.append(f"{title}: {result.status}, {result.seconds * 1000:.0f} ms" + loading
                         + (f" (from the {result.cached} cache)" if result.cached else ""))
            for d in result.diagnostics:
                lines.append(f"    {d['level']}: {d['message']}")
        lines.append(f"\nComputing time: {total * 1000:.0f} ms (results that were reused count again here)")
        message_box(self, self.analysis.label, "\n".join(lines), icon="info")

    def _show_graph(self):
        from tesselab.gui.graph_view import GraphDialog
        return GraphDialog.show_for(self)

    def add_in_graph(self):
        """Analysis → Add a node (Ctrl+Shift+A): the graph window, with the search for a node by name."""
        from tesselab.gui.graph_view import GraphDialog
        return GraphDialog.show_for(self, search=True)

    # --- the graph file -----------------------------------------------------------------------------------------
    def share_graph(self):
        """Put the graph in a new graph file of its own (sharing an analysis's graph, or a separate copy of a shared
        one), which other analyses can then use."""
        from tesselab.graph import Graph
        project = self.window.project
        if project is None:
            message_box(self, "No project", "Graph files live in a project (tesselab/graphs).", icon="info")
            return
        current = self.analysis.graph
        default = current.label + " copy" if current else self.analysis.label
        name, ok = QtWidgets.QInputDialog.getText(self, "Graph file", "Name of the new graph file:", text=default)
        if not ok or not name.strip():
            return
        from tesselab.graph import split
        if current is None:  # values that come from the data stay this analysis's own: paths, template placeholders
            own = self._own_from_data()
            doc = clone(self.analysis.doc)
            doc["own"] = own
            made = split(doc)[0]
        else:
            own, made = None, clone(current.doc)
        made["label"] = name.strip()
        self._switch_graph(Graph(made, project.new_graph_path(name.strip())), own)

    def _own_from_data(self):
        """{node: [parameters]} that belong to the sample when an analysis's graph goes into a file of its own: data
        paths, and the values its template fills in from the data."""
        from tesselab.graph import own_from_template
        own = {}
        for name in self.analysis.nodes:
            try:
                cls = self.engine.node_class(self.analysis, name)
            except Exception:
                continue
            for p in cls.params:
                if isinstance(p, (P.Path, P.Paths)) and p.name in self.analysis.nodes[name].get("params", {}):
                    own.setdefault(name, []).append(p.name)
        from tesselab.modules import read_template
        registry = self.engine.registry
        template_id = self.analysis.doc.get("provenance", {}).get("template")
        for template in registry.templates().values():  # the analysis's template, or one with the same nodes
            try:
                raw = read_template(template.module, template)
            except Exception:
                continue
            same = all(self.analysis.nodes.get(n, {}).get("node") == e.get("node") for n, e in raw["nodes"].items())
            if template.full_id == template_id or (template_id is None and same):
                for name, keys in own_from_template(raw).items():
                    if name in self.analysis.nodes:
                        own.setdefault(name, []).extend(k for k in keys if k not in own.get(name, []))
                break
        return own

    def choose_graph_file(self):
        """Use an existing graph file of the project: its nodes and connections replace this analysis's; values that
        differ from the graph's stay this analysis's own."""
        project = self.window.project
        if project is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Use a graph file", project.folder("graphs"),
                                                        "Graph files (*.json)")
        if not path:
            return
        try:
            graph = self.engine.graphs.get(path)
        except Exception as error:
            message_box(self, "Not a graph file", str(error), icon="warning")
            return
        if graph is self.analysis.graph:
            return
        dropped = [n for n in self.analysis.nodes if n not in graph.doc["nodes"]]
        if dropped and message_box(self, "Use a graph file", f"The graph {graph.label} has no "
                                   f"{', '.join(dropped)}: they are removed from this analysis.",
                                   ("Use it", "Cancel"), default=1, icon="warning") != 0:
            return
        own = self._own_from_data() if self.analysis.graph is None else None
        self._switch_graph(graph, own)

    def _switch_graph(self, graph, own):
        """This analysis uses another Graph from now on (not an undo step: the undo history starts anew)."""
        old = self.analysis.graph
        self.analysis.use_graph(graph, own)
        self.analysis.doc["graph"] = self.analysis.graph_ref() if self.analysis.path else None
        self.engine.graphs.add(graph)
        if old is not None and not self.engine.sharing(self.analysis) and \
                not any(a.graph is old for a in self.engine.open.values()):
            self.engine.graphs.drop(old)
        self.shared.undo_steps.clear()
        self.shared.redo_steps.clear()
        self.rebuild_all()
        for i, card in enumerate(self.shared.cards):
            card._after_change(recompute=i == 0)
        self.window.statusBar().showMessage(f"{self.analysis.label} uses the graph file "
                                            f"{os.path.basename(graph.path)} now (saved with the analysis).", 8000)

    # --- groups ------------------------------------------------------------------------------------------------------
    def rebuild(self):
        """Make the groups again (after nodes were added, removed or moved): the views (not switched off) as the
        card's sections, then the other nodes in Details (folded), or as the card's groups if there is no view."""
        self.open_groups = [n for n in self.open_groups if n in self.analysis.nodes]
        open_ = set(self.open_groups)
        for group in self.groups.values():
            group.setParent(None)
            group.deleteLater()
        self.groups = {}
        views, details = [], []
        for name in self.analysis.doc["view"]["order"]:
            if name not in self.analysis.nodes:
                continue
            group = NodeGroup(self, name)
            if group.kind == "view":
                if self.analysis.nodes[name].get("off"):
                    group.deleteLater()
                    continue
                views.append(group)
            else:
                details.append(group)
            self.groups[name] = group
        for group in views:
            self.groups_layout.addWidget(group)
        target = self.details_layout if views else self.groups_layout
        for group in details:
            target.addWidget(group)
        self.details_header.setVisible(bool(views and details))
        self._show_details()
        for group in views + details:
            if group.name in open_:
                group.set_open(True)
        self.empty.setVisible(not self.groups)
        self.header.raise_()  # above the new groups: they would cover the pinned header
        self.update_header()
        self.renumber()
        self.update_results()

    @property
    def details_open(self):
        return bool(self.analysis.doc["view"].get("details"))

    def toggle_details(self):
        self.analysis.doc["view"]["details"] = not self.details_open  # saved, as the open groups
        self._show_details()
        self.window.view_changed(self)

    def _show_details(self):
        """Details (only when there are views and other nodes): its chevron, and its nodes shown when open."""
        self.details_chevron.setIcon(theme.icon("chevron_down" if self.details_open else "chevron_right"))
        self.details_box.setVisible(self.details_open and self.details_header.isVisibleTo(self))
        count = self.details_layout.count()
        self.details_hint.setText(f"inputs and computations · {count} node{'s' if count != 1 else ''}")

    def renumber(self):
        number = 1
        for name in self.analysis.doc["view"]["order"]:
            group = self.groups.get(name)
            if group is not None:
                group.number_plots(number)
                number += group.plot_count()

    def set_open(self, name, open_, emit=True):
        if open_ and name not in self.open_groups:
            self.open_groups.append(name)
        elif not open_:
            self.open_groups = [n for n in self.open_groups if n != name]
        self.analysis.doc["view"]["open"] = list(self.open_groups)  # saved: the folds of the card changed last
        group = self.groups.get(name)
        if group is not None:
            group.set_open(open_)
        if emit:
            self.fold_changed.emit(self, self.analysis.nodes[name]["node"], open_)
        self.window.view_changed(self)

    def update_results(self):
        results = self.runner.results(self.analysis)
        running = self.runner.is_running(self.analysis)
        levels = []
        for name, group in self.groups.items():
            result = results.get(name)
            group.show_result(result, name in running)
            if name in running:
                levels.append("running")
            elif result is not None:
                levels.append(result.level())
        parts = [part for group in self.groups.values() for part in group.parts]
        self.results_label.setText("   ".join(parts))
        if "running" in levels:
            level = "running"
        else:
            level = next((lv for lv in ("error", "warning") if lv in levels), None)
        self.status.setIcon(theme.dot_icon(level))
        total = sum(r.seconds for r in results.values())
        self.status.setToolTip({"running": "computing …", "error": "an error: click for details",
                                "warning": "warnings: click for details"}.get(level, "everything fine")
                               + f"  ({total * 1000:.0f} ms)")
        self.update_header(check_unsaved=False)  # (results arriving change nothing that is saved)
        if self._waiting_drag is not None and not self.runner.is_running(self.analysis):
            name, param, value, coalesce, propagate = self._waiting_drag
            self.set_param(name, param, value, False, coalesce, propagate)

    def group_of(self, name):
        return self.groups.get(name)

    def scroll_to(self, name):
        group = self.groups.get(name)
        if group is None:
            return
        if group.parentWidget() is self.details_box and not self.details_open:
            self.toggle_details()
        if name not in self.open_groups:
            self.set_open(name, True)
        area = self.window.scroll_area_of(self)
        if area is not None:
            area.ensureWidgetVisible(group, 0, self.header_space.height() + 30)  # not below the pinned header

    def link_group(self, group):
        self.window.link_plots(self)

    # --- editing --------------------------------------------------------------------------------------------------
    def edit(self, change, coalesce=None, recompute=True, redraw=True):
        """Make a change to the analysis as one undo step (or merged into the last one, for a dragged cursor). The
        other cards of the same analysis show it too. recompute=False: nothing that results depend on changed
        (a plot setting); redraw=False: no plot changed either (a node moved in the graph)."""
        shared, analysis = self.shared, self.analysis
        before = analysis.content()
        graph_before = clone(analysis.graph.doc) if analysis.graph else None
        nodes_before = list(analysis.nodes)
        change()
        graph_changed = analysis.push_graph()
        after = analysis.content()
        if after == before and not graph_changed:
            return False
        graph_after = clone(analysis.graph.doc) if analysis.graph else None
        if coalesce is None or coalesce != shared.coalesce or not shared.undo_steps:
            # an undo step: (content before, the graph then, the graph this edit left)
            shared.undo_steps.append((before, graph_before, graph_after))
            del shared.undo_steps[:-200]
        else:  # merged into the last step (a dragged cursor): it now leads to this graph
            shared.undo_steps[-1] = shared.undo_steps[-1][:2] + (graph_after,)
        shared.coalesce = coalesce
        shared.redo_steps.clear()
        if list(analysis.nodes) != nodes_before:
            self.rebuild_all()  # first: what follows (settings, results) must not meet a removed node's group
        self._after_change(recompute, redraw)
        for other in self.others():
            other._after_change(recompute=False, redraw=redraw)
        if graph_changed:
            self.window.graph_changed(analysis, recompute, redraw)
        return True

    def end_coalesce(self):
        self.shared.coalesce = None

    def _after_change(self, recompute=True, redraw=True):
        self.update_header()
        self.saved_changed.emit(self)
        if recompute:
            self.changed.emit(self)
        for group in self.groups.values():
            if group.form is not None or group.input_widgets:
                group.update_settings()
            if group.is_open and not recompute and redraw:
                group.draw()  # e.g. a λ marker changed in another card of the same analysis
                group.update_heading()

    def undo(self):
        self._restore(self.shared.undo_steps, self.shared.redo_steps)

    def redo(self):
        self._restore(self.shared.redo_steps, self.shared.undo_steps)

    def _restore(self, steps, back):
        """Put back the last of steps (content, the graph it was taken with, the graph it expects now) and add the
        step that goes back again to back, before anything is told of the change (the undo and redo buttons read
        both lists then). A step that changed a shared graph puts the graph back only if no other analysis changed
        it since."""
        if not steps:
            return
        content, content_graph, expected = steps.pop()
        analysis = self.analysis
        now = analysis.content()
        graph_now = clone(analysis.graph.doc) if analysis.graph else None
        nodes_before = list(analysis.nodes)
        if analysis.graph is None:
            analysis.restore(content)
            target = None
        else:
            target = content_graph
            if content_graph != expected and graph_now != expected:
                target = graph_now
                self.window.statusBar().showMessage("The graph was changed in another analysis since: only this "
                                                    "analysis's values were put back.", 8000)
            analysis.restore(content, content_graph, target)
        back.append((now, graph_now, target))
        self.shared.coalesce = None
        if list(analysis.nodes) != nodes_before:
            self.rebuild_all()  # every card first, then the recomputation (see edit)
        for card in self.shared.cards:
            card._after_change(recompute=card is self)
        if analysis.graph is not None and target != graph_now:
            self.window.graph_changed(analysis)

    def rebuild_all(self):
        """All cards of this analysis make their groups again (nodes added, removed or moved)."""
        for card in self.shared.cards:
            card.rebuild()

    def param_value(self, name, param):
        values, _ = self.engine.resolve_params(self.analysis, name, absolute=False)
        head, _, rest = param.partition(".")
        if not rest:
            return values.get(param)
        kind, _, sub = rest.partition(".")
        for entry in values.get(head) or []:
            if entry["kind"] == kind:
                return entry.get(sub)
        return None

    def set_param(self, name, param, value, final=True, coalesce=None, propagate=True):
        """Set a parameter (as typed, dragged or picked). 'steps.chirp.onsets' sets a parameter of the first step
        of that kind in a Steps parameter (adding the step if there is none). A parameter that a linked cursor
        shows is passed on to the linked analyses (propagate)."""
        if not final and coalesce is not None and self.runner.is_running(self.analysis):
            # A step of a dragged cursor while the last one is still being computed: only the newest waits for it
            # (update_results sends it). Sending every step made the workers compute positions nobody would see, and
            # the plots that depend on λ showed nothing until the mouse stopped; now the cursor follows the mouse
            # at once and what depends on it follows as fast as the computations are (2026-11-24).
            self._waiting_drag = (name, param, value, coalesce, propagate)
            return
        self._waiting_drag = None
        entry = self.analysis.nodes[name]
        cls = self.engine.node_class(self.analysis, name)
        p = P.find(cls.params, param)

        def change():
            saved = entry.setdefault("params", {})
            head, _, rest = param.partition(".")
            if rest:
                kind, _, sub = rest.partition(".")
                steps_param = P.find(cls.params, head)
                steps = [dict(s) for s in (saved.get(head) if isinstance(saved.get(head), list)
                                           else steps_param.default_value() or [])]
                target = next((s for s in steps if s["kind"] == kind), None)
                if target is None:
                    target = steps_param.kinds[kind].defaults()
                    steps.append(target)
                target[sub] = value
                saved[head] = steps_param.check(steps)
                self.analysis.set_own(name, head)
                return
            current = saved.get(param)
            if is_var(current):
                self.analysis.vars[current["var"]] = value
            elif current != value or param not in saved:
                saved[param] = value
                self.analysis.set_own(name, param)  # a value set in this analysis: its own, not the graph's

        redraw_only = p is not None and p.redraw
        saved = entry.get("params", {}).get(param)
        bound_var = saved["var"] if isinstance(saved, dict) and set(saved) == {"var"} else None
        changed = self.edit(change, coalesce=coalesce, recompute=not redraw_only)
        if final:
            self.shared.coalesce = None
        if not changed and not final:
            return  # e.g. a linked analysis got the value already through another of its cursors
        if redraw_only:
            group = self.groups.get(name)
            if group is not None:
                group.draw()
        if bound_var is not None:
            self._redraw_bound(bound_var)
        if propagate:
            self._propagate(name, param, value, final, bound_var)

    def _cursor_links(self, name, param):
        """The link keys of the cursors that show parameter `param` of node instance `name`."""
        try:
            cls = self.engine.node_class(self.analysis, name)
        except Exception:
            return []
        return [i.link for i in cls.interactions if getattr(i, "link", None) and getattr(i, "param", None) == param]

    def _propagate(self, name, param, value, final, bound_var=None):
        """Tell linked analyses about a changed parameter that a linked cursor shows (also when it was typed): for
        a parameter bound to a variable, every cursor parameter bound to the same variable counts."""
        targets = [(name, param)]
        if bound_var is not None:
            targets = [(n, k) for n, e in self.analysis.nodes.items() for k, v in e.get("params", {}).items()
                       if v == {"var": bound_var}]
        sent = set()
        for n, k in targets:
            for link in self._cursor_links(n, k):
                key = f"{self.analysis.nodes[n]['node']}:{link}"
                if key not in sent:
                    sent.add(key)
                    self.cursor_moved.emit(self, key, value, final)

    def _redraw_bound(self, var):
        """Plots of nodes whose redraw-only parameters are bound to the variable (e.g. λ markers)."""
        for n, e in self.analysis.nodes.items():
            if any(v == {"var": var} for v in e.get("params", {}).values()):
                group = self.groups.get(n)
                if group is not None:
                    group.draw()

    def set_var(self, var, value):
        self.edit(lambda: self.analysis.vars.__setitem__(var, value))
        self._redraw_bound(var)
        self._propagate(None, None, value, True, var)

    def bind(self, name, param, var):
        entry = self.analysis.nodes[name].setdefault("params", {})

        def change():
            if var is None:
                current = entry.get(param)
                if isinstance(current, dict) and "var" in current:
                    entry[param] = self.analysis.vars.get(current["var"])
            else:
                followed = any(v == {"var": var} for e in self.analysis.nodes.values()
                               for v in e.get("params", {}).values())
                if var not in self.analysis.vars or not followed:  # a new variable takes the parameter's value
                    value = entry.get(param)
                    if value is None or isinstance(value, dict):
                        cls = self.engine.node_class(self.analysis, name)
                        value = P.find(cls.params, param).default_value()
                    self.analysis.vars[var] = value
                entry[param] = {"var": var}
                self.analysis.set_own(name, param, False)  # bound: it follows the variable in every analysis
        self.edit(change)

    def reset_param(self, name, param):
        """The parameter takes the graph's value again (a shared graph file)."""
        self.edit(lambda: self.analysis.reset_param(name, param))

    def promote_param(self, name, param):
        """This analysis's value becomes the graph's (for every analysis using the graph that does not set its
        own)."""
        self.edit(lambda: self.analysis.promote_param(name, param))

    def connect_form(self, form, name):
        """A parameter form of node instance `name` (in this card or in the graph window) edits through this card."""
        form.changed.connect(lambda key, value: self.set_param(name, key, value))
        form.var_changed.connect(self.set_var)
        form.bind_requested.connect(lambda key, var: self.bind(name, key, var))
        form.reset_requested.connect(lambda key: self.reset_param(name, key))
        form.promote_requested.connect(lambda key: self.promote_param(name, key))

    def update_form(self, form, name, only_if_changed=False):
        """A parameter form shows the current values; with a shared graph file it marks this analysis's own."""
        entry = self.analysis.nodes[name]
        own = None
        if self.analysis.graph is not None:
            own = {p.name: self.analysis.graph_value(name, p.name) for p in form.params
                   if self.analysis.is_own(name, p.name)}
        shown = (repr(entry.get("params", {})), repr(self.analysis.vars), repr(own), theme.current()["accent"])
        if only_if_changed and getattr(form, "shown_values", None) == shown:
            return
        form.shown_values = shown
        form.set_values(entry.get("params", {}), self.analysis.vars, own)

    def source_text(self, source):
        """An input's source as shown: "Spectra › spectra", or the other analysis's name for a link."""
        if source.file:
            return f"{os.path.splitext(os.path.basename(source.file))[0]} › {source.instance}.{source.output}"
        group = self.groups.get(source.instance)
        return f"{group.title() if group else source.instance} › {source.output}"

    def set_input(self, name, input_name, source):
        def change():
            self.analysis.set_input(name, input_name, source)
        try:
            self.edit(change)
        except CycleError as error:
            message_box(self, "Not possible", str(error), icon="warning")

    def set_binding(self, name, input_name, var, value):
        current = Source.parse(self.analysis.nodes[name]["inputs"][input_name])
        bind = dict(current.bind)
        bind[var] = value
        self.edit(lambda: self.analysis.nodes[name]["inputs"].__setitem__(
            input_name, Source(current.instance, current.output, current.file, bind, current.seen).to_json()))

    def choose_link(self, name, input_name):
        from tesselab.gui.dialogs import LinkDialog
        dialog = LinkDialog(self, self.window.project, self.analysis, self.engine)
        if dialog.exec() and dialog.source:
            self.set_input(name, input_name, dialog.source)
        else:
            self.groups[name].update_settings()

    def remove(self, name):
        group = self.groups.get(name)
        title = group.title() if group else name
        if message_box(self, "Remove", f"Remove {title} from this analysis? (Undo brings it back.)",
                       ("Remove", "Cancel"), default=1) != 0:
            return
        self.edit(lambda: self.analysis.remove(name))

    def move(self, name, delta):
        order = self.analysis.doc["view"]["order"]
        i = order.index(name)
        if 0 <= i + delta < len(order):
            order[i], order[i + delta] = order[i + delta], order[i]
            self.rebuild_all()
            self.window.view_changed(self)

    def run_node(self, name):
        self.runner.request(self.analysis, run=[name])

    def run_action(self, name, action):
        """A node's button: run(ctx) here, or with job=True in a worker process (the window stays usable, the button
        shows that it is busy); its answer is a parameter change, a file to save, or nothing."""
        result = self.runner.results(self.analysis).get(name)
        ctx = self.engine.draw_context(self.analysis, name, result)
        if action.enabled is not None and not action.enabled(ctx):
            return
        if action.job:
            from tesselab import worker
            info = self.engine.node_info(self.analysis, name)
            task = {"folder": info.module.folder, "module_id": info.module.id, "code_hash": info.module.code_hash,
                    "class_spec": info.class_spec}
            button = self.groups[name].action_buttons.get(action.id)
            if button is not None:
                button.setEnabled(False)
                button.setText(f"{action.label} …")
            future = worker.pool().submit(worker.run_action, task, action.id, ctx)
            future.add_done_callback(lambda f: self._emit_job_done(name, action, f))
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            answer = action.run(ctx)
        except Exception as error:
            traceback.print_exc()
            message_box(self, action.label, f"{type(error).__name__}: {error}", icon="warning")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._apply_answer(name, action, answer)

    def _emit_job_done(self, name, action, future):
        try:
            self._job_done.emit(name, action, future)  # to the window's thread (a queued connection)
        except RuntimeError:
            pass  # the card was closed meanwhile

    def _finish_job(self, name, action, future):
        group = self.groups.get(name)
        button = group.action_buttons.get(action.id) if group is not None else None
        if button is not None:
            button.setEnabled(True)
            button.setText(action.label)
        if future.cancelled():
            return
        try:
            answer = future.result()
        except Exception as error:
            message_box(self, action.label, f"{type(error).__name__}: {error}", icon="warning")
            return
        if name in self.analysis.nodes:  # not removed while the job ran
            self._apply_answer(name, action, answer)

    def _apply_answer(self, name, action, answer):
        if isinstance(answer, dict):
            cls = self.engine.node_class(self.analysis, name)
            try:  # the values as the parameters' types store them (np.float64 → float), checked
                checked = {}
                for key, value in answer.items():
                    p = P.find(cls.params, key)
                    if p is None:
                        raise ValueError(f"the node has no parameter {key!r}")
                    checked[key] = p.check(value)
                answer = checked
            except (ValueError, TypeError) as error:
                message_box(self, action.label, f"The node's answer cannot be used: {error}", icon="warning")
                return

            def change():
                params = self.analysis.nodes[name].setdefault("params", {})
                for key, value in answer.items():
                    current = params.get(key)
                    if is_var(current):
                        self.analysis.vars[current["var"]] = value
                    else:
                        params[key] = value
                        self.analysis.set_own(name, key)
            self.edit(change)
        elif isinstance(answer, tuple) and len(answer) == 3 and answer[0] == "file":
            _, suggested, data = answer
            start = os.path.join(self.project_root or os.getcwd(), suggested)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, action.label, start)
            if path:
                with open(path, "wb") as f:
                    f.write(data if isinstance(data, bytes) else str(data).encode())

    # --- adding nodes -----------------------------------------------------------------------------------------------
    def fill_add_menu(self, menu, pos=None, done=None):
        """The nodes that can be added, into a menu: those whose inputs fit what this analysis already produces
        first, then readers, then the rest. pos: where the node goes in the graph (grid units); done(name) is
        called with the new instance's name."""
        menu.clear()
        add = lambda f: self._added(self.add_node(f, pos=pos, scroll=False), done)
        found = self.addable()
        for kind, section in (("fits", "Fits this analysis"), ("reads", "Read data")):
            items = [item for item in found if item[3] == kind]
            if items:
                menu.addSection(section)
            for label, full_id, module, _ in items:
                menu.addAction(f"{label}   ({module})", lambda f=full_id: add(f))
        others = [item for item in found if item[3] in ("other", "broken")]
        if others:
            sub = menu.addMenu("Other nodes (inputs not connected yet)")
            for label, full_id, module, kind in others:
                action = sub.addAction(f"{label}   ({module})", lambda f=full_id: add(f))
                action.setEnabled(kind != "broken")
        if menu.isEmpty():
            menu.addAction("No modules found").setEnabled(False)

    @staticmethod
    def _added(name, done):
        if name and done is not None:
            done(name)

    def addable(self):
        """[(label, full id, module name, kind)] of every node that can be added; kind: "fits", "reads", "other"
        (as in the Add menu), or "broken" with the error as label."""
        found = []
        results = self.runner.results(self.analysis)
        for full_id, info in sorted(self.engine.registry.nodes().items(), key=lambda kv: kv[1].module.name):
            try:
                cls = self.engine.registry.node_class(full_id)
            except Exception as error:
                found.append((f"{info.id} (cannot be loaded: {error})", full_id, info.module.name, "broken"))
                continue
            if all(s.optional for s in cls.inputs.values()):
                kind = "reads"
            else:
                sources = self._auto_inputs(cls, results)
                kind = "fits" if sources and all(n in sources or s.optional for n, s in cls.inputs.items()) \
                    else "other"
            found.append((cls.name or info.id, full_id, info.module.name, kind))
        return found

    def _auto_inputs(self, cls, results):
        """For each input of a node class, a source in this analysis: the source of a node that takes the same kind
        of input (nodes of one kind sit side by side: MCR next to SVD), else the output whose variables carry the
        signature's names, else the latest fitting output. Optional inputs stay unconnected."""
        chosen = {}
        for input_name, signature in cls.inputs.items():
            if signature.optional:
                continue  # an optional input is the user's choice
            for name, entry in self.analysis.nodes.items():
                try:
                    other = self.engine.node_class(self.analysis, name)
                except Exception:
                    continue
                for other_input, other_signature in other.inputs.items():
                    if repr(other_signature) == repr(signature) and entry.get("inputs", {}).get(other_input):
                        chosen[input_name] = entry["inputs"][other_input]
                        break
                if input_name in chosen:
                    break
            if input_name in chosen:
                continue
            best = None
            for name in reversed(self.analysis.order()):
                result = results.get(name)
                if result is None or not result.ok:
                    continue
                for output, dataset in result.outputs.items():
                    if bindings(signature, dataset):
                        score = sum(var in dataset.variables for var in signature.variables)
                        if best is None or score > best[0]:
                            best = (score, f"{name}.{output}")
            if best is not None:
                chosen[input_name] = best[1]
        return chosen

    def add_node(self, full_id, pos=None, inputs=None, scroll=True):
        """Add a node (its inputs connected as _auto_inputs finds them; inputs: some given instead); returns the new
        instance's name. pos: its place in the graph (grid units). scroll: show its group in the card."""
        registry = self.engine.registry
        cls = registry.node_class(full_id)
        info = registry.node(full_id)
        chosen = self._auto_inputs(cls, self.runner.results(self.analysis))
        chosen.update(inputs or {})
        added = []
        views = []  # the node's default views (Node.views): added with it, connected to it
        for view_id in getattr(cls, "views", []):
            view_full = view_id if "/" in view_id else f"{info.module.id}/{view_id}"
            try:
                views.append((view_full, registry.node_class(view_full), registry.node(view_full)))
            except Exception:
                continue

        def change():
            name = self.analysis.add(full_id, inputs=chosen, version=info.module.version, schema=cls.schema, pos=pos)
            added.append(name)
            self.open_groups.append(name)  # open here (before edit builds it); closed in the other cards
            outputs = cls.output_names()
            for i, (view_full, view_cls, view_info) in enumerate(views):
                # each input from the output of the same name, or from the node's only output
                wiring = {k: f"{name}.{k if k in outputs else outputs[0]}" for k, sig in view_cls.inputs.items()
                          if k in outputs or (len(outputs) == 1 and not sig.optional)}
                view_pos = (pos[0] + 16, pos[1] + 6 * i) if pos else None
                view_name = self.analysis.add(view_full, inputs=wiring, version=view_info.module.version,
                                              schema=view_cls.schema, pos=view_pos)
                self.open_groups.append(view_name)
        try:
            self.edit(change)
        except CycleError as error:
            message_box(self, "Not possible", str(error), icon="warning")
            return None
        if added and scroll:
            QtCore.QTimer.singleShot(50, lambda: self.scroll_to(added[0]))
        return added[0] if added else None

    # --- saving -------------------------------------------------------------------------------------------------------
    def save(self):
        return self.window.save_card(self)

    def mark_saved(self):
        self.shared.saved_content = self.analysis.state()
        for card in self.shared.cards:
            card.update_header()
            card.saved_changed.emit(card)

    def export(self):
        self.window.export_card(self)
