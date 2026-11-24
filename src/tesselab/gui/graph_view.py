"""
The graph window of an analysis: a window of its own with the graph as an editable drawing on a dot grid, a small
toolbar above it and an inspector beside it (inspector.py: the selected node's inputs and parameters, or the
analysis's graph file and variables when nothing is selected).

The drawing: each node a box, shaded by its kind (readers bluish, compute nodes plain, views greenish: input,
computation, output; views stand in the last column, a view switched off is faint), with status dot, name, instance
and time; inputs as ports on the left edge, outputs on the right; then the parameters bound to an analysis variable, a row each in that variable's color; then the files the
node reads, a row each with a file or folder symbol). An input port is filled when connected, dashed when it is
optional and left open, red when it is needed and left open. Wires run from an output to the inputs it feeds, dashed
from another analysis (a dashed box left of all nodes, at the height of the node it feeds). The analysis variables
are a table in the canvas's top right corner, each in its own color (no wires: the colored rows in the nodes name
them); a legend of the symbols sits in the bottom right corner.

Mouse: drag a box to move it (it snaps to the grid; positions belong to the graph and are saved with it); drag on
empty space to select several; pan with the middle button or Space + drag; zoom with Ctrl or Shift + wheel (the wheel
scrolls). Drag from an output to an input (or back) to connect, a connection off its input to move it, or onto empty
space to remove it; drop a new connection from an output on empty space to add a node that takes it. Drag a
variable's row from the table onto a node to bind one of its parameters. Double-click a node: its section in the main
window; double-click empty space or Shift+A: add a node by name; double-click a variable: edit its value. Right-click
anything for its menu. Keys: Delete removes the selection, Ctrl+D duplicates it, F2 renames, Home fits everything
into view, Ctrl+A selects all, Esc cancels, Ctrl+Z undoes, Ctrl+Shift+Z or Ctrl+Y redoes (also the toolbar's
arrows), Ctrl+S saves, Ctrl+B folds the side panel. The window opens at a readable zoom (READABLE); scroll bars
show where the rest of the graph is.

Every edit goes through the AnalysisCard (set_input, add_node, edit, ...), so undo, the other cards of the analysis,
the other analyses sharing its graph file and recomputation follow. Nodes without a stored position are placed in
columns by dependency; the first move stores every position (in grid units, see graph.py).
"""
import copy
import json
import math
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import params as P
from tesselab.analysis import CycleError, Source, sources_of
from tesselab.engine import node_kind
from tesselab.graph import GRID, is_var
from tesselab.gui import theme
from tesselab.gui.widgets import message_box
from tesselab.signature import bindings

NODE_W, HEADER_H, ROW_H = 12 * GRID, 2 * GRID, GRID  # NODE_W: the narrowest box (boxes grow to fit their text)
MAX_W = 26 * GRID  # the widest box
TIP_DELAY = 350  # ms the mouse rests before a tool tip shows (Qt's own wait is about twice that)
READABLE = 0.85  # the smallest zoom when the window opens: the boxes' text can be read
DIM = 0.3  # opacity of what does not accept the dragged data
GHOST = 0.3  # opacity of a wire where it passes behind a box (drawn faintly over the box)
STATUS = ("good", "warning", "error", "running", "waiting")


# --- placing -------------------------------------------------------------------------------------------------------
def layout_graph(analysis, heights=None, kinds=None, widths=None):
    """{instance: (x, y)} in grid units: columns by the longest path from a node without inputs, each column's nodes
    below each other (heights: {instance: height in grid units}, default 4); the views (kinds: {instance: kind}) in a
    column of their own after all others, so that the results stand at the right end. widths: {instance: width in
    grid units} (default NODE_W): each column as wide as its widest box."""
    order = analysis.order()
    kinds, widths = kinds or {}, widths or {}
    column = {}
    for name in order:
        deps = [d for d in analysis.dependencies(name) if d in analysis.nodes]
        column[name] = 1 + max((column[d] for d in deps), default=-1) if deps else 0
    last = max((c for n, c in column.items() if kinds.get(n) != "view"), default=-1)
    for name in column:
        if kinds.get(name) == "view":
            column[name] = last + 1
    if analysis.links():  # room for the linked analyses on the left
        column = {n: c + 1 for n, c in column.items()}
    heights = heights or {}
    column_width = {}
    for name, c in column.items():
        column_width[c] = max(column_width.get(c, 0), widths.get(name, NODE_W // GRID))
    lefts, x = {}, 0
    for c in sorted(column_width):
        lefts[c] = x
        x += column_width[c] + 4
    tops, positions = {}, {}
    for name in analysis.doc["view"].get("order", order):
        if name in column:
            c = column[name]
            positions[name] = (lefts[c], tops.get(c, 0))
            tops[c] = tops.get(c, 0) + heights.get(name, 4) + 1
    return positions


def place(analysis, heights, kinds=None, widths=None):
    """Where each node goes (grid units): its stored position, else the automatic one, moved down until it overlaps
    no stored box."""
    widths = widths or {}
    auto = layout_graph(analysis, heights, kinds, widths)
    taken, positions = [], {}
    for name, entry in analysis.nodes.items():
        if isinstance(entry.get("pos"), list) and len(entry["pos"]) == 2:
            positions[name] = tuple(entry["pos"])
            taken.append((*positions[name], widths.get(name, NODE_W // GRID), heights.get(name, 4)))
    for name in analysis.order():
        if name in positions:
            continue
        x, y = auto.get(name, (0, 0))
        h = heights.get(name, 4)
        width = widths.get(name, NODE_W // GRID)
        while any(x < tx + tw + 1 and tx < x + width + 1 and y < ty + th + 1 and ty < y + h + 1
                  for tx, ty, tw, th in taken):
            y += 1
        positions[name] = (x, y)
        taken.append((x, y, width, h))
    return positions


def box_height(ports, rows):
    """A node box's height in pixels (a multiple of the grid): header, ports (the longer side), then the rows of
    bound parameters and files."""
    height = HEADER_H + ROW_H * (ports + rows) + (6 if ports or rows else 0)
    return math.ceil(height / GRID) * GRID


def _font(size=None, bold=False):
    font = QtGui.QFont()
    font.setBold(bold)
    if size:
        font.setPointSizeF(size)
    return font


def text_width(text, size=None, bold=False) -> float:
    return QtGui.QFontMetricsF(_font(size, bold)).horizontalAdvance(text) if text else 0.0


def box_width(title, inputs, outputs, dims, bound, files) -> int:
    """A box's width in pixels (a multiple of the grid, NODE_W to MAX_W): room for its title and, on every row, the
    input's name and dimensions on the left and the output's on the right, without cutting them."""
    need = text_width(title, bold=True) + 33
    for i in range(max(len(inputs), len(outputs))):
        left = right = 0.0
        if i < len(inputs):
            extra = dims.get(("in", inputs[i]), "")
            left = 9 + text_width(inputs[i], 8) + (3 + text_width(extra, 7.5) if extra else 0)
        if i < len(outputs):
            extra = dims.get(("out", outputs[i]), "")
            right = 9 + text_width(outputs[i], 8) + (3 + text_width(extra, 7.5) if extra else 0)
        need = max(need, left + right + 16)
    for _, label, var in bound:
        need = max(need, 20 + text_width(f"{label} ← {var}", 8))
    for _, text, _, _, _ in files:
        need = max(need, 34 + text_width(text, 8))
    return int(min(MAX_W, max(NODE_W, math.ceil(need / GRID) * GRID)))


def dims_text(dims, sizes=None) -> str:
    """A port's dimensions as shown after its name: their sizes once there is a result ("5 × 201"), before that
    only how many there are ("(n, m)"). Their names are in the port's tool tip. (The user chose sizes from rendered
    variants, 2026-11-24: full names did not fit.)"""
    dims = list(dims)
    if not dims:
        return ""
    if sizes and all(d in sizes for d in dims):
        return " × ".join(str(sizes[d]) for d in dims)
    return "(" + ", ".join("nmklij"[i] if i < 6 else "·" for i in range(len(dims))) + ")"


def port_dims(signature=None, dataset=None) -> tuple[list, dict]:
    """(dimensions, sizes) of an input (its signature's variables) or an output: the dimensions of the variable that
    spans all of the dataset's dimensions (the spectra of a dataset with their totals). A dataset whose variables
    lie along unrelated dimensions (the rig's records: spectrum times, stirring periods, interrupts) is no one table:
    none (shape_text shows its tables' lengths)."""
    found, sizes = [], {}
    if dataset is not None:
        every = set(dataset.dims)
        main = next((v for v in dataset.data_vars.values() if set(v.dims) == every and v.ndim), None)
        if main is None:
            return [], {}
        found, sizes = list(main.dims), dict(dataset.sizes)
    elif signature is not None:
        for var in signature.variables.values():
            for d in var.dims:
                if d not in found:
                    found.append(d)
    return found, sizes


def shape_text(dataset) -> str:
    """What an output holds, after the port's name: one table ("51 × 160", its variables spanning the same
    dimensions), or several tables of their own lengths ("153 | 51 | 1": the rig's records, spectrum times, stirring
    periods and interrupts), single numbers left out."""
    dims, sizes = port_dims(dataset=dataset)
    if dims:
        return dims_text(dims, sizes)
    shapes = []
    for variable in dataset.data_vars.values():
        if variable.ndim:
            shape = " × ".join(str(dataset.sizes[d]) for d in variable.dims)
            if shape not in shapes:
                shapes.append(shape)
    return " | ".join(shapes)


def _shade(color, factor):
    c = QtGui.QColor(color)
    return QtGui.QColor.fromHsvF(c.hsvHueF() % 1.0, c.hsvSaturationF(), max(0.0, min(1.0, c.valueF() * factor))).name()


def var_colors(variables) -> dict:
    """{variable: color}: a color of the theme's series each, in the order of the variables (red and grey left out:
    red marks a missing input, grey is the color of the lines); beyond eight variables the same hues darker (on a
    dark theme lighter), so that no two variables share a color."""
    t = theme.current()
    series = t["series"]
    palette = [c for i, c in enumerate(series) if i % 10 not in (3, 7)] or series
    palette = palette + [_shade(c, 1.35 if t["dark"] else 0.62) for c in palette]
    return {var: palette[i % len(palette)] for i, var in enumerate(variables)}


def file_rows(engine, analysis, name, cls) -> list:
    """[(label, text, tooltip, exists, folder)] of the files and folders a node reads: its path parameters that have a
    value."""
    rows = []
    try:
        values, _ = engine.resolve_params(analysis, name)
    except Exception:
        return rows
    for p in cls.params:
        if not isinstance(p, (P.Path, P.Paths)) or not values.get(p.name):
            continue
        folder = isinstance(p, P.Path) and p.folder
        paths = [values[p.name]] if isinstance(p, P.Path) else list(values[p.name])
        exists = all(os.path.isdir(f) if folder else os.path.isfile(f) for f in paths) and bool(paths)
        text = os.path.basename(paths[0].rstrip("/\\")) if len(paths) == 1 else f"{len(paths)} files"
        tip = (f"<b>{p.label}</b>: reads " + "<br>".join(paths[:12]) + ("<br>…" if len(paths) > 12 else "")
               + ("" if exists else "<br><b>Not found.</b>"))
        if folder and exists:  # what is inside: the node may read any of it (a change to any file runs it again)
            inside = sorted(f for f in os.listdir(paths[0]) if not f.startswith("."))
            tip += (f"<br>with {len(inside)} files: " + ", ".join(inside[:20]) + (", …" if len(inside) > 20 else "")
                    + "<br><i>A change to any of them runs the node again.</i>")
        rows.append((p.label, text, tip, exists, folder))
    return rows


def file_symbol(folder, x, y, size=10) -> QtGui.QPainterPath:
    """A small document (a sheet with a folded corner) or folder outline, its top left at (x, y)."""
    path = QtGui.QPainterPath()
    if folder:
        w, h = size * 1.2, size * 0.85
        top = y + size * 0.15
        path.moveTo(x, top + h)
        path.lineTo(x, top)
        path.lineTo(x + w * 0.4, top)
        path.lineTo(x + w * 0.5, top + h * 0.18)
        path.lineTo(x + w, top + h * 0.18)
        path.lineTo(x + w, top + h)
        path.closeSubpath()
    else:
        w, h, fold = size * 0.8, size, size * 0.3
        path.moveTo(x, y)
        path.lineTo(x + w - fold, y)
        path.lineTo(x + w, y + fold)
        path.lineTo(x + w, y + h)
        path.lineTo(x, y + h)
        path.closeSubpath()
        path.moveTo(x + w - fold, y)
        path.lineTo(x + w - fold, y + fold)
        path.lineTo(x + w, y + fold)
    return path


def diamond(x, y, r=5) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y - r), QtCore.QPointF(x + r, y), QtCore.QPointF(x, y + r),
                            QtCore.QPointF(x - r, y)])


def bound_params(analysis) -> dict:
    """{variable: [(instance, parameter)]} of the parameters bound to each analysis variable."""
    bound = {}
    for name, entry in analysis.nodes.items():
        for key, value in entry.get("params", {}).items():
            if is_var(value):
                bound.setdefault(value["var"], []).append((name, key))
    return bound


def bindable(p) -> bool:
    """Parameters that can follow an analysis variable (not lists of steps or files). Picked points can: a view picks
    them, the compute node bound to the same variable uses them."""
    return not isinstance(p, (P.Steps, P.Path, P.Paths))


def kind_fill(kind) -> str:
    """The background of a node's box: input (reader), computation, output (view)."""
    t = theme.current()
    return {"reader": t.get("node_reader", t["tile"]), "view": t.get("node_view", t["tile"])}.get(kind, t["tile"])


def kind_of(p):
    return type(p).__name__, getattr(p, "unit", None)


def signature_text(signature) -> str:
    return ", ".join(f"{name}({', '.join(var.dims)})" for name, var in signature.variables.items())


def _snap(value):
    return QtCore.QPointF(round(value.x() / GRID) * GRID, round(value.y() / GRID) * GRID)


def _text(parent, text, x, y, color, bold=False, size=None, width=None):
    font = QtGui.QFont()
    font.setBold(bold)
    if size:
        font.setPointSizeF(size)
    if width:
        text = QtGui.QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, int(width))
    item = QtWidgets.QGraphicsSimpleTextItem(text, parent)
    item.setFont(font)
    item.setBrush(QtGui.QColor(color))
    item.setPos(x, y)
    return item


def _status_color(level):
    t = theme.current()
    return {None: t["roles"]["good"], "info": t["roles"]["good"], "warning": t["roles"]["warning"],
            "error": t["roles"]["error"], "running": t["accent"], "waiting": t["muted"]}.get(level, t["muted"])


# --- items ---------------------------------------------------------------------------------------------------------
class Box(QtWidgets.QGraphicsPathItem):
    """A node (name set; movable) or a linked analysis (name None, dashed, fixed): status dot and title, a subtitle,
    input ports on the left edge, output ports on the right, then a row per parameter bound to a variable (in the
    variable's color) and per file the node reads."""

    def __init__(self, editor, name, title, subtitle, level, inputs=(), outputs=(), bound=(), dashed=False,
                 tooltips=None, connected=(), optional=(), files=(), colors=None, dims=None, kind="compute",
                 off=False, width=NODE_W):
        t = theme.current()
        self.editor, self.name, self.width = editor, name, width
        self.inputs, self.outputs, self.bound = list(inputs), list(outputs), list(bound)  # bound: [(param, label, var)]
        rows = max(len(self.inputs), len(self.outputs))
        self.height = box_height(rows, len(self.bound) + len(files))
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(0, 0, width, self.height), theme.TILE_R, theme.TILE_R)
        super().__init__(path)
        self.dashed, self.kind, self.off = dashed, kind, off
        self.setBrush(QtGui.QColor(kind_fill(kind) if name else t["tile"]))
        self._set_pen(False)
        if off:
            self.setOpacity(0.5)  # a view switched off
        tooltips, colors, dims = tooltips or {}, colors or {}, dims or {}
        self.setToolTip(tooltips.get("node", ""))
        if name:
            self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                          QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                          QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
            self.setZValue(1)
        x = 12
        self.dot = None
        if level is not False:
            self.dot = QtWidgets.QGraphicsEllipseItem(10, 9, 9, 9, self)
            self.dot.setPen(QtGui.QPen(Qt.PenStyle.NoPen))
            self.dot.setBrush(QtGui.QColor(_status_color(level)))
            x = 25
        self.title = _text(self, title, x, 5, t["ink"], bold=True, width=width - x - 8)
        self.subtitle = _text(self, subtitle, 12, 22, t["ink2"], size=8, width=width - 20)
        port_pen = QtGui.QPen(QtGui.QColor(t["plot_axis"]), 1.2)
        self.input_items, self.output_items, self.param_items = {}, {}, {}
        # what each row's output needs on the right: the input's side gets the rest
        right = [9 + text_width(o, 8) + (3 + text_width(dims[("out", o)], 7.5) if dims.get(("out", o)) else 0)
                 for o in self.outputs]
        for i, port in enumerate(self.inputs):
            y = HEADER_H + i * ROW_H + ROW_H / 2
            circle = QtWidgets.QGraphicsEllipseItem(-4, y - 4, 8, 8, self)
            # filled when connected; open: dashed if optional, red if the node needs it
            color = t["ink2"]
            if ("in", port) in connected:
                circle.setBrush(QtGui.QColor(t["plot_axis"]))
                circle.setPen(port_pen)
            elif port in optional:
                circle.setBrush(self.brush())
                pen = QtGui.QPen(QtGui.QColor(t["ink2"]), 1.2, Qt.PenStyle.DashLine)
                pen.setDashPattern([2.0, 1.6])
                circle.setPen(pen)
                color = t["muted"]
            else:
                circle.setBrush(self.brush())
                circle.setPen(QtGui.QPen(QtGui.QColor(t["roles"]["error"]), 1.8))
                color = t["roles"]["error"]
            circle.setToolTip(tooltips.get(("in", port), ""))
            circle.setCursor(Qt.CursorShape.CrossCursor)
            room = width - 16 - (right[i] if i < len(right) else 0)
            label = _text(self, port, 9, y - 8, color, size=8, width=room - 9)
            label.setToolTip(circle.toolTip())
            self._dims_after(label, dims.get(("in", port)), 9 + label.boundingRect().width() + 3, y - 8,
                             room - 8 - label.boundingRect().width())  # (+4: the measured widths round)
            self.input_items[port] = (circle, label)
        for i, port in enumerate(self.outputs):
            y = HEADER_H + i * ROW_H + ROW_H / 2
            circle = QtWidgets.QGraphicsEllipseItem(width - 4, y - 4, 8, 8, self)
            circle.setBrush(QtGui.QColor(t["plot_axis"]) if ("out", port) in connected else self.brush())
            circle.setPen(port_pen)
            circle.setToolTip(tooltips.get(("out", port), ""))
            circle.setCursor(Qt.CursorShape.CrossCursor)
            label = _text(self, port, 0, y - 8, t["ink2"], size=8, width=width - 30)
            label.setPos(width - 9 - label.boundingRect().width(), y - 8)
            label.setToolTip(circle.toolTip())
            extra = self._dims_after(label, dims.get(("out", port)), 0, y - 8,
                                     right[i] - 8 - label.boundingRect().width())
            if extra is not None:
                extra.setPos(label.pos().x() - extra.boundingRect().width() - 3, y - 8)
            self.output_items[port] = (circle, label)
        row_y = HEADER_H + rows * ROW_H + ROW_H / 2 + (3 if rows else 0)
        for i, (param, label_text, var) in enumerate(self.bound):
            y = row_y + i * ROW_H
            color = colors.get(var, t["accent"])
            mark = QtWidgets.QGraphicsPolygonItem(diamond(0, y), self)
            mark.setBrush(QtGui.QColor(color))
            mark.setPen(QtGui.QPen(Qt.PenStyle.NoPen))
            label = _text(self, f"{label_text} ← {var}", 10, y - 8, color, size=8, width=width - 20)
            mark.setToolTip(f"{label_text} follows the analysis variable {var!r} (the table at the top right). "
                            f"Right-click the node to unbind.")
            label.setToolTip(mark.toolTip())
            self.param_items[param] = (mark, label, y, var)
        self.file_items = []
        for i, (label_text, text, tip, exists, folder) in enumerate(files):
            y = row_y + (len(self.bound) + i) * ROW_H
            color = t["ink2"] if exists else t["roles"]["error"]
            symbol = QtWidgets.QGraphicsPathItem(file_symbol(folder, 10, y - 5), self)
            symbol.setPen(QtGui.QPen(QtGui.QColor(color), 1.1))
            symbol.setBrush(self.brush())
            label = _text(self, text, 26, y - 8, color, size=8, width=width - 34)
            symbol.setToolTip(tip)
            label.setToolTip(tip)
            self.file_items.append((symbol, label))

    def _dims_after(self, label, text, x, y, room):
        """A port's dimensions in muted text next to its name, or None when there is no room."""
        if not text or room < 14:
            return None
        item = _text(self, text, x, y, theme.current()["muted"], size=7.5, width=room)
        item.setToolTip(label.toolTip())
        return item

    def paint(self, painter, option, widget=None):
        option = QtWidgets.QStyleOptionGraphicsItem(option)  # no dotted selection frame: the border shows it
        option.state &= ~QtWidgets.QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def _set_pen(self, selected):
        t = theme.current()
        pen = QtGui.QPen(QtGui.QColor(t["accent"] if selected else t["plot_axis"]), 2 if selected else 1.25)
        if self.dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

    def set_status(self, level, subtitle):
        if self.dot is not None:
            self.dot.setBrush(QtGui.QColor(_status_color(level)))
        font = self.subtitle.font()
        self.subtitle.setText(QtGui.QFontMetrics(font).elidedText(subtitle, Qt.TextElideMode.ElideRight,
                                                                  self.width - 20))

    def highlight_var(self, var):
        """The rows of parameters bound to var in bold with a larger diamond (the variable is selected), the others
        plain."""
        for mark, label, y, bound_var in self.param_items.values():
            on = bound_var == var
            font = label.font()
            font.setBold(on)
            label.setFont(font)
            mark.setPolygon(diamond(0, y, 6.5 if on else 5))

    def port(self, kind, name) -> QtCore.QPointF:
        """The scene position of an input ("in"), output ("out") or bound parameter ("param") port."""
        if kind == "param" and name in self.param_items:
            return self.mapToScene(QtCore.QPointF(0, self.param_items[name][2]))
        ports = self.inputs if kind == "in" else self.outputs
        x = 0 if kind in ("in", "param") else self.width
        y = HEADER_H + ports.index(name) * ROW_H + ROW_H / 2 if name in ports else self.height / 2
        return self.mapToScene(QtCore.QPointF(x, y))

    def port_at(self, scene_pos, kind, reach=10):
        for name in self.inputs if kind == "in" else self.outputs:
            if QtCore.QLineF(self.port(kind, name), scene_pos).length() <= reach:
                return name
        return None

    def mark(self, kind, accepted):
        """While dragging: the accepted ports of a kind in the accent color and larger, the others dimmed."""
        t = theme.current()
        for port, (circle, label) in (self.input_items if kind == "in" else self.output_items).items():
            on = port in accepted
            circle.setOpacity(1 if on else DIM)
            label.setOpacity(circle.opacity())
            if on:
                circle.setBrush(QtGui.QColor(t["accent"]))
                circle.setPen(QtGui.QPen(QtGui.QColor(t["accent"]), 2))
                c = circle.rect().center()
                circle.setRect(QtCore.QRectF(c.x() - 6, c.y() - 6, 12, 12))

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            return _snap(value)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.editor.box_moved(self)
        elif change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._set_pen(bool(value))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if self.name:
            self.editor.go_to(self.name)
        elif self.editor.link_file(self) is not None:
            self.editor.open_linked(self)


# --- the corner overlays: variables and legend ----------------------------------------------------------------------
def _overlay_style(widget, name):
    t = theme.current()
    widget.setObjectName(name)
    widget.setStyleSheet(f"#{name} {{ background: {t['card']}; border: 1px solid {t['border']}; "
                         f"border-radius: {theme.TILE_R}px; }} #{name} QLabel {{ background: transparent; }}")


def _symbol_pixmap(draw, widget, width=22, height=14):
    """A small symbol for a legend or table row, drawn by draw(painter) in a width × height box, sharp on scaled
    screens."""
    ratio = widget.devicePixelRatioF() if widget is not None else 1.0
    pixmap = QtGui.QPixmap(int(width * ratio), int(height * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    draw(painter)
    painter.end()
    return pixmap


class VarTable(QtWidgets.QFrame):
    """The analysis variables, over the canvas's top right corner: a row each with a diamond in the variable's color,
    its name and its value. The parameters bound to a variable carry the same color in the nodes. Click a row to
    select the variable (the inspector shows it), double-click to edit its value, right-click for its menu, drag the
    row onto a node to bind one of its parameters."""

    def __init__(self, dialog):
        super().__init__(dialog.view.viewport())
        self.dialog = dialog
        self.rows, self.selected, self._press = {}, None, None
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(8, 6, 10, 6)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setVisible(False)

    def rebuild(self, variables, colors, texts, tooltips):
        """A row per variable: {var: (diamond, name label, value label)}."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        _overlay_style(self, "varTable")
        t = theme.current()
        self.rows = {}
        title = QtWidgets.QLabel("Variables")
        title.setStyleSheet(f"color: {t['muted']}; font-size: 8pt;")
        self.grid.addWidget(title, 0, 0, 1, 3)
        title.show()
        for i, var in enumerate(variables, start=1):
            color = colors[var]

            def draw(p, c=color):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(c))
                p.drawPolygon(diamond(7, 7, 5))
            mark = QtWidgets.QLabel()
            mark.setPixmap(_symbol_pixmap(draw, self, 14, 14))
            name = QtWidgets.QLabel(var)
            value = QtWidgets.QLabel(texts.get(var, ""))
            for label in (mark, name, value):
                label.setToolTip(tooltips.get(var, ""))
                label.setProperty("var", var)
            for column, label in enumerate((mark, name, value)):
                self.grid.addWidget(label, i, column)
                label.show()  # added to a visible frame: shown only when asked
            self.rows[var] = (mark, name, value)
        self.selected = self.selected if self.selected in self.rows else None
        self._style_rows()
        self.setVisible(bool(self.rows))
        self.adjustSize()

    def _style_rows(self):
        t = theme.current()
        colors = var_colors(self.rows)
        for var, (mark, name, value) in self.rows.items():
            on = var == self.selected
            back = f"background: {t['accent_light']};" if on else ""
            name.setStyleSheet(f"color: {colors[var]}; font-weight: bold; font-size: 9pt; {back}")
            value.setStyleSheet(f"color: {t['ink']}; font-size: 9pt; {back}")

    def set_values(self, texts):
        for var, (_, _, value) in self.rows.items():
            value.setText(texts.get(var, ""))
        self.adjustSize()

    def set_selected(self, var):
        self.selected = var if var in self.rows else None
        self._style_rows()

    def var_at(self, pos):
        child = self.childAt(pos)
        return child.property("var") if child is not None else None

    def value_rect(self, var):
        """The value's place in the canvas's viewport (for editing it in place)."""
        value = self.rows[var][2] if var in self.rows else None
        if value is None:
            return None
        return QtCore.QRect(value.mapTo(self.parentWidget(), QtCore.QPoint(0, 0)), value.size())

    def name_rect(self, var):
        name = self.rows[var][1] if var in self.rows else None
        if name is None:
            return None
        return QtCore.QRect(name.mapTo(self.parentWidget(), QtCore.QPoint(0, 0)), name.size())

    def _scene_pos(self, pos):
        return self.dialog.view.mapToScene(self.mapTo(self.parentWidget(), pos))

    def mousePressEvent(self, event):
        var = self.var_at(event.position().toPoint())
        self._press = (var, event.position().toPoint()) if event.button() == Qt.MouseButton.LeftButton else None
        if var is not None and event.button() == Qt.MouseButton.LeftButton:
            self.dialog.select(var, "var")

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dialog = self.dialog
        if dialog.drag is None and self._press and self._press[0] is not None and \
                (pos - self._press[1]).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
            mark = self.rows[self._press[0]][0]
            start = mark.mapTo(self, mark.rect().center())
            dialog.begin_var_drag(self._press[0], self._scene_pos(start))
        if dialog.drag is not None and dialog.drag.kind == "var":
            dialog.drag_to(self._scene_pos(pos), self.mapTo(self.parentWidget(), pos))

    def mouseReleaseEvent(self, event):
        self._press = None
        if self.dialog.drag is not None and self.dialog.drag.kind == "var":
            self.dialog.drop(self._scene_pos(event.position().toPoint()), event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event):
        var = self.var_at(event.position().toPoint())
        if var is not None:
            self.dialog.edit_var_value(var)

    def contextMenuEvent(self, event):
        var = self.var_at(event.pos())
        if var is not None:
            self.dialog.select(var, "var")
            menu = QtWidgets.QMenu(self)
            self.dialog._var_menu(menu, var)
            menu.exec(event.globalPos())


class Legend(QtWidgets.QFrame):
    """What the symbols mean, over the canvas's bottom right corner."""

    def __init__(self, dialog):
        super().__init__(dialog.view.viewport())
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(8, 6, 10, 6)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(1)

    def rebuild(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        _overlay_style(self, "graphLegend")
        t = theme.current()
        axis, error, ink2 = QtGui.QColor(t["plot_axis"]), QtGui.QColor(t["roles"]["error"]), QtGui.QColor(t["ink2"])

        def port(fill, pen):
            def draw(p):
                p.setPen(pen)
                p.setBrush(fill)
                p.drawEllipse(QtCore.QPointF(11, 7), 4, 4)
            return draw
        dashed = QtGui.QPen(ink2, 1.2, Qt.PenStyle.DashLine)
        dashed.setDashPattern([2.0, 1.6])

        def bound(p):
            p.setPen(Qt.PenStyle.NoPen)
            for i, color in enumerate(list(var_colors(range(3)).values())):
                p.setBrush(QtGui.QColor(color))
                p.drawPolygon(diamond(4 + 7 * i, 7, 3.5))

        def document(p):  # a file and a folder
            p.setPen(QtGui.QPen(ink2, 1.1))
            p.drawPath(file_symbol(False, 1, 2))
            p.drawPath(file_symbol(True, 10, 2))

        def line(style):
            def draw(p):
                pen = QtGui.QPen(axis, 1.4, style)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(1, 7), QtCore.QPointF(21, 7))
            return draw

        def shade(kind):
            def draw(p):
                p.setPen(QtGui.QPen(axis, 1))
                p.setBrush(QtGui.QColor(kind_fill(kind)))
                p.drawRoundedRect(QtCore.QRectF(2, 2, 18, 10), 2.5, 2.5)
            return draw
        entries = ((shade("reader"), "input: reads files"),
                   (shade("compute"), "computation"),
                   (shade("view"), "output: a view, a section of the card"),(port(axis, QtGui.QPen(axis, 1.2)), "input, connected"),
                   (port(QtGui.QColor(t["tile"]), dashed), "optional input, open"),
                   (port(QtGui.QColor(t["tile"]), QtGui.QPen(error, 1.8)), "needed input, missing"),
                   (bound, "parameter set by a variable"),
                   (document, "file or folder the node reads"),
                   (line(Qt.PenStyle.SolidLine), "data"),
                   (line(Qt.PenStyle.DashLine), "from another analysis"))
        for i, (draw, text) in enumerate(entries):
            symbol = QtWidgets.QLabel()
            symbol.setPixmap(_symbol_pixmap(draw, self))
            label = QtWidgets.QLabel(text)
            label.setStyleSheet(f"color: {t['ink2']}; font-size: 8pt;")
            self.grid.addWidget(symbol, i, 0)
            self.grid.addWidget(label, i, 1)
            symbol.show()  # added to a visible frame: shown only when asked
            label.show()
        self.adjustSize()


class _Ghost(QtWidgets.QGraphicsPathItem):
    """A wire's faint copy over the boxes: drawn only, never under the mouse (it covered ports' tool tips)."""

    def shape(self):
        return QtGui.QPainterPath()


class Wire(QtWidgets.QGraphicsPathItem):
    """A data wire from an output to an input: key (node, input, source); style "data" or "link" (from another
    analysis)."""

    def __init__(self, start, start_port, end, end_kind, end_port, key, style="data"):
        super().__init__()
        self.start, self.start_port, self.end, self.end_kind, self.end_port = start, start_port, end, end_kind, end_port
        self.key, self.style = key, style
        self.setZValue(0)  # behind the boxes
        # the same wire faintly over the boxes: where it passes behind one it does not seem to start at that box's
        # ports (a wire from further left came out exactly at another node's outputs)
        self.ghost = _Ghost()
        self.ghost.setZValue(2)
        self.ghost.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.set_highlight(False)
        self.update_path()

    def add_to(self, scene):
        scene.addItem(self)
        scene.addItem(self.ghost)

    def shape(self):
        """Under the mouse only along the line (a curve's own shape is the area it encloses: its tool tip showed over
        empty canvas)."""
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(8)
        return stroker.createStroke(self.path())

    def setOpacity(self, opacity):
        super().setOpacity(opacity)
        self.ghost.setOpacity(opacity * GHOST)

    def setVisible(self, visible):
        super().setVisible(visible)
        self.ghost.setVisible(visible)

    def points(self):
        return self.start.port("out", self.start_port), self.end.port(self.end_kind, self.end_port)

    def update_path(self):
        self.setPath(curve(*self.points()))
        self.ghost.setPath(self.path())

    def set_highlight(self, on):
        t = theme.current()
        pen = QtGui.QPen(QtGui.QColor(t["accent"] if on else t["plot_axis"]), (2.4 if on else 1.4))
        pen.setStyle({"data": Qt.PenStyle.SolidLine, "link": Qt.PenStyle.DashLine}[self.style])
        self.setPen(pen)
        self.ghost.setPen(pen)
        self.setOpacity(self.opacity())  # the ghost follows


def curve(p1, p2, down=False) -> QtGui.QPainterPath:
    """A wire from a port on a right edge (p1; down: from above, a variable dragged from the table) to one on a left
    edge (p2): it leaves and arrives perpendicular to the edges."""
    c = max(40.0, abs(p2.x() - p1.x()) * 0.5)
    path = QtGui.QPainterPath(p1)
    first = QtCore.QPointF(p1.x(), p1.y() + max(30.0, (p2.y() - p1.y()) * 0.6)) if down else \
        QtCore.QPointF(p1.x() + c, p1.y())
    path.cubicTo(first, QtCore.QPointF(p2.x() - (min(c, 60.0) if down else c), p2.y()), p2)
    return path


class Drag:
    """A connection being dragged: from an output (kind "out") or an input ("in") port of node, or a variable from
    the table (kind "var", node: the variable); lifted is (node, input) of a connection picked up at its input. accepted: {node: [ports or
    parameters]} it can be dropped on."""

    def __init__(self, kind, node, port, lifted=None):
        self.kind, self.node, self.port, self.lifted = kind, node, port, lifted
        self.accepted, self.start, self.line = {}, None, None


# --- the canvas ----------------------------------------------------------------------------------------------------
class GraphView(QtWidgets.QGraphicsView):
    """The canvas: a dot grid behind the boxes; panning, zooming, rubber-band selection; presses on ports start
    connections (GraphDialog.begin_drag)."""

    def __init__(self, dialog):
        super().__init__(dialog.scene)
        self.dialog = dialog
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # scroll bars show where else the graph goes (panned by dragging too, Space + drag, the middle button)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._pan, self._space = None, False
        # tool tips of our own: sooner than Qt's, and from the topmost item under the mouse that has one (a box's
        # port, a row, the box), whichever box it is
        self._tip_at = None
        self._tip_timer = QtCore.QTimer(self, singleShot=True, interval=TIP_DELAY, timeout=self._show_tip)

    def viewportEvent(self, event):
        if event.type() == QtCore.QEvent.Type.ToolTip:
            return True  # shown by _show_tip
        return super().viewportEvent(event)

    def _show_tip(self):
        if self._tip_at is None or self.dialog.drag is not None or self._pan is not None:
            return
        scene_pos = self.mapToScene(self._tip_at)
        text = next((item.toolTip() for item in self.scene().items(scene_pos) if item.toolTip()), "")
        if text:
            QtWidgets.QToolTip.showText(self.viewport().mapToGlobal(self._tip_at), text, self.viewport())
        else:
            QtWidgets.QToolTip.hideText()

    def leaveEvent(self, event):
        self._tip_at = None
        self._tip_timer.stop()
        super().leaveEvent(event)

    def drawBackground(self, painter, rect):
        t = theme.current()
        painter.fillRect(rect, QtGui.QColor(t["page"]))
        scale = self.transform().m11()
        step = GRID
        while step * scale < 12:  # zoomed out: fewer dots
            step *= 2
        color = QtGui.QColor(t["muted"])
        color.setAlphaF(0.45 if t["dark"] else 0.4)
        pen = QtGui.QPen(color, 2.2)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        left, top = math.floor(rect.left() / step) * step, math.floor(rect.top() / step) * step
        xs = range(int(left), int(rect.right()) + step, step)
        points = [QtCore.QPointF(x, y) for y in range(int(top), int(rect.bottom()) + step, step) for x in xs]
        painter.drawPoints(points)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.dialog.place_overlays()

    def zoom(self, factor, anchor=None):
        scale = self.transform().m11()
        factor = max(0.25 / scale, min(2.5 / scale, factor))
        self.scale(factor, factor)
        self.dialog.zoom_changed()

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            delta = event.angleDelta().y() or event.angleDelta().x()
            self.zoom(1.0015 ** delta)
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape:
            self.dialog.escape()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().unsetCursor()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton or (self._space and event.button() == Qt.MouseButton.LeftButton):
            self._pan = position
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self.dialog.begin_drag(self.mapToScene(position)):
            return
        if event.button() == Qt.MouseButton.LeftButton and self.dialog.var_table.selected is not None and \
                self.dialog.item_at(self.mapToScene(position)) is None:
            self.dialog.clear_selection()  # a click on empty space: the variable is no longer selected
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        position = event.position().toPoint()
        if self._pan is not None:
            delta = position - self._pan
            self._pan = position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self.dialog.drag is not None:
            self.dialog.drag_to(self.mapToScene(position), position)
            return
        super().mouseMoveEvent(event)
        self.dialog.hover(self.mapToScene(position))
        self._tip_at = position
        self._tip_timer.start()

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor) if self._space else self.viewport().unsetCursor()
            return
        if self.dialog.drag is not None:
            self.dialog.drop(self.mapToScene(event.position().toPoint()), event.globalPosition().toPoint())
            return
        super().mouseReleaseEvent(event)
        self.dialog.commit_moves()

    def mouseDoubleClickEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        if self.dialog.item_at(scene_pos) is None and self.dialog.wire_at(scene_pos) is None:
            self.dialog.search(scene_pos, event.globalPosition().toPoint())
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        self.dialog.context_menu(self.mapToScene(event.pos()), event.globalPos())


class NodeSearch(QtWidgets.QDialog):
    """Add a node by typing part of its name: a popup at the mouse (double-click on empty space, Shift+A). items:
    [(label, data, detail)]; chosen(data) is called with the picked item's data."""

    def __init__(self, parent, items, chosen, title="Add a node"):
        super().__init__(parent, Qt.WindowType.Popup)
        self.items, self.chosen = items, chosen
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QtWidgets.QFrame(objectName="palette")
        outer.addWidget(frame)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.addWidget(QtWidgets.QLabel(title, objectName="section"))
        self.filter = QtWidgets.QLineEdit(placeholderText="Type a node's name …")
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.filter)
        layout.addWidget(self.list)
        self.filter.textChanged.connect(self._fill)
        self.filter.returnPressed.connect(self._pick)
        self.list.itemActivated.connect(lambda item: self._pick())
        self.resize(400, 320)
        self._fill("")

    def _fill(self, text):
        self.list.clear()
        words = text.lower().split()
        for label, data, detail in self.items:
            if all(w in f"{label} {detail}".lower() for w in words):
                item = QtWidgets.QListWidgetItem(f"{label}    {detail}" if detail else label)
                item.setData(Qt.ItemDataRole.UserRole, data)
                if data is None:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                self.list.addItem(item)
        for row in range(self.list.count()):
            if self.list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentRow(row)
                break

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            self.list.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def _pick(self):
        item = self.list.currentItem()
        if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
            data = item.data(Qt.ItemDataRole.UserRole)
            self.close()
            self.chosen(data)


# --- the window ----------------------------------------------------------------------------------------------------
class GraphDialog(QtWidgets.QDialog):
    _open = {}

    @classmethod
    def show_for(cls, card, search=False):
        dialog = cls._open.get(id(card))
        if dialog is None:
            dialog = cls(card)
            cls._open[id(card)] = dialog
            card.destroyed.connect(lambda: cls._open.pop(id(card), None))
            dialog.show()
            dialog.redraw()
            dialog.fit(readable=True)
        else:
            dialog.show()
            dialog.redraw()
        dialog.raise_()
        dialog.activateWindow()
        if search:
            center = dialog.view.viewport().rect().center()
            dialog.search(dialog.view.mapToScene(center), dialog.view.viewport().mapToGlobal(center))
        return dialog

    def __init__(self, card):
        from tesselab.gui.inspector import Inspector
        super().__init__(card, Qt.WindowType.Window)
        self.card = card
        self.resize(1250, 720)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.boxes, self.link_boxes, self.wires = {}, {}, []
        self.drag, self.hovered, self.drawn, self._structure, self._building = None, None, {}, None, False
        self._lit = set()  # the wires highlighted under the mouse
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(self._make_toolbar())
        self.splitter = QtWidgets.QSplitter()
        self.view = GraphView(self)
        self.inspector = Inspector(self)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setSizes([870, 360])
        layout.addWidget(self.splitter, 1)
        settings = getattr(card.window, "settings", None)
        if settings is not None and str(settings.value("graph/panel", True)).lower() == "false":
            self.show_panel(False)  # folded the last time
        self.var_table = VarTable(self)
        self.legend = Legend(self)
        self.hint = QtWidgets.QLabel(self.view.viewport())  # what a drop does, next to the mouse while dragging
        self.hint.setVisible(False)
        self.scene.selectionChanged.connect(self._selection_changed)
        # redrawn later, not inside the handler of a click on an item the redraw deletes; not while a connection or a
        # box is dragged (a result arriving from a worker would cancel it): the drop redraws
        self._redraw_timer = QtCore.QTimer(self, singleShot=True, interval=0, timeout=self._refresh)
        self._message_timer = QtCore.QTimer(self, singleShot=True, interval=6000,
                                            timeout=lambda: self.message.setText(""))
        # methods, not lambdas: Qt disconnects them when this window is deleted (the runner lives on; a lambda
        # would call into the deleted window at every result of any analysis)
        self._values_changed = False
        card.runner.updated.connect(self._results_arrived)
        card.saved_changed.connect(self._edited)  # every edit, undo and redo, also from another card
        self._shortcuts()

    def _make_toolbar(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        add = QtWidgets.QPushButton("Add node", objectName="add", toolTip="Add a node (or double-click empty space, "
                                                                          "Shift+A, to find one by name)")
        self.add_menu = QtWidgets.QMenu(add)
        self.add_menu.aboutToShow.connect(lambda: self.card.fill_add_menu(self.add_menu, self.free_spot(),
                                                                          done=self.select))
        add.setMenu(self.add_menu)
        row.addWidget(add)
        for text, tip, slot in (("Arrange", "Place every node automatically, in columns by dependency", self.arrange),
                                ("Fit", "Show the whole graph (Home)", self.fit)):
            button = QtWidgets.QPushButton(text, toolTip=tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addSpacing(4)
        self.undo_button = theme.icon_button("undo", "Undo (Ctrl+Z)")
        self.redo_button = theme.icon_button("redo", "Redo (Ctrl+Shift+Z or Ctrl+Y)")
        for button, slot in ((self.undo_button, self.card.undo), (self.redo_button, self.card.redo)):
            button.clicked.connect(slot)
            row.addWidget(button)
        self.update_history()
        self.zoom_label = QtWidgets.QLabel(objectName="hint")
        row.addWidget(self.zoom_label)
        self.message = QtWidgets.QLabel(objectName="hint")  # what just happened ("Removed …: Ctrl+Z brings it back")
        row.addWidget(self.message, 1)
        self.file_label = QtWidgets.QLabel(objectName="hint")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.file_label)
        help_button = QtWidgets.QToolButton(text="?", toolTip=_HELP)
        help_button.clicked.connect(lambda: QtWidgets.QToolTip.showText(
            help_button.mapToGlobal(QtCore.QPoint(0, help_button.height())), _HELP, help_button))
        row.addWidget(help_button)
        self.panel_button = theme.icon_button("sidebar", "Show or fold the side panel (Ctrl+B)", checkable=True)
        self.panel_button.setChecked(True)
        self.panel_button.toggled.connect(self.show_panel)
        row.addWidget(self.panel_button)
        return row

    def show_panel(self, shown):
        """The side panel (inspector) shown or folded (Ctrl+B, as the main window's explorer); remembered."""
        self.inspector.setVisible(shown)
        if self.panel_button.isChecked() != shown:
            self.panel_button.setChecked(shown)
        settings = getattr(self.card.window, "settings", None)
        if settings is not None:
            settings.setValue("graph/panel", shown)

    def update_history(self):
        """The undo and redo buttons: usable when the analysis has a step to take back or to do again."""
        self.undo_button.setEnabled(bool(self.card.undo_steps))
        self.redo_button.setEnabled(bool(self.card.redo_steps))

    def _shortcuts(self):
        card = self.card
        for keys, slot in (("Ctrl+Z", card.undo), ("Ctrl+Shift+Z", card.redo), ("Ctrl+Y", card.redo),
                           ("Ctrl+S", card.save), ("Ctrl+B", lambda: self.panel_button.toggle())):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, activated=slot)  # the main window's do not reach here
        for keys, slot in (("Delete", self.delete_selection), ("Backspace", self.delete_selection),
                           ("Ctrl+D", self.duplicate_selection), ("F2", self.rename_selection),
                           ("Home", self.fit), ("Ctrl+A", self.select_all),
                           ("Shift+A", lambda: self.search_here())):
            # the canvas's own: not while typing in the inspector or in a line edit over the canvas (its child)
            QtGui.QShortcut(QtGui.QKeySequence(keys), self.view, activated=slot,
                            context=Qt.ShortcutContext.WidgetShortcut)

    # --- refreshing --------------------------------------------------------------------------------------------
    def _results_arrived(self, analysis):
        if analysis is self.card.analysis:
            self._redraw_later()

    def _edited(self, *_):
        self._values_changed = True
        self.update_history()
        self._redraw_later()

    def _redraw_later(self, *_):
        if self.isVisible():
            self._redraw_timer.start()

    def _refresh(self):
        """After an edit or new results: everything drawn again if the structure changed, else only the status,
        times and values (a dragged λ does not rebuild the drawing)."""
        if self.drag is not None or self.scene.mouseGrabberItem() is not None:
            self._redraw_timer.start(50)  # a box is being moved: after its drop
            return
        if self._structure_key() != self._structure:
            self.redraw()
        else:
            self.update_status()
        # the form's values only after edits: new results must not reset a field being typed in
        self.inspector.refresh(values=self._values_changed)
        self._values_changed = False

    def _structure_key(self):
        analysis, engine = self.card.analysis, self.card.engine
        nodes = {}
        for name, entry in analysis.nodes.items():
            try:
                cls = engine.node_class(analysis, name)
                ports = [list(cls.inputs), cls.output_names()]
            except Exception:
                ports = None
            nodes[name] = [entry.get("node"), entry.get("label"), entry.get("pos"), entry.get("inputs"), ports,
                           entry.get("off"), sorted(k for k, v in entry.get("params", {}).items() if is_var(v))]
        return json.dumps([nodes, sorted(analysis.vars), analysis.doc.get("var_pos"),
                           analysis.doc["view"].get("order"), theme.current().get("name"),
                           analysis.graph.path if analysis.graph else None], sort_keys=True, default=str)

    def _level(self, name, results, running):
        result = results.get(name)
        if name in running:
            return "running"
        if result is None or result.status == "waiting":
            return "waiting"
        return result.level()

    def _subtitle(self, name, result):
        if self.card.analysis.nodes.get(name, {}).get("off"):
            return f"'{name}' · switched off"
        return f"'{name}'" + (f" · {result.seconds * 1000:.0f} ms" if result and result.seconds else "")

    def update_status(self):
        results = self.card.runner.results(self.card.analysis)
        running = self.card.runner.is_running(self.card.analysis)
        for name, box in self.boxes.items():
            box.set_status(self._level(name, results, running), self._subtitle(name, results.get(name)))
        self.var_table.set_values({var: self.var_text(var) for var in self.card.analysis.vars})
        self.place_overlays()
        self._update_title()

    def _update_title(self):
        analysis = self.card.analysis
        self.setWindowTitle(f"Graph: {analysis.label}" + (" •" if self.card.is_unsaved() else ""))
        graph = analysis.graph
        if graph is None:
            self.file_label.setText("graph in the analysis file")
            self.file_label.setToolTip("The graph is saved in the analysis file. Right-click empty space → Graph "
                                       "file, to share it with other analyses.")
        else:
            others = [a.label for a in self.card.engine.sharing(analysis)]
            self.file_label.setText(f"graph file {os.path.basename(graph.path)}"
                                    + (f" · shared with {', '.join(others)}" if others else ""))
            self.file_label.setToolTip(f"{graph.path}\nEvery analysis that uses this file has these nodes and "
                                       f"connections; changes here change them all.")

    def zoom_changed(self):
        self.zoom_label.setText(f"{self.view.transform().m11() * 100:.0f} %")

    def show_message(self, text):
        self.message.setText(text)
        self._message_timer.start()

    # --- drawing -----------------------------------------------------------------------------------------------
    def _tooltips(self, name, cls, result):
        analysis = self.card.analysis
        tips = {"node": f"<b>{cls.name}</b> ('{name}')<br>{cls.help}<br><i>Click to select (its settings appear on "
                        f"the right), double-click to show it in the main window.</i>"}
        for input_name, signature in cls.inputs.items():
            sources = sources_of(analysis.nodes[name].get("inputs", {}).get(input_name))
            source = ", ".join(self.card.source_text(s) for s in sources) or "not connected"
            what = f"{signature.help}<br>" if getattr(signature, "help", "") else ""
            tips[("in", input_name)] = (f"<b>{input_name}</b> ({'optional' if signature.optional else 'needed'}): "
                                        f"{what}takes {signature_text(signature)}<br>from {source}<br><i>Drag to an "
                                        f"output to connect; right-click for the choices.</i>")
        for output in cls.output_names():
            dataset = result.outputs.get(output) if result is not None and result.ok else None
            what = (", ".join(f"{k}({', '.join(map(str, v.dims))})" for k, v in list(dataset.data_vars.items())[:6])
                    if dataset is not None else "no result yet")
            tips[("out", output)] = f"<b>{output}</b>: {what}<br><i>Drag to an input to connect.</i>"
        return tips

    def _dims(self, name, cls, result, results):
        """{("in" or "out", port): text} of the ports' dimensions; sizes from the results."""
        found = {}
        for output in cls.output_names():
            dataset = result.outputs.get(output) if result is not None and result.ok else None
            found[("out", output)] = shape_text(dataset) if dataset is not None else ""
        for input_name, signature in cls.inputs.items():
            dims, _ = port_dims(signature=signature)
            sizes = {}
            for source in sources_of(self.card.analysis.nodes[name].get("inputs", {}).get(input_name)):
                upstream = results.get(source.instance) if not source.file else None
                if upstream is not None and upstream.ok and source.output in upstream.outputs:
                    sizes = dict(upstream.outputs[source.output].sizes)
            found[("in", input_name)] = dims_text(dims, sizes)
        return found

    def redraw(self):
        """Draw everything again (keeping the selection)."""
        keep = self.selection()
        self._building = True
        self.drag, self.hovered, self._lit = None, None, set()
        self.hint.setVisible(False)
        self.scene.clear()
        self.boxes, self.link_boxes, self.wires, self.drawn = {}, {}, [], {}
        analysis, engine = self.card.analysis, self.card.engine
        results = self.card.runner.results(analysis)
        running = self.card.runner.is_running(analysis)
        colors = var_colors(analysis.vars)
        specs, kinds = {}, {}
        for name, entry in analysis.nodes.items():
            group = self.card.groups.get(name)
            title = group.title() if group else self._title(name)
            try:
                cls = engine.node_class(analysis, name)
                kinds[name] = node_kind(cls)
                inputs, outputs = list(cls.inputs), cls.output_names()
                optional = {k for k, signature in cls.inputs.items() if signature.optional}
                dims = self._dims(name, cls, results.get(name), results)
                labels = {p.name: p.label for p in cls.params}
                tips = self._tooltips(name, cls, results.get(name))
                files = file_rows(engine, analysis, name, cls)
            except Exception as error:
                inputs, outputs, labels, optional, files = list(entry.get("inputs", {})), [], {}, set(), []
                dims = {}
                tips = {"node": f"{entry['node']}: {error}"}
            rows = [(k, labels.get(k, k), v["var"]) for k, v in entry.get("params", {}).items() if is_var(v)]
            specs[name] = (title, inputs, outputs, rows, tips, optional, files, dims)
        heights = {name: box_height(max(len(inputs), len(outputs)), len(rows) + len(files)) // GRID
                   for name, (_, inputs, outputs, rows, _, _, files, _) in specs.items()}
        widths = {name: box_width(title, inputs, outputs, dims, rows, files)
                  for name, (title, inputs, outputs, rows, _, _, files, dims) in specs.items()}
        positions = place(analysis, heights, kinds, {n: w // GRID for n, w in widths.items()})
        connected = {}  # instance: {("in", input), ("out", output)} that have a wire
        for name, entry in analysis.nodes.items():
            for input_name, value in entry.get("inputs", {}).items():
                for source in sources_of(value):
                    connected.setdefault(name, set()).add(("in", input_name))
                    if not source.file:
                        connected.setdefault(source.instance, set()).add(("out", source.output))
        for name, (title, inputs, outputs, rows, tips, optional, files, dims) in specs.items():
            box = Box(self, name, title, self._subtitle(name, results.get(name)),
                      self._level(name, results, running), inputs, outputs, rows, tooltips=tips,
                      connected=connected.get(name, ()), optional=optional, files=files, colors=colors, dims=dims,
                      kind=kinds.get(name, "compute"), off=bool(analysis.nodes[name].get("off")), width=widths[name])
            self.scene.addItem(box)
            box.setPos(positions[name][0] * GRID, positions[name][1] * GRID)
            self.boxes[name] = box
            self.drawn[("node", name)] = positions[name]
        # other analyses: a dashed box left of the first node they feed
        for name, input_name, source in analysis.links():
            key = (source.file, source.instance)
            if key not in self.link_boxes:
                self.link_boxes[key] = [name, []]
            if source.output not in self.link_boxes[key][1]:
                self.link_boxes[key][1].append(source.output)
        stacked = {}
        left = min((x for x, _ in positions.values()), default=0) - NODE_W // GRID - 3  # left of every node
        for key, (target, outputs) in list(self.link_boxes.items()):
            box = Box(self, None, os.path.splitext(os.path.basename(key[0]))[0], f"another analysis: {key[1]}", False,
                      (), outputs, dashed=True, tooltips={"node": f"{key[0]}: a live link to another analysis. "
                                                                  f"Double-click to open it."},
                      connected={("out", o) for o in outputs})
            y = positions.get(target, (0, 0))[1]
            while any(abs(y - other) < box.height // GRID + 1 for other in stacked.values()):
                y += 1
            stacked[key] = y
            box.setPos(left * GRID, y * GRID)
            self.scene.addItem(box)
            box.link_key = key
            self.link_boxes[key] = box
        for name, entry in analysis.nodes.items():
            for input_name, value in entry.get("inputs", {}).items():
                for source in sources_of(value):
                    start = self.link_boxes.get((source.file, source.instance)) if source.file else \
                        self.boxes.get(source.instance)
                    if start is None or input_name not in self.boxes[name].inputs:
                        continue
                    wire = Wire(start, source.output, self.boxes[name], "in", input_name, (name, input_name, source),
                                "link" if source.file else "data")
                    wire.setToolTip(f"{self.card.source_text(source)} → {self._title(name)} › {input_name}<br>"
                                    f"<i>Right-click to remove; drag its end off the input to move it.</i>")
                    wire.add_to(self.scene)
                    self.wires.append(wire)
        bound = bound_params(analysis)
        self.var_table.rebuild(list(analysis.vars), colors, {var: self.var_text(var) for var in analysis.vars},
                               {var: self._var_tooltip(var, bound.get(var, [])) for var in analysis.vars})
        self.legend.rebuild()
        for item in self.selectable():
            item.setSelected(self._item_key(item) in keep)
        self.var_table.set_selected(next((name for kind, name in keep if kind == "var"), None))
        self._highlight_var()
        self._structure = self._structure_key()
        self._building = False
        self._update_scene_rect()
        self._update_title()
        self.zoom_changed()
        self.place_overlays()
        self.inspector.show_selection(self.selection())

    def place_overlays(self):
        """The variables table in the canvas's top right corner, the legend in its bottom right corner."""
        viewport, margin = self.view.viewport(), 10
        for widget in (self.var_table, self.legend):
            widget.adjustSize()
        self.var_table.move(viewport.width() - self.var_table.width() - margin, margin)
        self.legend.move(viewport.width() - self.legend.width() - margin,
                         viewport.height() - self.legend.height() - margin)

    def overlay_width(self):
        """How much of the canvas's right side the corner overlays cover."""
        widths = [w.width() for w in (self.var_table, self.legend) if not w.isHidden()]
        return max(widths, default=0) + 20

    def _update_scene_rect(self):
        """The graph and some room around it (to pan and to place nodes there): what the scroll bars span."""
        rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(rect.adjusted(-600, -400, 600, 400))

    def _title(self, name):
        group = self.card.groups.get(name)
        return group.title() if group else name

    def var_param(self, var):
        """The parameter class of the first parameter bound to a variable (its type and unit are the variable's), or
        None."""
        analysis = self.card.analysis
        for name, param in bound_params(analysis).get(var, []):
            try:
                p = P.find(self.card.engine.node_class(analysis, name).params, param)
            except Exception:
                continue
            if p is not None:
                return p
        return None

    def var_text(self, var):
        """The variable's value as shown: in the display unit of its parameters (260 nm rather than 2.6e-07)."""
        value = self.card.analysis.vars.get(var)
        if value is None or value == [] or value == "":
            return "–"  # not set (e.g. no baseline)
        p = self.var_param(var)
        if p is not None:
            try:
                unit = getattr(p, "display", None) or getattr(p, "unit", None) or ""
                return f"{p.to_text(value)} {unit}".strip()
            except Exception:
                pass
        return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else json.dumps(value)

    def _var_tooltip(self, var, targets):
        shown = ", ".join(f"{self._title(n)} › {k}" for n, k in targets) or "nothing yet"
        return (f"<b>{var}</b>: an analysis variable; every parameter bound to it takes its value (one cursor moves "
                f"all of them). Bound: {shown}.<br><i>Double-click to edit the value; drag the row onto a node to "
                f"bind a parameter; right-click for more.</i>")

    # --- selection, moving -------------------------------------------------------------------------------------
    def selectable(self):
        return list(self.boxes.values())

    @staticmethod
    def _item_key(item):
        return ("node", item.name)

    def selection(self):
        """[("node", name) or ("var", name)]: the selected boxes, or the variable selected in the table."""
        found = [self._item_key(item) for item in self.selectable() if item.isSelected()]
        if self.var_table.selected in self.card.analysis.vars:
            found.append(("var", self.var_table.selected))
        return found

    def selected_nodes(self):
        return [name for kind, name in self.selection() if kind == "node"]

    def select(self, name, kind="node"):
        """Select only this node box, or this variable (its row in the table)."""
        if kind == "var":
            self._building = True  # one change of the inspector, not two
            self.scene.clearSelection()
            self._building = False
            if name not in self.var_table.rows:
                self.redraw()
            self.var_table.set_selected(name)
            self._highlight_var()
            self.inspector.show_selection(self.selection())
            return
        if (kind, name) not in {self._item_key(item) for item in self.selectable()}:
            self.redraw()  # just added
        for item in self.selectable():
            item.setSelected(self._item_key(item) == (kind, name))

    def select_all(self):
        for item in self.selectable():
            item.setSelected(True)

    def clear_selection(self):
        self.var_table.set_selected(None)
        self._highlight_var()
        self.scene.clearSelection()
        self.inspector.show_selection(self.selection())

    def _highlight_var(self):
        for box in self.boxes.values():
            box.highlight_var(self.var_table.selected)

    def _selection_changed(self):
        if not self._building:
            if self.scene.selectedItems() and self.var_table.selected is not None:
                self.var_table.set_selected(None)  # a node selected: the variable no longer
                self._highlight_var()
            self.inspector.show_selection(self.selection())

    def box_moved(self, item):
        for wire in self.wires:
            if wire.start is item or wire.end is item:
                wire.update_path()

    def commit_moves(self):
        """After boxes were dragged: their grid positions into the graph, one undo step. The first move stores every
        node's position, so that the automatic layout no longer moves the others."""
        moved = False
        positions = {}
        for item in self.selectable():
            key = self._item_key(item)
            now = (round(item.pos().x() / GRID), round(item.pos().y() / GRID))
            positions[key] = now
            moved = moved or now != self.drawn.get(key)
        if not moved:
            return
        self.set_positions(positions)

    def set_positions(self, positions):
        analysis = self.card.analysis

        def change():
            for (kind, name), (x, y) in positions.items():
                if kind == "node" and name in analysis.nodes:
                    analysis.nodes[name]["pos"] = [x, y]
        self.drawn.update(positions)
        self.card.edit(change, recompute=False, redraw=False)

    def arrange(self):
        """Every node placed automatically again (columns by dependency)."""
        analysis = self.card.analysis
        heights = {name: box.height // GRID for name, box in self.boxes.items()}
        auto = layout_graph(analysis, heights, {name: box.kind for name, box in self.boxes.items()},
                            {name: box.width // GRID for name, box in self.boxes.items()})
        self.set_positions({("node", n): p for n, p in auto.items()})
        self.redraw()
        self.fit()

    def fit(self, readable=False):
        """The whole graph in view, left of the corner overlays (a small graph is not blown up). readable: not smaller
        than READABLE (the boxes' text can be read; when the window opens): a larger graph then shows its top left,
        the readers, and the scroll bars say where the rest is."""
        rect = self.scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        rect = rect.adjusted(-40, -40, 40, 40)
        viewport, reserve = self.view.viewport(), self.overlay_width()
        width, height = max(viewport.width() - reserve, 100), viewport.height()
        whole = min(width / rect.width(), height / rect.height())
        scale = max(READABLE if readable else 0.25, min(1.2, whole))
        self.view.resetTransform()
        self.view.scale(scale, scale)
        if scale <= whole:
            self.view.centerOn(rect.center() + QtCore.QPointF(reserve / 2 / scale, 0))
        else:  # too large to show whole at a readable size: its top left
            self.view.centerOn(QtCore.QPointF(rect.left() + width / 2 / scale + reserve / 2 / scale,
                                              rect.top() + height / 2 / scale))
        self.zoom_changed()

    def free_spot(self):
        """A free place (grid units) for a node added from the toolbar: right of everything, at the top."""
        if not self.boxes:
            return (0, 0)
        right = max(b.pos().x() + b.width for b in self.boxes.values())
        top = min(b.pos().y() for b in self.boxes.values())
        return (round(right / GRID) + 3, round(top / GRID))

    # --- connecting ----------------------------------------------------------------------------------------------
    def accepted(self, instance, output):
        """{node: [inputs]} that accept the output instance.output: the signature fits (Engine.input_choices) and
        connecting it makes no cycle."""
        engine, analysis = self.card.engine, self.card.analysis
        results = self.card.runner.results(analysis)
        ref, found = Source(instance, output).ref(), {}
        for name in self.boxes:
            try:
                inputs = engine.node_class(analysis, name).inputs
            except Exception:
                continue
            for input_name in inputs:
                if any(s.ref() == ref for s, _ in engine.input_choices(analysis, name, input_name, results)):
                    found.setdefault(name, []).append(input_name)
        return found

    def accepted_sources(self, name, input_name):
        """{node: [outputs]} that fit the input name.input_name."""
        found = {}
        try:
            choices = self.card.engine.input_choices(self.card.analysis, name, input_name,
                                                     self.card.runner.results(self.card.analysis))
        except Exception:
            return found
        for source, _ in choices:
            if source.instance in self.boxes:
                found.setdefault(source.instance, []).append(source.output)
        return found

    def accepted_params(self, var):
        """{node: [parameters]} that can be bound to the variable: same kind and unit as the parameters already
        bound to it (any bindable one if none is)."""
        analysis, engine = self.card.analysis, self.card.engine
        first = self.var_param(var)
        found = {}
        for name, entry in analysis.nodes.items():
            try:
                cls = engine.node_class(analysis, name)
            except Exception:
                continue
            for p in cls.params:
                current = entry.get("params", {}).get(p.name)
                if bindable(p) and not (is_var(current) and current["var"] == var) and \
                        (first is None or kind_of(p) == kind_of(first)):
                    found.setdefault(name, []).append(p.name)
        return found

    def _port_under(self, scene_pos, kind):
        for name, box in self.boxes.items():
            port = box.port_at(scene_pos, kind)
            if port is not None:
                return name, port
        return None

    def begin_drag(self, scene_pos):
        """A press on a port starts a drag; True if it did. From an output: a new connection to an input. From a
        connected input: that connection is picked up (moved to another input, or removed on empty space). From an
        unconnected input, or one of several sources or a link: a new connection from an output. (A variable is
        dragged from the table: begin_var_drag.)"""
        if self._redraw_timer.isActive():  # an edit just now: draw it first (its boxes are where they were)
            self._redraw_timer.stop()
            self._refresh()
        lifted = None
        hit = self._port_under(scene_pos, "out")
        if hit is not None:
            kind, (name, port) = "out", hit
        else:
            hit = self._port_under(scene_pos, "in")
            if hit is None:
                return False
            value = self.card.analysis.nodes[hit[0]].get("inputs", {}).get(hit[1])
            source = Source.parse(value) if value and not isinstance(value, list) else None
            if source is not None and not source.file and source.instance in self.boxes:
                kind, name, port, lifted = "out", source.instance, source.output, hit
            else:
                kind, (name, port) = "in", hit
        self.drag = Drag(kind, name, port, lifted)
        self.drag.accepted = self.accepted(name, port) if kind == "out" else self.accepted_sources(name, port)
        self.drag.start = self.boxes[name].port(kind, port)
        self._start_drag_line()
        other_kind = "in" if kind == "out" else "out"
        for other, box in self.boxes.items():
            box.mark(other_kind, self.drag.accepted.get(other, []))
            box.setOpacity(1 if other in self.drag.accepted or other == name else DIM)
        for wire in self.wires:
            wire.setOpacity(DIM)
            if (wire.key[0], wire.key[1]) == lifted:
                wire.setVisible(False)
        return True

    def begin_var_drag(self, var, start):
        """Dragging a variable from the table (start: its row, in the scene) onto a node binds one of its
        parameters."""
        if self._redraw_timer.isActive():
            self._redraw_timer.stop()
            self._refresh()
        self.drag = Drag("var", var, None)
        self.drag.accepted = self.accepted_params(var)
        self.drag.start = start
        self._start_drag_line(var_colors(self.card.analysis.vars).get(var))
        for name, box in self.boxes.items():
            box.setOpacity(1 if name in self.drag.accepted else DIM)
        for wire in self.wires:
            wire.setOpacity(DIM)

    def _start_drag_line(self, color=None):
        line = self.scene.addPath(QtGui.QPainterPath(self.drag.start),
                                  QtGui.QPen(QtGui.QColor(color or theme.current()["accent"]), 2))
        line.setZValue(10)
        self.drag.line = line
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def drag_hint(self, target=None):
        """What a drop does now (target: where it would land), or why nothing is highlighted."""
        d = self.drag
        if d.kind == "var":
            if target is not None:
                params = d.accepted[target]
                return (f"Bind {params[0]} of {self._title(target)} to {d.node}" if len(params) == 1 else
                        f"Drop to choose which parameter of {self._title(target)} follows {d.node}")
            return (f"Drop on a node to bind one of its parameters to {d.node}" if d.accepted else
                    f"No parameter fits {d.node}") + "  ·  Esc cancels"
        what = f"{self._title(d.node)} › {d.port}"
        if target is not None:
            if d.kind == "out":
                text = f"Connect {what} to {self._title(target[0])} › {target[1]}"
            else:
                text = f"Connect {self._title(target[0])} › {target[1]} to {what}"
        elif d.lifted:
            text = "Drop here to remove the connection"
        elif d.kind == "out":
            text = (f"Drop on a highlighted input, or on empty space to add a node that takes {what}"
                    if d.accepted else f"No input takes {what}: drop on empty space to add a node for it")
            result = self.card.runner.results(self.card.analysis).get(d.node)
            if not d.accepted and (result is None or not result.ok):
                text = f"No input takes {what} while {self._title(d.node)} has no result"
        else:
            text = f"Drop on a highlighted output" if d.accepted else f"No output fits {what}"
        return text + "  ·  Esc cancels"

    def _target(self, scene_pos):
        """Where a drop at scene_pos lands: (node, port) of an accepted port (for a variable: the node under it)."""
        if self.drag.kind == "var":
            box = self.box_at(scene_pos)
            return box.name if box is not None and box.name in self.drag.accepted else None
        kind = "in" if self.drag.kind == "out" else "out"
        for name, ports in self.drag.accepted.items():
            port = self.boxes[name].port_at(scene_pos, kind, reach=14)
            if port in ports:
                return name, port
        return None

    def drag_to(self, scene_pos, view_pos=None):
        d = self.drag
        target = self._target(scene_pos)
        if d.kind == "var":  # to the left edge of the node's header
            end = self.boxes[target].mapToScene(QtCore.QPointF(0, HEADER_H / 2)) if target else scene_pos
            d.line.setPath(curve(d.start, end, down=True))
        else:
            end = self.boxes[target[0]].port("in" if d.kind == "out" else "out", target[1]) if target else scene_pos
            start = d.start  # the curve always runs from the output to the input
            if d.kind == "in":
                start, end = end, start
            d.line.setPath(curve(start, end))
        if view_pos is not None:
            self._show_hint(self.drag_hint(target), view_pos)

    def _show_hint(self, text, view_pos):
        t = theme.current()
        self.hint.setStyleSheet(f"background: {t['card']}; color: {t['ink']}; border: 1px solid {t['border']}; "
                                f"border-radius: {theme.CONTROL_R}px; padding: 3px 7px;")
        self.hint.setText(text)
        self.hint.adjustSize()
        x = min(view_pos.x() + 16, self.view.viewport().width() - self.hint.width() - 4)
        self.hint.move(max(4, x), view_pos.y() + 18)
        self.hint.setVisible(True)
        self.hint.raise_()

    def cancel_drag(self):
        if self.drag is not None:
            self.view.viewport().unsetCursor()
            self.hint.setVisible(False)
            self.redraw()

    def escape(self):
        if self.drag is not None:
            self.cancel_drag()
        else:
            self.clear_selection()

    def drop(self, scene_pos, global_pos=None):
        d = self.drag
        target = self._target(scene_pos)
        self.cancel_drag()
        nodes = self.card.analysis.nodes
        if d.kind == "var":
            if target is not None and target in nodes and d.node in self.card.analysis.vars:
                params = d.accepted[target]
                if len(params) == 1:
                    self.card.bind(target, params[0], d.node)
                else:
                    menu = QtWidgets.QMenu(self)
                    labels = self._param_labels(target)
                    for param in params:
                        menu.addAction(f"{labels.get(param, param)}", lambda k=param: self.card.bind(target, k, d.node))
                    menu.exec(global_pos or QtGui.QCursor.pos())
            return
        if d.node not in nodes or (target is not None and target[0] not in nodes) or \
                (d.lifted is not None and d.lifted[0] not in nodes):
            return  # a node was removed while dragging (in another card, or by an undo)
        if d.kind == "out" and target is not None:
            self.connect_ports(d.node, d.port, *target, lifted=d.lifted)
        elif d.kind == "in" and target is not None:
            self.connect_ports(*target, d.node, d.port)
        elif d.lifted is not None and target is None:
            self.card.set_input(*d.lifted, None)
        elif d.kind == "out" and target is None and self.item_at(scene_pos) is None:
            self.search(scene_pos, global_pos or QtGui.QCursor.pos(), source=(d.node, d.port))

    def _param_labels(self, name):
        try:
            return {p.name: p.label for p in self.card.engine.node_class(self.card.analysis, name).params}
        except Exception:
            return {}

    def connect_ports(self, instance, output, node, input_name, lifted=None):
        """Connect instance.output to node.input_name (added to the list of a many=True input); a picked-up
        connection leaves its old input in the same undo step."""
        analysis = self.card.analysis
        source = f"{instance}.{output}"
        if lifted == (node, input_name):
            return  # put back where it was
        if self.card.engine.node_class(analysis, node).inputs[input_name].many:
            current = sources_of(analysis.nodes[node].get("inputs", {}).get(input_name))
            if any(s.ref() == source for s in current):
                return
            source = [s.to_json() for s in current] + [source]
        if lifted is None:
            self.card.set_input(node, input_name, source)
            return

        def change():
            analysis.set_input(*lifted, None)
            analysis.set_input(node, input_name, source)
        try:
            self.card.edit(change)
        except CycleError as error:
            message_box(self, "Not possible", str(error), icon="warning")

    def disconnect_input(self, name, input_name, source):
        value = self.card.analysis.nodes[name]["inputs"][input_name]
        if isinstance(value, list):  # many=True: only this source
            self.card.set_input(name, input_name, [s.to_json() for s in sources_of(value) if s.ref() != source.ref()])
        else:
            self.card.set_input(name, input_name, None)

    # --- finding things under the mouse ------------------------------------------------------------------------
    def box_at(self, scene_pos):
        return next((b for b in self.boxes.values() if b.sceneBoundingRect().contains(scene_pos)), None)

    def item_at(self, scene_pos):
        """The box (node or linked analysis) under a scene position, or None."""
        for item in [*self.boxes.values(), *self.link_boxes.values()]:
            if item.sceneBoundingRect().adjusted(-6, -6, 6, 6).contains(scene_pos):
                return item
        return None

    def wire_at(self, scene_pos):
        """The wire under a scene position, or None; inside a box the box counts, not a wire that crosses it."""
        if any(b.sceneBoundingRect().contains(scene_pos)
               for b in [*self.boxes.values(), *self.link_boxes.values()]):
            return None
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(10)
        return next((w for w in self.wires if w.isVisible() and stroker.createStroke(w.path()).contains(scene_pos)),
                    None)

    def edge_at(self, scene_pos):
        """(node, input, source) of the data connection under a scene position, or None."""
        wire = self.wire_at(scene_pos)
        return wire.key if wire is not None else None

    def hover(self, scene_pos):
        """The wire under the mouse is highlighted (the one a right-click acts on); over a box, all of its wires."""
        wire = self.wire_at(scene_pos)
        box = self.item_at(scene_pos) if wire is None else None
        lit = {wire} if wire is not None else {w for w in self.wires if box is not None and box in (w.start, w.end)}
        self.hovered = wire
        if lit == self._lit:
            return
        for w in self._lit - lit:
            w.set_highlight(False)
        for w in lit - self._lit:
            w.set_highlight(True)
        self._lit = lit

    def link_file(self, box):
        return getattr(box, "link_key", (None,))[0]

    def open_linked(self, box):
        path = os.path.normpath(os.path.join(self.card.engine.link_folder(self.card.analysis), self.link_file(box)))
        self.card.window.open_analysis(path)

    def go_to(self, name):
        """The node's section in the main window (opened and scrolled to), the main window in front."""
        self.card.scroll_to(name)
        self.card.window.raise_()
        self.card.window.activateWindow()

    # --- menus ---------------------------------------------------------------------------------------------------
    def context_menu(self, scene_pos, global_pos):
        menu = self.menu_at(scene_pos, global_pos)
        if menu is not None:
            menu.exec(global_pos)

    def menu_at(self, scene_pos, global_pos=None):
        """The right-click menu at a scene position: of a wire, a node, an input port, a linked analysis, or the
        canvas's own on empty space (a variable's is the table's)."""
        menu = QtWidgets.QMenu(self)
        global_pos = global_pos or QtGui.QCursor.pos()
        port = self._port_under(scene_pos, "in")
        if port is not None:
            self.fill_input_menu(menu, *port)
            return menu
        wire = self.wire_at(scene_pos)
        if wire is not None:
            self._wire_menu(menu, wire)
            return menu
        item = self.item_at(scene_pos)
        if isinstance(item, Box) and item.name:
            if not item.isSelected():
                self.select(item.name)
            self._node_menu(menu, item.name)
        elif isinstance(item, Box):
            menu.addAction("Open this analysis", lambda: self.open_linked(item))
        else:
            self._canvas_menu(menu, scene_pos, global_pos)
        return menu

    def _wire_menu(self, menu, wire):
        name, input_name, source = wire.key
        menu.addAction("Remove connection", lambda: self.disconnect_input(name, input_name, source))
        self._binding_choices(menu, name, input_name, source)

    def _binding_choices(self, menu, name, input_name, source):
        """Where several variables of the source fit one of the input's (two wavelength axes, say): which to use."""
        results = self.card.runner.results(self.card.analysis)
        upstream = results.get(source.instance) if not source.file else None
        if upstream is None or not upstream.ok or source.output not in upstream.outputs:
            return
        try:
            signature = self.card.engine.node_class(self.card.analysis, name).inputs[input_name]
        except Exception:
            return
        options = bindings(signature, upstream.outputs[source.output])
        for var in signature.variables:
            values = []
            for option in options:
                if option.get(var) not in values:
                    values.append(option.get(var))
            if len(values) > 1:
                sub = menu.addMenu(f"Use as {var}")
                chosen = source.bind.get(var, values[0])
                for value in values:
                    action = sub.addAction(str(value) if value is not None else "(none)",
                                           lambda v=value, var=var: self.card.set_binding(name, input_name, var, v))
                    action.setCheckable(True)
                    action.setChecked(value == chosen)

    def fill_input_menu(self, menu, name, input_name):
        """The choices for one input: the outputs that fit (as the old dropdown had), not connected, a link."""
        menu.addSection(f"{self._title(name)} › {input_name}")
        value = self.card.analysis.nodes[name].get("inputs", {}).get(input_name)
        current = {s.ref() for s in sources_of(value)}
        try:
            choices = self.card.engine.input_choices(self.card.analysis, name, input_name,
                                                     self.card.runner.results(self.card.analysis))
        except Exception:
            choices = []
        for source, _ in choices:
            action = menu.addAction(self.card.source_text(source),
                                    lambda s=source: self.connect_ports(s.instance, s.output, name, input_name))
            action.setCheckable(True)
            action.setChecked(source.ref() in current)
        if not choices:
            menu.addAction("No output fits yet").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Not connected", lambda: self.card.set_input(name, input_name, None)).setEnabled(bool(current))
        menu.addAction("Link to another analysis…", lambda: self.card.choose_link(name, input_name))

    def _node_menu(self, menu, name):
        selected = self.selected_nodes()
        if len(selected) > 1:
            menu.addAction(f"Duplicate {len(selected)} nodes", self.duplicate_selection)
            menu.addAction(f"Remove {len(selected)} nodes", self.delete_selection)
            return
        card, analysis = self.card, self.card.analysis
        menu.addAction("Show in the main window", lambda: self.go_to(name))
        menu.addAction("Rename…", lambda: self.rename(name))
        menu.addAction("Duplicate", self.duplicate_selection)
        try:
            cls = card.engine.node_class(analysis, name)
        except Exception:
            cls = None
        if cls is not None and node_kind(cls) == "view":
            off = bool(analysis.nodes[name].get("off"))
            menu.addAction("Switch on (show in the card)" if off else "Switch off (hide in the card)",
                           lambda: self.switch_view(name))
        if cls is not None:
            if cls.run_on == "button" or cls.actions:
                menu.addSeparator()
            if cls.run_on == "button":
                menu.addAction("Run", lambda: card.run_node(name))
            for action in cls.actions:
                menu.addAction(action.label, lambda a=action: card.run_action(name, a))
            if cls.inputs:
                inputs = menu.addMenu("Inputs")
                for input_name in cls.inputs:
                    self.fill_input_menu(inputs.addMenu(input_name), name, input_name)
            params = analysis.nodes[name].get("params", {})
            free = [p for p in cls.params if bindable(p) and not is_var(params.get(p.name))]
            if free:
                bind = menu.addMenu("Bind to a variable")
                for p in free:
                    sub = bind.addMenu(p.label)
                    for var in analysis.vars:
                        first = self.var_param(var)
                        if first is None or kind_of(first) == kind_of(p):
                            sub.addAction(var, lambda v=var, k=p.name: card.bind(name, k, v))
                    sub.addAction("New variable…", lambda k=p: self.new_variable(name, k))
            bound = [(k, v["var"]) for k, v in params.items() if is_var(v)]
            if bound:
                unbind = menu.addMenu("Unbind")
                for key, var in bound:
                    unbind.addAction(f"{key} from {var}", lambda k=key: card.bind(name, k, None))
        menu.addSeparator()
        menu.addAction("Remove", self.delete_selection)

    def _var_menu(self, menu, var):
        menu.addAction("Edit the value…", lambda: self.edit_var_value(var))
        menu.addAction("Rename…", lambda: self.rename_var(var))
        targets = self.accepted_params(var)
        if targets:
            bind = menu.addMenu("Bind")
            for name, params in targets.items():
                labels = self._param_labels(name)
                for param in params:
                    bind.addAction(f"{self._title(name)} › {labels.get(param, param)}",
                                   lambda n=name, k=param: self.card.bind(n, k, var))
        menu.addSeparator()
        menu.addAction("Remove the variable", lambda: self.remove_var(var))

    def _canvas_menu(self, menu, scene_pos, global_pos):
        spot = (round(scene_pos.x() / GRID), round(scene_pos.y() / GRID))
        add = menu.addMenu("Add node")
        self.card.fill_add_menu(add, spot, done=self.select)
        menu.addAction("Add node by name…\tShift+A", lambda: self.search(scene_pos, global_pos))
        new_var = menu.addMenu("New variable from")
        analysis = self.card.analysis
        for name in analysis.order():
            try:
                cls = self.card.engine.node_class(analysis, name)
            except Exception:
                continue
            params = analysis.nodes[name].get("params", {})
            free = [p for p in cls.params if bindable(p) and not is_var(params.get(p.name))]
            if free:
                sub = new_var.addMenu(self._title(name))
                for p in free:
                    sub.addAction(p.label, lambda n=name, k=p: self.new_variable(n, k))
        new_var.setEnabled(not new_var.isEmpty())
        menu.addSeparator()
        menu.addAction("Select all\tCtrl+A", self.select_all)
        menu.addAction("Arrange automatically", self.arrange)
        menu.addAction("Fit to the window\tHome", self.fit)
        menu.addSeparator()
        self.fill_graph_file_menu(menu.addMenu("Graph file"))
        menu.addSeparator()
        menu.addAction("Undo\tCtrl+Z", self.card.undo).setEnabled(bool(self.card.undo_steps))
        menu.addAction("Redo\tCtrl+Shift+Z", self.card.redo).setEnabled(bool(self.card.redo_steps))

    def fill_graph_file_menu(self, menu):
        graph = self.card.analysis.graph
        if graph is None:
            menu.addAction("The graph is in the analysis file").setEnabled(False)
            menu.addAction("Share it in a graph file of its own…", self.card.share_graph)
        else:
            others = [a.label for a in self.card.engine.sharing(self.card.analysis)]
            menu.addAction(f"{os.path.basename(graph.path)}" + (f" (shared with {', '.join(others)})"
                                                                 if others else "")).setEnabled(False)
            menu.addAction("Use a separate copy of this graph…", self.card.share_graph)
            menu.addAction("Open the graph file", lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(graph.path)))
        menu.addAction("Use another graph file…", self.card.choose_graph_file)

    # --- editing -------------------------------------------------------------------------------------------------
    def search_here(self):
        position = self.view.mapFromGlobal(QtGui.QCursor.pos())
        if not self.view.viewport().rect().contains(position):
            position = self.view.viewport().rect().center()
        self.search(self.view.mapToScene(position), self.view.viewport().mapToGlobal(position))

    def search(self, scene_pos, global_pos, source=None):
        """Add a node by name at scene_pos; with source (instance, output) only the nodes that take that output,
        connected to it."""
        spot = (round(scene_pos.x() / GRID), round(scene_pos.y() / GRID))
        items = []
        if source is None:
            for label, full_id, module, kind in self.card.addable():
                detail = {"fits": f"{module} · fits", "reads": f"{module} · reads data"}.get(kind, module)
                items.append((label, None if kind == "broken" else (full_id, None), detail))
            title = "Add a node"
        else:
            for label, full_id, input_name in self.takers(*source):
                items.append((f"{label} › {input_name}", (full_id, input_name), ""))
            title = f"Add a node that takes {self._title(source[0])} › {source[1]}"
            if not items:
                self.show_message(f"No node takes {self._title(source[0])} › {source[1]}.")
                return

        def chosen(data):
            full_id, input_name = data
            inputs = {input_name: f"{source[0]}.{source[1]}"} if source else None
            name = self.card.add_node(full_id, pos=spot, inputs=inputs, scroll=False)
            if name:
                self.select(name)
        popup = NodeSearch(self, items, chosen, title)
        popup.move(global_pos - QtCore.QPoint(20, 20))
        popup.show()
        popup.filter.setFocus()
        self._search = popup  # for tests
        return popup

    def takers(self, instance, output):
        """[(label, full id, input)] of the node classes with an input that takes instance.output."""
        result = self.card.runner.results(self.card.analysis).get(instance)
        dataset = result.outputs.get(output) if result is not None and result.ok else None
        found = []
        registry = self.card.engine.registry
        for full_id, info in sorted(registry.nodes().items(), key=lambda kv: kv[1].module.name):
            try:
                cls = registry.node_class(full_id)
            except Exception:
                continue
            for input_name, signature in cls.inputs.items():
                if dataset is None or bindings(signature, dataset):
                    found.append((cls.name or info.id, full_id, input_name))
        return found

    def delete_selection(self):
        """Remove the selected nodes and variables, one undo step, without asking (Ctrl+Z brings them back)."""
        selection = self.selection()
        if not selection:
            return
        analysis = self.card.analysis

        def change():
            for kind, name in selection:
                if kind == "node" and name in analysis.nodes:
                    analysis.remove(name)
                elif kind == "var" and name in analysis.vars:
                    self._unbind_all(name)
        titles = [self._title(n) if k == "node" else n for k, n in selection]
        self.card.edit(change)
        self.show_message(f"Removed {', '.join(titles)}: Ctrl+Z brings {'it' if len(titles) == 1 else 'them'} back.")

    def _unbind_all(self, var):
        """Inside an edit: remove a variable; its parameters keep its value."""
        analysis = self.card.analysis
        value = analysis.vars.pop(var)
        analysis.doc.get("var_pos", {}).pop(var, None)
        for entry in analysis.nodes.values():
            for key, current in entry.get("params", {}).items():
                if current == {"var": var}:
                    entry["params"][key] = copy.deepcopy(value)

    def switch_view(self, name):
        """A view off (its section is not shown in the card; it stays in the graph) or on again: one undo step,
        saved in the graph."""
        entry = self.card.analysis.nodes[name]

        def change():
            if entry.get("off"):
                entry.pop("off")
            else:
                entry["off"] = True
        self.card.edit(change, recompute=False)
        self.card.rebuild_all()
        self.show_message(f"{self._title(name)} is switched {'off' if entry.get('off') else 'on'}.")

    def remove_var(self, var):
        self.card.edit(lambda: self._unbind_all(var))
        self.show_message(f"Removed the variable {var}: its parameters keep its value (Ctrl+Z brings it back).")

    def duplicate_selection(self):
        """Copies of the selected nodes, two grid steps lower right; connections among them are copied too."""
        names = self.selected_nodes()
        if not names:
            return
        analysis = self.card.analysis
        made = {}

        def change():
            for name in names:
                entry = analysis.nodes[name]
                x, y = entry.get("pos") or self.drawn.get(("node", name), (0, 0))
                made[name] = analysis.add(entry["node"], name=name, params=copy.deepcopy(entry.get("params", {})),
                                          version=entry.get("version", ""), schema=entry.get("schema", 1),
                                          label=entry.get("label"), pos=(x + 2, y + 2))
                for key in analysis.doc.get("own", {}).get(name, []):
                    analysis.set_own(made[name], key)
            for name in names:
                for key, value in analysis.nodes[name].get("inputs", {}).items():
                    copies = [Source(made.get(s.instance, s.instance), s.output, s.file, s.bind).to_json()
                              if not s.file else s.to_json() for s in sources_of(value)]
                    analysis.nodes[made[name]]["inputs"][key] = (copies if isinstance(value, list) else
                                                                 copies[0] if copies else None)
        self.card.edit(change)
        for item in self.selectable():
            item.setSelected(self._item_key(item) in {("node", m) for m in made.values()})

    def rename_selection(self):
        selection = self.selection()
        if len(selection) == 1:
            kind, name = selection[0]
            self.rename(name) if kind == "node" else self.rename_var(name)

    def rename(self, name):
        entry = self.card.analysis.nodes[name]
        box = self.boxes.get(name)
        rect = box.mapRectToScene(box.title.boundingRect().translated(box.title.pos())) if box else None

        def done(text):
            text = text.strip()
            self.card.edit(lambda: entry.__setitem__("label", text) if text else entry.pop("label", None),
                           recompute=False)
            group = self.card.groups.get(name)
            if group is not None:
                group.update_heading()
        self.inline_edit(rect, entry.get("label", ""), done, "Label (empty: the node's name)")

    def rename_var(self, var):
        rect = self.var_table.name_rect(var)

        def done(text):
            new = text.strip()
            analysis = self.card.analysis
            if not new or new == var:
                return
            if new in analysis.vars or not new.replace("_", "a").isalnum() or new[0].isdigit():
                self.show_message(f"{new!r} is taken or not a name (letters, digits and _).")
                return

            def change():
                analysis.vars[new] = analysis.vars.pop(var)
                for entry in analysis.nodes.values():
                    for key, value in entry.get("params", {}).items():
                        if value == {"var": var}:
                            entry["params"][key] = {"var": new}
            self.card.edit(change, recompute=False)
            self.var_table.set_selected(new)
        self.inline_edit(rect, var, done, "Name of the variable", in_view=True)

    def edit_var_value(self, var):
        rect = self.var_table.value_rect(var)
        p = self.var_param(var)
        value = self.card.analysis.vars.get(var)
        text = p.to_text(value) if p is not None else json.dumps(value)

        def done(typed):
            try:
                new = p.from_text(typed) if p is not None else json.loads(typed)
            except (ValueError, TypeError) as error:
                self.show_message(str(error))
                return
            self.card.set_var(var, new)
        unit = (getattr(p, "display", None) or getattr(p, "unit", None) or "") if p is not None else ""
        self.inline_edit(rect, text, done, f"Value of {var}" + (f" in {unit}" if unit else ""), in_view=True)

    def new_variable(self, name, p):
        """A new variable bound to parameter p of node `name`, with its value."""
        analysis = self.card.analysis
        base, i, var = p.name, 2, p.name
        while var in analysis.vars:
            var, i = f"{base}{i}", i + 1
        self.card.bind(name, p.name, var)
        self.select(var, "var")
        self.rename_var(var)

    def inline_edit(self, scene_rect, text, done, placeholder="", in_view=False):
        """A line edit over a place of the drawing (in_view: scene_rect is a QRect in the canvas's viewport, e.g. a row
        of the variables table), else a small dialog: Enter takes it, Esc or leaving cancels."""
        if scene_rect is None or not self.isVisible():
            typed, ok = QtWidgets.QInputDialog.getText(self, placeholder or "Edit", placeholder, text=text)
            if ok:
                done(typed)
            return
        editor = QtWidgets.QLineEdit(text, self.view.viewport(), placeholderText=placeholder)
        rect = scene_rect if in_view else self.view.mapFromScene(scene_rect).boundingRect()
        editor.setGeometry(rect.x() - 4, rect.y() - 3, max(rect.width() + 60, 150), rect.height() + 8)
        editor.selectAll()
        editor.show()
        editor.setFocus()
        state = {"done": False}

        def finish(take):
            if state["done"]:
                return
            state["done"] = True
            typed = editor.text()
            editor.deleteLater()
            self.view.setFocus()
            if take:
                done(typed)
        editor.returnPressed.connect(lambda: finish(True))
        editor.editingFinished.connect(lambda: finish(False))
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), editor, activated=lambda: finish(False),
                        context=Qt.ShortcutContext.WidgetShortcut)
        self._inline = editor  # for tests

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape()  # instead of closing the window
            return
        super().keyPressEvent(event)


_HELP = """<b>The graph</b><br>
Drag a node to move it (it snaps to the grid). Drag on empty space to select several; pan with the middle mouse
button or Space + drag; zoom with Ctrl or Shift + wheel.<br><br>
<b>Connect</b>: drag from an output (right edge) to an input (left edge), or back. Drag a connection off its input
to move it, or onto empty space to remove it. Drop a new connection on empty space to add a node that takes it.
Right-click an input for its choices.<br><br>
<b>Variables</b> (the table at the top right): every parameter bound to one takes its value; the bound parameters
are rows in the variable's color in the nodes. Drag a variable's row onto a node to bind a parameter; double-click
to edit the value; right-click for more.<br><br>
<b>Inputs</b>: filled when connected; dashed when optional and open; red when needed and missing. A file symbol
names a file the node reads (red: not found). The legend at the bottom right repeats this.<br><br>
<b>Kinds</b>: bluish boxes read files (input), plain ones compute, greenish ones are views (output): each view is a
section of the card; right-click a view to switch it off. Select a reader or compute node to see a preview of its
outputs on the right.<br><br>
<b>Lines</b>: solid data, dashed from another analysis. <b>Dot</b>: green fine, orange warning, red error, blue
running, grey waiting.<br><br>
<b>Keys</b>: double-click empty space or Shift+A adds a node by name · Delete removes · Ctrl+D duplicates · F2
renames · Home fits · Ctrl+A selects all · Esc cancels · Ctrl+B folds the side panel · Ctrl+Z undoes · Ctrl+Shift+Z
or Ctrl+Y redoes · Ctrl+S
saves.<br>
Double-click a node to show it in the main window."""
