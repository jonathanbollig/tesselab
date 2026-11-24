"""
Drawing plot descriptions (plotspec.PlotSpec) with pyqtgraph, and what makes the plots pleasant to use. Taken from
the kufner viewer, where each of these was measured or chosen by the user:

- PlotView keeps a picture of itself and redraws only when its scene changes: on Wayland with a fractionally scaled
  screen Qt cannot move pixels when scrolling, so every scroll step redrew all visible plots (~145 ms per step; ~22 ms
  with the picture). render() needs a target in logical units, otherwise the picture is 1.5× too large.
- The mouse wheel scrolls the page; Shift + wheel zooms; Shift + drag zooms into a rectangle; dragging pans.
- Tick labels as matplotlib writes them (same decimals, a factor ×10ⁿ below 1e-5 or from 1e6).
- Lines are PlotCurveItems (5× faster to create than PlotDataItems), 1 px pens (Qt's fast path).

Units and scales: every value is converted to its axis's display unit and scale before it is drawn (Transform), so
axes in nm, cm⁻¹ or eV, logarithmic or linear-then-logarithmic axes all work the same; the coordinates under the
mouse and dragged cursors are converted back to the stored units.
"""
import dataclasses
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab import units
from tesselab.colors import black_red
from tesselab.gui import theme
from tesselab.plotspec import (Band, Heatmap, HLine, Line, Lines, Points, Region, Text, VLine)

PLOT_HEIGHT = 320
LEFT_AXIS_WIDTH = 58    # in plots with several panels, so that their data areas line up
RIGHT_AXIS_WIDTH = 58
COLORBAR_AXIS_WIDTH = 50
LONG_LINE = 20000       # lines with more points are drawn min/max per pixel
COLORBAR_WIDTH = 88     # room a color bar takes: the bar (12), its axis and its label
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
DASH = {"solid": Qt.PenStyle.SolidLine, "dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine,
        "dashdot": Qt.PenStyle.DashDotLine}


class Transform:
    """Stored value -> screen coordinate along one axis: the display unit, then the scale."""

    def __init__(self, axis, display=None):
        self.axis = axis
        self.unit = axis.unit
        self.display = display if display is not None else axis.displays[0]
        self.scale = axis.scale
        self.threshold = None
        if self.scale == "linlog":
            self.threshold = abs(float(units.to_display(axis.linthresh, self.unit, self.display)))

    def to_display(self, values):
        return units.to_display(np.asarray(values, dtype=float), self.unit, self.display) if self.display \
            else np.asarray(values, dtype=float)

    def forward(self, values):
        v = self.to_display(values)
        if self.scale == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(v > 0, np.log10(np.where(v > 0, v, 1)), np.nan)
        if self.scale == "linlog":
            t = self.threshold
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(v) <= t, v / t, np.sign(v) * (1 + np.log10(np.abs(v) / t)))
        return v

    def inverse_display(self, screen):
        """Screen coordinate -> value in the display unit."""
        s = np.asarray(screen, dtype=float)
        if self.scale == "log":
            return 10 ** s
        if self.scale == "linlog":
            t = self.threshold
            return np.where(np.abs(s) <= 1, s * t, np.sign(s) * t * 10 ** (np.abs(s) - 1))
        return s

    def inverse(self, screen):
        """Screen coordinate -> stored value."""
        shown = self.inverse_display(screen)
        return units.from_display(shown, self.unit, self.display) if self.display else shown

    def label(self):
        return units.label(self.axis.label, self.unit, self.display)


def format_coordinate(value, pixel) -> str:
    """A coordinate under the mouse, as precise as one pixel (as matplotlib shows it)."""
    if not math.isfinite(value):
        return "–"
    if value == 0 or not pixel or not math.isfinite(pixel):
        return f"{value:g}"
    decimals = max(0, -math.floor(math.log10(abs(pixel))))
    if decimals > 6 or abs(value) >= 1e6:
        digits = max(0, math.floor(math.log10(abs(value))) + decimals)
        return f"{value:.{digits}e}".replace("-", "−")
    return f"{value:.{decimals}f}".replace("-", "−")


class TickAxis(pg.AxisItem):
    """Tick labels for the three scales. linear: as matplotlib (same decimals, ×10ⁿ for very small or large
    values); log: powers of ten; linlog: 0, ±threshold, ±10·threshold, ..."""
    transform = None

    def updateAutoSIPrefix(self):
        if self.transform is not None and self.transform.scale != "linear":
            self.autoSIPrefixScale = 1.0
        else:
            largest = max(abs(self.range[0]), abs(self.range[1])) * self.scale
            exponent = int(np.floor(np.log10(largest))) if np.isfinite(largest) and largest > 0 else 0
            self.autoSIPrefixScale = 1.0 if -5 < exponent < 6 else 10.0 ** -exponent
        self._updateLabel()

    def labelString(self):
        text = self.labelText
        if self.autoSIPrefixScale != 1.0:
            text += "  ×10" + str(round(-np.log10(self.autoSIPrefixScale))).translate(SUPERSCRIPT)
        style = ";".join(f"{key}: {value}" for key, value in self.labelStyle.items())
        return f"<span style='{style}'>{text}</span>"

    def tickValues(self, minVal, maxVal, size):
        t = self.transform
        if t is None or t.scale == "linear":
            return super().tickValues(minVal, maxVal, size)
        if t.scale == "log":
            low, high = math.floor(minVal), math.ceil(maxVal)
            if high - low > 12:
                step = math.ceil((high - low) / 8)
                return [(step, [float(e) for e in range(low, high + 1, step)])]
            minor = [e + math.log10(m) for e in range(low, high + 1) for m in range(2, 10)
                     if minVal <= e + math.log10(m) <= maxVal] if high - low <= 4 else []
            return [(1.0, [float(e) for e in range(low, high + 1)]), (0.1, minor)]
        # linlog: 0, ±1 (the threshold), ±2, ±3 ... in screen units (each a power of ten beyond the threshold)
        low, high = math.floor(minVal), math.ceil(maxVal)
        major = [float(v) for v in range(low, high + 1)]
        minor = []
        for v in range(low, high + 1):
            if abs(v) >= 1:
                sign = 1 if v > 0 else -1
                minor += [sign * (abs(v) + math.log10(m)) for m in range(2, 10)]
        if high - low <= 2:
            minor += [x / 5 for x in range(-5, 6)]
        return [(1.0, major), (0.1, [m for m in minor if minVal <= m <= maxVal])]

    def tickStrings(self, values, scale, spacing):
        t = self.transform
        if t is None or t.scale == "linear":
            decimals = max(0, int(np.ceil(-np.log10(spacing * scale)))) if spacing > 0 else 0
            return [f"{round(value * scale, decimals) + 0.0:.{decimals}f}".replace("-", "−") for value in values]
        strings = []
        for v in values:
            shown = float(t.inverse_display(v))
            if t.scale == "log":
                exponent = round(v)
                strings.append(f"10{str(exponent).translate(SUPERSCRIPT)}" if abs(v - exponent) < 1e-9 else "")
            else:
                strings.append(f"{shown:g}".replace("-", "−") if abs(v - round(v)) < 1e-9 or abs(v) < 1 else "")
        return strings

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(event)
        else:
            event.ignore()


class ViewBox(pg.ViewBox):
    """Shift + wheel zooms (the wheel alone scrolls the page); Shift + drag zooms into a rectangle; dragging pans."""
    _box_zoom = False

    def mouseDragEvent(self, event, axis=None):
        if event.isStart():
            self._box_zoom = (event.button() == Qt.MouseButton.LeftButton and axis is None
                              and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        if not self._box_zoom:
            return super().mouseDragEvent(event, axis)
        event.accept()
        if event.isFinish():
            self.rbScaleBox.hide()
            rect = QtCore.QRectF(pg.Point(event.buttonDownPos()), pg.Point(event.pos()))
            self.showAxRect(self.childGroup.mapRectFromParent(rect), padding=0)
        else:
            self.updateScaleBox(event.buttonDownPos(), event.pos())

    def wheelEvent(self, event, axis=None):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(event, axis)
        else:
            event.ignore()


class RightViewBox(pg.ViewBox):
    """The view box of a right axis: on top of the panel's own, but it leaves the mouse to it."""

    def wheelEvent(self, event, axis=None):
        event.ignore()

    def mouseDragEvent(self, event, axis=None):
        event.ignore()

    def mouseClickEvent(self, event):
        event.ignore()


def _pen(color, width=None, dash=None, alpha=None):
    c = QtGui.QColor(color)
    if alpha is not None:
        c.setAlphaF(alpha)
    pen = pg.mkPen(c, width=width or theme.current()["plot"]["line_width"])
    if dash:
        pen.setStyle(DASH[dash])
    pen.setCosmetic(True)
    return pen


def _colormap(name):
    if name == "diverging":
        return pg.colormap.get("RdBu_r", source="matplotlib")
    if name == "greys":
        return pg.colormap.get("Greys", source="matplotlib")
    return pg.colormap.get("viridis")


def line_colors(mark, n) -> list[str]:
    if mark.colors is not None:
        return [theme.role_color(c, i) for i, c in enumerate(mark.colors)]
    style = theme.mark_style(mark.style)
    if style.color is not None:
        return [theme.role_color(style.color)] * n
    if mark.colormap == "black-red":
        return black_red(theme.current(), n)
    if mark.colormap in ("viridis", "greys"):
        cmap = _colormap(mark.colormap)
        return [cmap.map(0.1 + 0.8 * i / max(1, n - 1), mode="qcolor").name() for i in range(n)]
    return [theme.role_color(None, i) for i in range(n)]


def _edges(centers):
    c = np.asarray(centers, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    middle = (c[:-1] + c[1:]) / 2
    return np.concatenate(([c[0] - (middle[0] - c[0])], middle, [c[-1] + (c[-1] - middle[-1])]))


class PanelView:
    """One panel of a plot: its PlotItem, axes, transforms, and the items of its marks."""

    def __init__(self, view, name, panel, displays):
        self.view, self.name, self.panel = view, name, panel
        t = theme.current()
        self.box = ViewBox()
        self.axes = {"bottom": TickAxis("bottom"), "left": TickAxis("left")}
        self.item = pg.PlotItem(viewBox=self.box, axisItems=self.axes)
        self.item.showGrid(x=panel.grid, y=panel.grid, alpha=t["grid_alpha"])
        for axis in self.axes.values():
            axis.setPen(t["plot_axis"])
            axis.setTextPen(t["ticks"])
            axis.setStyle(maxTextLevel=0, tickTextOffset=4)
            axis.setTickDensity(1.4)
        fill = QtGui.QColor(t["accent"])
        fill.setAlpha(40)  # see-through, as in the kufner viewer
        self.box.rbScaleBox.setPen(pg.mkPen(t["accent"]))
        self.box.rbScaleBox.setBrush(pg.mkBrush(fill))
        self.tx = Transform(panel.x, displays.get((name, "x")))
        self.ty = Transform(panel.y, displays.get((name, "y")))
        self.axes["bottom"].transform, self.axes["left"].transform = self.tx, self.ty
        self.item.setLabel("bottom", self.tx.label(), color=t["ink2"])
        self.item.setLabel("left", self.ty.label(), color=t["ink2"])
        if panel.x.invert:
            self.box.invertX(True)
        if panel.y.invert:
            self.box.invertY(True)
        self.right_box, self.tr = None, None
        if panel.y_right is not None:
            self._make_right_axis(panel.y_right, displays.get((name, "y_right")))
        self.items, self.tips, self.region_tips = [], [], []
        self.drawn = None  # [Drawn] per mark, in order; None: nothing drawn yet
        self._entry = None  # the Drawn that _draw is filling
        self.note = pg.LabelItem("", color=t["ink2"], size="10pt")
        self.note.setParentItem(self.box)
        self.note.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        self.legend = None

    def _make_right_axis(self, axis_spec, display):
        t = theme.current()
        axis = TickAxis("right")
        self.item.setAxisItems({"right": axis})
        axis.setPen(t["plot_axis"])
        axis.setTextPen(t["ticks"])
        axis.setStyle(maxTextLevel=0)
        self.tr = Transform(axis_spec, display)
        axis.transform = self.tr
        self.item.showAxis("right")
        self.item.setLabel("right", self.tr.label(), color=t["ink2"])
        self.right_box = RightViewBox(enableMenu=False)  # added to the scene by attach_right
        self._right_axis = axis

    def attach_right(self):
        """After the panel is in a scene: put the right view box over the main one."""
        if self.right_box is None:
            return
        if self.right_box.scene() is None:
            self.item.scene().addItem(self.right_box)
        self._right_axis.linkToView(self.right_box)
        self.right_box.setXLink(self.box)
        self.box.sigResized.connect(lambda: self.right_box.setGeometry(self.box.sceneBoundingRect()))
        self.right_box.setGeometry(self.box.sceneBoundingRect())

    def right_width(self) -> float:
        """The width of what stands right of the data area: a right axis, a color bar."""
        width = RIGHT_AXIS_WIDTH if self.right_box is not None else 0
        width += sum(COLORBAR_WIDTH for entry in self.drawn or [] for item, box in entry.items
                     if isinstance(item, pg.ColorBarItem))
        return width

    def pad_right(self, width):
        """Empty room right of the data area (to line up with a panel that has a color bar there)."""
        if self.right_box is not None:
            return
        axis = self.item.getAxis("right")
        if width > 0:
            self.item.showAxis("right")
            axis.setStyle(showValues=False, tickLength=0)
            axis.setPen(pg.mkPen(None))
            axis.setWidth(width)
        else:
            self.item.hideAxis("right")

    def clear(self):
        """Remove every item: the marks', the zero line, the legend."""
        for item, box in self.items + [pair for entry in self.drawn or [] for pair in entry.items]:
            self._remove_item(item, box)
        self.items, self.tips, self.region_tips = [], [], []
        self.drawn = None
        if self.legend is not None:
            if self.legend in [self.item.layout.itemAt(i) for i in range(self.item.layout.count())]:
                self.item.layout.removeItem(self.legend)
            if self.legend.scene():
                self.legend.scene().removeItem(self.legend)
            self.legend = None

    def _remove_item(self, item, box):
        """One item: in a view box, or a color bar or tile text, which are not."""
        if box is None:
            if isinstance(item, pg.ColorBarItem):
                self.item.layout.removeItem(item)
            if item.scene():
                item.scene().removeItem(item)
        else:
            box.removeItem(item)

    def add(self, item, right=False, ignore=False):
        box = self.right_box if right and self.right_box is not None else self.box
        box.addItem(item, ignoreBounds=ignore)
        self._own(item, box)
        return item

    def _own(self, item, box):
        """Note an item as part of the mark being drawn (or of the panel's frame: the zero line)."""
        if self._entry is not None:
            self._entry.items.append((item, box))
        else:
            self.items.append((item, box))

    def set_labels(self, panel):
        """The axes' labels of a description with the same layout (only the text may differ)."""
        t = theme.current()
        for name, axis, transform in (("bottom", panel.x, self.tx), ("left", panel.y, self.ty),
                                      ("right", panel.y_right, self.tr)):
            if transform is not None and axis is not None and transform.axis.label != axis.label:
                transform.axis = axis
                self.item.setLabel(name, transform.label(), color=t["ink2"])

    def set_marks(self, panel):
        """Draw the marks of `panel` (the same layout as before; only the data may differ). A mark equal to the one
        drawn before at its place keeps its items: a moved λ marker replaces one line, not the hundred spectra
        around it (each item costs pyqtgraph ~0.5 ms to add)."""
        self.set_labels(panel)
        old = self.drawn
        if old is None or (panel.legend, panel.zero_line) != (self.panel.legend, self.panel.zero_line):
            self.clear()
            old = []
            self._frame(panel)
        self.panel = panel
        if self.note.text != panel.note:  # (setText lays out HTML: not for every redraw)
            self.note.setText(panel.note)
        self.note.setVisible(bool(panel.note))
        self.drawn = []
        for i, mark in enumerate(panel.marks):
            if i < len(old) and _same_mark(old[i].mark, mark):
                self.drawn.append(old[i])
                continue
            if i < len(old) and self._update(old[i], mark):  # the same look, other data: the items take them
                self.drawn.append(old[i])
                continue
            if i < len(old):
                for item, box in old[i].items:
                    self._remove_item(item, box)
            self._entry = Drawn(mark)
            try:
                self._draw(mark, i)
            except Exception as error:  # one bad mark must not lose the others
                self.note.setText(f"{mark.kind}: {error}")
                self.note.setVisible(True)
            finally:
                entry, self._entry = self._entry, None
            for item, box in entry.items:  # in the order of the marks, also when only some are new
                if box is not None:
                    item.setZValue(item.zValue() + i * 1e-6)
            self.drawn.append(entry)
        for entry in old[len(panel.marks):]:
            for item, box in entry.items:
                self._remove_item(item, box)
        self.tips = [tip for entry in self.drawn for tip in entry.tips]
        self.region_tips = [tip for entry in self.drawn for tip in entry.region_tips]
        entries = [pair for entry in self.drawn for pair in entry.legend]
        if self.legend is not None and [(id(i), n) for i, n in entries] != getattr(self, "_legend_shown", None):
            self._legend_shown = [(id(i), n) for i, n in entries]
            self.legend.clear()
            for item, name in entries:
                self.legend.addItem(item, name)

    def _point_clicked(self, scatter, points, event=None):
        if len(points):
            self.view.point_clicked.emit(self.name, int(scatter.keys[points[0].index()]))

    def _frame(self, panel):
        """What does not depend on the marks: the legend's place, the zero line."""
        t = theme.current()
        self._legend_shown = None
        if panel.legend == "below":  # under the x axis, in rows of three, never over the data
            self.legend = pg.LegendItem(colCount=3, labelTextColor=t["ink2"])
            self.item.layout.addItem(self.legend, 4, 1)
        elif panel.legend:
            corner = {"top right": ((1, 0), (1, 0), (-10, 10)), "top left": ((0, 0), (0, 0), (10, 10)),
                      "bottom right": ((1, 1), (1, 1), (-10, -10)), "bottom left": ((0, 1), (0, 1), (10, -10))}
            item_pos, parent_pos, offset = corner.get(panel.legend, corner["top right"])
            self.legend = pg.LegendItem(offset=offset, labelTextColor=t["ink2"], brush=pg.mkBrush(
                QtGui.QColor(t["tile"]).lighter(100)), pen=pg.mkPen(t["border"]))
            self.legend.setParentItem(self.box)
            self.legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
        if panel.zero_line and self.ty.scale != "log":
            line = pg.InfiniteLine(0, angle=0, movable=False, pen=pg.mkPen(t["zero"], width=1,
                                                                          style=Qt.PenStyle.DotLine))
            line.setZValue(-5)
            self.add(line, ignore=True)

    def _update(self, entry, mark) -> bool:
        """Give the items of an earlier mark the data of a new mark of the same look (kind, style, name, ...) instead
        of making and adding new ones: dragging λ moves a marker and changes a few points, and creating a pyqtgraph
        item and adding it to its view box costs far more than setting its data (2026-11-24). False: the mark
        differs in more than its data, or the items are not the ones set up below; then it is drawn anew."""
        old, items = entry.mark, [item for item, _ in entry.items]
        data = UPDATABLE.get(type(mark))
        if data is None or type(old) is not type(mark) or old.__dict__.keys() != mark.__dict__.keys():
            return False
        if any(not _same_value(value, mark.__dict__[key]) for key, value in old.__dict__.items() if key not in data):
            return False
        right = mark.axis == "right"
        tx, ty = self.tx, (self.tr if right and self.tr else self.ty)
        if isinstance(mark, Line):
            if (old.error_y is None) != (mark.error_y is None) or len(mark.x) > LONG_LINE:
                return False
            curve = items[-1]
            if not isinstance(curve, pg.PlotCurveItem) or len(items) != (1 if mark.error_y is None else 2):
                return False
            x, y = tx.forward(mark.x), ty.forward(mark.y)
            if mark.error_y is not None:
                low, high = ty.forward(mark.y - mark.error_y), ty.forward(mark.y + mark.error_y)
                items[0].curves[0].setData(x, low)
                items[0].curves[1].setData(x, high)
            curve.setData(x, y)
        elif isinstance(mark, Points):
            if (old.error_x is None, old.error_y is None) != (mark.error_x is None, mark.error_y is None):
                return False
            scatter = items[-1]
            if not isinstance(scatter, pg.ScatterPlotItem) or \
                    len(items) != 1 + (mark.error_x is not None or mark.error_y is not None):
                return False
            x, y = tx.forward(mark.x), ty.forward(mark.y)
            if len(items) == 2:
                kwargs = {}
                if mark.error_y is not None:
                    kwargs.update(top=ty.forward(mark.y + mark.error_y) - y, bottom=y - ty.forward(mark.y - mark.error_y))
                if mark.error_x is not None:
                    kwargs.update(left=x - tx.forward(mark.x - mark.error_x), right=tx.forward(mark.x + mark.error_x) - x)
                items[0].setData(x=x, y=y, **kwargs)
            scatter.setData(x, y, data=mark.labels or [mark.name or ""] * len(x))
            if mark.keys is not None:
                scatter.keys = mark.keys
        elif isinstance(mark, Lines):
            if len(items) != len(mark.y) or len(old.y) != len(mark.y) or \
                    not all(isinstance(item, pg.PlotCurveItem) for item in items):
                return False
            x = tx.forward(mark.x)
            for curve, row in zip(items, mark.y):
                curve.setData(x, ty.forward(row))
        elif isinstance(mark, (VLine, HLine)):
            vertical = isinstance(mark, VLine)
            if len(items) != 1 or not isinstance(items[0], pg.InfiniteLine):
                return False
            items[0].setPos(float((tx if vertical else ty).forward(mark.x if vertical else mark.y)))
        else:
            return False
        entry.mark = mark
        return True

    def _draw(self, mark, index):
        right = mark.axis == "right"
        tx, ty = self.tx, (self.tr if right and self.tr else self.ty)
        style = theme.mark_style(mark.style)  # its role's look in the theme, the module's own fields on top
        if isinstance(mark, Line):
            x, y = tx.forward(mark.x), ty.forward(mark.y)
            color = theme.role_color(style.color, index)
            if mark.error_y is not None:
                low, high = ty.forward(mark.y - mark.error_y), ty.forward(mark.y + mark.error_y)
                c1, c2 = pg.PlotCurveItem(x, low), pg.PlotCurveItem(x, high)
                fill = QtGui.QColor(color)
                fill.setAlphaF(0.18)
                self.add(pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(fill)), right)
            pen = _pen(color, style.width, style.dash, style.alpha)
            if len(x) > LONG_LINE and not mark.step:
                # long traces (power meter readings): min and max per pixel, only the visible part
                curve = pg.PlotDataItem(x, y, pen=pen, connect="finite")
                curve.setDownsampling(auto=True, method="peak")
                curve.setClipToView(True)
            else:
                curve = pg.PlotCurveItem(x, y, pen=pen, connect="finite", stepMode="left" if mark.step else None)
            self.add(curve, right)
            if mark.name:
                if isinstance(curve, pg.PlotCurveItem):
                    self._entry.tips.append((curve, mark.name))
                self._entry.legend.append((curve, mark.name))
        elif isinstance(mark, Points):
            x, y = tx.forward(mark.x), ty.forward(mark.y)
            color = theme.role_color(style.color, index)
            fill = QtGui.QColor(color)
            if style.alpha is not None:
                fill.setAlphaF(style.alpha)
            if mark.error_y is not None or mark.error_x is not None:
                kwargs = {}
                if mark.error_y is not None:
                    top, bottom = ty.forward(mark.y + mark.error_y) - y, y - ty.forward(mark.y - mark.error_y)
                    kwargs.update(top=top, bottom=bottom)
                if mark.error_x is not None:
                    r, l = tx.forward(mark.x + mark.error_x) - x, x - tx.forward(mark.x - mark.error_x)
                    kwargs.update(left=l, right=r)
                self.add(pg.ErrorBarItem(x=x, y=y, pen=_pen(color, style.width or 1), beam=0, **kwargs), right)
            labels = mark.labels or [mark.name or ""] * len(x)
            scatter = pg.ScatterPlotItem(x, y, symbol=style.symbol or "o",
                                         size=style.size or theme.current()["plot"]["point_size"],
                                         brush=pg.mkBrush(fill), pen=pg.mkPen(None), data=labels, hoverable=True,
                                         tip=lambda x, y, data: str(data) if data else None)
            self.add(scatter, right)
            if mark.keys is not None:  # a click on a point tells its key (a Toggle interaction)
                scatter.keys = mark.keys
                scatter.sigClicked.connect(self._point_clicked)
                scatter.setCursor(Qt.CursorShape.PointingHandCursor)
            if mark.name:
                self._entry.legend.append((scatter, mark.name))
        elif isinstance(mark, Lines):
            x = tx.forward(mark.x)
            colors = line_colors(mark, len(mark.y))
            width = style.width
            for i, (row, color) in enumerate(zip(mark.y, colors)):
                curve = pg.PlotCurveItem(x, ty.forward(row), pen=_pen(color, width, style.dash, style.alpha),
                                         connect="finite")
                self.add(curve, right)
                label = mark.labels[i] if mark.labels and i < len(mark.labels) else None
                if label:
                    self._entry.tips.append((curve, f"{mark.name + ': ' if mark.name else ''}{label}"))
                    if len(mark.y) <= 12 and getattr(mark, "legend", True):
                        self._entry.legend.append((curve, label))
        elif isinstance(mark, Band):
            x = tx.forward(mark.x)
            color = QtGui.QColor(theme.role_color(style.color, index))
            color.setAlphaF(style.alpha if style.alpha is not None else 0.2)
            c1, c2 = pg.PlotCurveItem(x, ty.forward(mark.low)), pg.PlotCurveItem(x, ty.forward(mark.high))
            self.add(pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(color)), right)
        elif isinstance(mark, (VLine, HLine)):
            vertical = isinstance(mark, VLine)
            position = float((tx if vertical else ty).forward(mark.x if vertical else mark.y))
            color = theme.role_color(style.color or "muted")
            line = pg.InfiniteLine(position, angle=90 if vertical else 0, movable=False,
                                   pen=_pen(color, style.width or 1.2, style.dash or "dash"),
                                   label=mark.label, labelOpts=dict(position=0.95, color=theme.current()["ink2"]))
            self.add(line, right, ignore=True)
        elif isinstance(mark, Region):
            color = QtGui.QColor(theme.role_color(style.color or "muted"))
            color.setAlphaF(style.alpha if style.alpha is not None else 0.15)
            labels = mark.labels or [mark.name or mark.label or ""] * len(mark.starts)
            for start, stop, label in zip(tx.forward(mark.starts), tx.forward(mark.stops), labels):
                region = pg.LinearRegionItem((start, stop), movable=False, brush=pg.mkBrush(color),
                                             pen=pg.mkPen(None))
                region.setZValue(-10)
                self.add(region, right, ignore=True)
                if label:
                    self._entry.region_tips.append((min(start, stop), max(start, stop), label))
            if mark.name:
                sample = pg.BarGraphItem(x0=[], x1=[], height=[], brush=pg.mkBrush(color), pen=pg.mkPen(None))
                self._entry.legend.append((sample, mark.name))
        elif isinstance(mark, Heatmap) and not np.isfinite(mark.z).any():
            pass  # nothing to color
        elif isinstance(mark, Heatmap):
            x, y = tx.forward(mark.x), ty.forward(mark.y)
            order_x, order_y = np.argsort(x), np.argsort(y)
            z = mark.z[np.ix_(order_x, order_y)]
            X, Y = np.meshgrid(_edges(x[order_x]), _edges(y[order_y]), indexing="ij")
            cmap = _colormap(mark.colormap)
            # missing cells as NaN, which PColorMeshItem leaves out; a masked array crashed the process (segfault in
            # pyqtgraph 0.14 as soon as any cell was masked, found 2026-11-22 with a TA map shifted out of its data)
            z = np.where(np.isfinite(z), z, np.nan)
            mesh = pg.PColorMeshItem(X, Y, z, colorMap=cmap, enableAutoLevels=False)
            low, high = mark.color_limits()
            mesh.setLevels((low, high))
            self.add(mesh, right)
            if mark.colorbar:
                bar = pg.ColorBarItem(colorMap=cmap, label=mark.colorbar, interactive=False, width=12,
                                      values=(low, high))
                bar.setImageItem(mesh, insert_in=self.item)
                bar.getAxis("right").setTextPen(theme.current()["ticks"])
                bar.getAxis("right").setPen(theme.current()["plot_axis"])
                bar.getAxis("right").setStyle(tickTextWidth=40)
                bar.getAxis("right").setWidth(COLORBAR_AXIS_WIDTH)
                self._own(bar, None)
        elif isinstance(mark, Text):
            color = theme.role_color(style.color or "ink")
            anchor = {"left": (0, 0.5), "right": (1, 0.5), "center": (0.5, 0.5)}.get(mark.anchor, (0, 0.5))
            if mark.coords == "tile":
                label = pg.LabelItem(mark.text, color=color, size="9pt")
                label.setParentItem(self.box)
                label.anchor(itemPos=anchor, parentPos=(mark.x, 1 - mark.y))
                self._own(label, None)
            else:
                item = pg.TextItem(mark.text, color=color, anchor=anchor)
                item.setPos(float(tx.forward(mark.x)), float(ty.forward(mark.y)))
                self.add(item, right, ignore=True)


# the attributes of a mark that are its data: _update puts new ones into the items it made; the others (style, name,
# labels, ...) must be the same
UPDATABLE = {Line: ("x", "y", "error_y"), Points: ("x", "y", "error_x", "error_y", "labels", "keys"),
             Lines: ("y",), VLine: ("x",), HLine: ("y",)}


class Drawn:
    """A mark and what drawing it made: its items (with their view box), tool tips and legend entries."""

    def __init__(self, mark):
        self.mark = mark
        self.items, self.tips, self.region_tips, self.legend = [], [], [], []


def _same_value(a, b) -> bool:
    if a is b:  # (a plot description that was kept: its marks are the same objects)
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:  # not numbers
            return bool(np.array_equal(a, b))
    try:
        return type(a) is type(b) and bool(a == b)
    except (TypeError, ValueError):  # e.g. a list of arrays
        return False


def _same_mark(a, b) -> bool:
    """The same mark: the same kind, data and style (its items can stay)."""
    if type(a) is not type(b) or a.__dict__.keys() != b.__dict__.keys():
        return False
    return all(_same_value(value, b.__dict__[key]) for key, value in a.__dict__.items())


def _shape(axis) -> str:
    """An axis without its label: a new label (the absorbance plot names the wavelength, which moves with the λ cursor)
    changes no layout, so it is set in place (PanelView.set_labels) and does not build the plot anew."""
    return repr(None if axis is None else dataclasses.replace(axis, label=""))


def _structure(spec, displays):
    """What decides the layout of a plot: when it stays the same, new data only replace the marks (the zoom stays)."""
    return (tuple(map(tuple, spec.mosaic)), tuple(spec.sizes[0]), tuple(spec.sizes[1]),
            tuple((n, _shape(p.x), _shape(p.y), _shape(p.y_right)) for n, p in spec.panels.items()),
            tuple(map(tuple, spec.share_x)), tuple(map(tuple, spec.share_y)), tuple(sorted(displays.items())))


class PlotView(pg.GraphicsLayoutWidget):
    """A plot: one or more panels in a grid. set_spec() draws a description; when its layout is the same as
    before, only the marks are replaced, so zoom and cursors stay. Keeps a picture of itself (see the module
    docstring)."""
    mouse_left = QtCore.Signal()
    coordinates = QtCore.Signal(str)  # the coordinates under the mouse, "" when it leaves the data
    point_clicked = QtCore.Signal(str, int)  # panel, key of a clicked point (Points with keys)

    def __init__(self, rows=1):
        super().__init__()
        # No BSP index in the scene. pyqtgraph's axes with a grid count their view box into their bounding rect, so it
        # changes without prepareGeometryChange when the view box narrows (a color bar, pad_right); the index kept the
        # old rect, missed the axis when the plot was rebuilt, and painting then read the deleted axis: a segfault
        # (2026-11-23, about one full test run in four). Qt looks items up from the top-level items down and a plot
        # has one, so the index never sped anything up (measured: the same render and drag times without it).
        self.scene().setItemIndexMethod(QtWidgets.QGraphicsScene.ItemIndexMethod.NoIndex)
        self._picture = None
        self.scene().changed.connect(self._forget_picture)
        self.setMinimumSize(280, PLOT_HEIGHT * rows)
        self.ci.layout.setContentsMargins(2, 4, 6, 2)
        self.ci.layout.setSpacing(4)
        self.panels: dict[str, PanelView] = {}
        self.spec, self._structure = None, None
        self.displays = {}  # (panel, axis) -> chosen display unit (axes with several)
        self.note = pg.LabelItem("", color=theme.current()["ink2"], size="10pt")
        self.cursors = []
        self._postponed = None  # a description with a new layout that arrived while a cursor was being dragged
        self._grabbed = self._hovered = None  # the cursor being dragged, the one under the mouse
        self._draw_later = QtCore.QTimer(self, singleShot=True, interval=0, timeout=self._draw_postponed)
        self.scene().sigMouseMoved.connect(self._mouse_moved)

    # --- drawing -----------------------------------------------------------------------------------------------
    def dragging(self) -> bool:
        """A cursor or range of this plot is being dragged with the mouse."""
        return any(cursor.dragging() for cursor in self.cursors)

    def drag_finished(self):
        """A drag ended: draw what was held back meanwhile (not inside the signal of the line that was dragged:
        building the plot anew deletes that line while its own method is still running)."""
        if self._postponed is not None:
            self._draw_later.start()

    def _draw_postponed(self):
        spec, self._postponed = self._postponed, None
        if spec is not None:
            self.set_spec(spec)

    def set_spec(self, spec):
        structure = _structure(spec, self.displays)
        if structure != self._structure and self.dragging():
            # A new layout (typically the note "Computing …" while the results of the new λ are on their way) builds
            # the plot anew and deletes the line under the mouse; pyqtgraph then kept sending the drag to the deleted
            # line ("Error sending hover exit event", RuntimeError: ... already deleted, 2026-11-24). So the plot
            # stays as it is until the mouse is released.
            self._postponed = spec
            return
        self._postponed = None
        if spec is self.spec and structure == self._structure:  # the very description drawn already (kept by the card)
            for cursor in self.cursors:
                cursor.redraw()
            return
        if structure != self._structure:
            self._build(spec)
            self._structure = structure
        self.spec = spec
        for name, panel_view in self.panels.items():
            panel_view.set_marks(spec.panels[name])
        self._align(spec)
        for cursor in self.cursors:
            cursor.redraw()

    def _align(self, spec):
        """Panels that share an axis line up: the same width for every left axis, and room on the right of a panel
        whose partner has a color bar (or a right axis) there."""
        if len(self.panels) < 2:
            return
        for panel_view in self.panels.values():
            panel_view.axes["left"].setWidth(LEFT_AXIS_WIDTH)
        for group in spec.share_x:
            extra = {name: self.panels[name].right_width() for name in group}
            widest = max(extra.values())
            for name in group:
                self.panels[name].pad_right(widest - extra[name])

    def _build(self, spec):
        for panel_view in self.panels.values():
            panel_view.clear()
        self.ci.clear()
        self.panels = {}
        for name, (row, column, row_span, column_span) in spec.cells().items():
            panel_view = PanelView(self, name, spec.panels[name], self.displays)
            self.ci.addItem(panel_view.item, row, column, row_span, column_span)
            panel_view.attach_right()
            self.panels[name] = panel_view
        grid = self.ci.layout
        for column, size in enumerate(spec.sizes[0]):
            grid.setColumnStretchFactor(column, int(size * 10))
        for row, size in enumerate(spec.sizes[1]):
            grid.setRowStretchFactor(row, int(size * 10))
        self.spec = spec
        self.link_internal()
        for cursor in self.cursors:
            cursor.attach()

    def link_internal(self):
        """The panels that share an axis (share_x, share_y of the description) zoom together."""
        if self.spec is None:
            return
        for group in self.spec.share_x:
            for name in group[1:]:
                self.panels[name].box.setXLink(self.panels[group[0]].box)
        for group in self.spec.share_y:
            for name in group[1:]:
                self.panels[name].box.setYLink(self.panels[group[0]].box)

    def panel(self, name=None) -> PanelView | None:
        if name is None:
            return next(iter(self.panels.values()), None)
        return self.panels.get(name)

    def set_display(self, panel, axis, display):
        """Show an axis in another of its display units (the plot is drawn again)."""
        self.displays[(panel, axis)] = display
        if self.spec is not None:
            self.set_spec(self.spec)

    def auto_range(self):
        for panel_view in self.panels.values():
            panel_view.box.enableAutoRange()

    # --- mouse ------------------------------------------------------------------------------------------------
    def _mouse_moved(self, scene_pos):
        for panel_view in self.panels.values():
            box = panel_view.box
            if box.sceneBoundingRect().contains(scene_pos):
                point = box.mapSceneToView(scene_pos)
                px, py = box.viewPixelSize()
                x, y = float(panel_view.tx.inverse_display(point.x())), float(panel_view.ty.inverse_display(point.y()))
                dx = abs(float(panel_view.tx.inverse_display(point.x() + px)) - x)
                dy = abs(float(panel_view.ty.inverse_display(point.y() + py)) - y)
                self.coordinates.emit(f"x = {format_coordinate(x, dx)}    y = {format_coordinate(y, dy)}")
                self._show_tip(panel_view, point, px, py)
                return
        self.coordinates.emit("")
        QtWidgets.QToolTip.hideText()

    def _show_tip(self, panel_view, point, px, py):
        if QtWidgets.QApplication.mouseButtons() != Qt.MouseButton.NoButton:  # dragging: no tips in the way
            QtWidgets.QToolTip.hideText()
            return
        best, best_distance = None, 6.0
        for curve, text in panel_view.tips:
            x, y = curve.getData()
            if x is None or not len(x):
                continue
            with np.errstate(invalid="ignore"):
                distance = np.nanmin(np.hypot((x - point.x()) / px, (y - point.y()) / py))
            if distance < best_distance:
                best, best_distance = text, distance
        if not best:  # inside a shaded span with a label (e.g. a stirring band)?
            best = next((text for start, stop, text in panel_view.region_tips if start <= point.x() <= stop), None)
        if best:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), best, self)
        else:
            QtWidgets.QToolTip.hideText()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hover(None)
        self.coordinates.emit("")
        self.mouse_left.emit()

    # --- cursors: painted over the picture, and the mouse is theirs where they are ---------------------------------
    def _cursor_at(self, pos):
        return next((c for c in reversed(self.cursors) if c.hit(pos)), None)

    def _hover(self, pos):
        cursor = self._cursor_at(pos) if pos is not None else None
        if cursor is self._hovered:
            return
        if self._hovered is not None:
            self._hovered.hover = False
        self._hovered = cursor
        if cursor is None:
            self.viewport().unsetCursor()
        else:
            cursor.hover = True
            self.viewport().setCursor(Qt.CursorShape.SizeHorCursor if cursor.axis == "x"
                                      else Qt.CursorShape.SizeVerCursor)
        self.viewport().update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            cursor = self._cursor_at(event.position())
            if cursor is not None:
                self._grabbed = cursor
                cursor.begin_drag(event.position())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._grabbed is not None:
            self._grabbed.drag_to(event.position())
            event.accept()
            return
        self._hover(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._grabbed is not None and event.button() == Qt.MouseButton.LeftButton:
            cursor, self._grabbed = self._grabbed, None
            cursor.end_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- the picture (see the module docstring) ---------------------------------------------------------------------
    def _forget_picture(self):
        self._picture = None

    def paintEvent(self, event):
        self.scene().prepareForPaint()
        for _ in range(2):
            QtCore.QCoreApplication.sendPostedEvents(self.scene(), QtCore.QEvent.Type.MetaCall)
        ratio = self.devicePixelRatioF()
        if (self._picture is None or self._picture.devicePixelRatio() != ratio
                or self._picture.deviceIndependentSize().toSize() != self.viewport().size()):
            self._picture = QtGui.QPixmap(self.viewport().size() * ratio)
            self._picture.setDevicePixelRatio(ratio)
            self._picture.fill(self.backgroundBrush().color())
            painter = QtGui.QPainter(self._picture)
            painter.setRenderHints(self.renderHints())
            area = self.viewport().rect()
            self.render(painter, QtCore.QRectF(area), area)
            painter.end()
        painter = QtGui.QPainter(self.viewport())
        painter.drawPixmap(0, 0, self._picture)
        for cursor in self.cursors:  # (not in the picture: moving one must not draw the plot again)
            cursor.paint(painter)
        painter.end()


class CursorItem:
    """A draggable line on one panel that shows a parameter's value (stored units) and sets it when dragged.
    on_change(value, final): called while dragging (at the pace the window can keep, newest position) and at the end
    (final True). snap: values (stored units) the line snaps to while dragged.

    The line is not an item of the plot's scene: PlotView paints it over the picture it keeps of the scene and takes
    the mouse for it (grab, hover, drag). Moving a pyqtgraph line changed the scene, so the whole plot was drawn
    again for every step of the drag (the spectra, a hundred and more curves, 10 ms); now only the line is (2026-11-24,
    the user: dragging λ must be as smooth as in the old viewer)."""
    GRAB = 5  # pixels either side of the line that take hold of it

    def __init__(self, view, panel, axis, value, on_change, snap=None, label=None, color=None):
        self.view, self.panel_name, self.axis = view, panel, axis
        self.value, self.on_change, self.snap, self.label = value, on_change, snap, label
        self.color = color or theme.role_color("error")
        self.position, self.hover, self._dragging, self._offset = None, False, False, 0.0  # (screen coordinate)
        # while dragging, the newest position is used as soon as the window is idle (as in the kufner viewer): a
        # fast drag skips positions instead of falling behind, and where the mouse rests is always used
        self._pending = None
        self._timer = QtCore.QTimer(singleShot=True, interval=0, timeout=self._apply)
        view.cursors.append(self)
        self.attach()

    def _transform(self, panel_view):
        return panel_view.tx if self.axis == "x" else panel_view.ty

    def attach(self):
        """The plot was built anew: where the line is follows from the value again."""
        self.position = None
        self.redraw()

    def redraw(self):
        if self._dragging:
            return  # the mouse decides where it is (new results must not pull it back)
        panel_view = self.view.panel(self.panel_name)
        position = None
        if panel_view is not None and self.value is not None:
            position = float(self._transform(panel_view).forward(self.value))
            position = position if math.isfinite(position) else None
        if position != self.position:
            self.position = position
            self.view.viewport().update()

    def set_value(self, value):
        self.value = value
        self.redraw()

    def dragging(self) -> bool:
        return self._dragging

    # --- where the line is on the screen -----------------------------------------------------------------------
    def _geometry(self):
        """(the line's pixel in the viewport along its axis, the panel's data area in the viewport); None if the line
        is not shown (no value, or outside the data area)."""
        panel_view = self.view.panel(self.panel_name)
        if panel_view is None or self.position is None:
            return None
        box, to_view = panel_view.box, self.view.viewportTransform()
        area = to_view.mapRect(box.sceneBoundingRect())
        at = QtCore.QPointF(self.position, 0) if self.axis == "x" else QtCore.QPointF(0, self.position)
        point = to_view.map(box.mapViewToScene(at))
        pixel = point.x() if self.axis == "x" else point.y()
        low, high = (area.left(), area.right()) if self.axis == "x" else (area.top(), area.bottom())
        return (pixel, area) if low <= pixel <= high else None

    def hit(self, pos) -> bool:
        """Is the viewport point (QPointF) on the line?"""
        geometry = self._geometry()
        if geometry is None:
            return False
        pixel, area = geometry
        along = pos.x() if self.axis == "x" else pos.y()
        return abs(along - pixel) <= self.GRAB and area.contains(pos)

    def paint(self, painter):
        """Over the picture of the plot: the line, and its label by the end of it."""
        geometry = self._geometry()
        if geometry is None:
            return
        pixel, area = geometry
        vertical = self.axis == "x"
        t = theme.current()
        painter.save()
        painter.setRenderHints(self.view.renderHints())
        painter.setClipRect(area)
        pen = QtGui.QPen(QtGui.QColor(self.color), 3 if self.hover or self._dragging else 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        if vertical:
            painter.drawLine(QtCore.QPointF(pixel, area.top()), QtCore.QPointF(pixel, area.bottom()))
        else:
            painter.drawLine(QtCore.QPointF(area.left(), pixel), QtCore.QPointF(area.right(), pixel))
        if self.label:
            try:
                text = self.label.format(value=self.position)
            except (KeyError, IndexError, ValueError):
                text = self.label
            metrics = painter.fontMetrics()
            width, height = metrics.horizontalAdvance(text) + 8, metrics.height() + 4
            if vertical:  # near the top, on the side of the line that has more room
                x = pixel - width - 3 if pixel > area.center().x() else pixel + 3
                box = QtCore.QRectF(x, area.top() + 0.04 * area.height(), width, height)
            else:
                y = pixel - height - 3 if pixel > area.center().y() else pixel + 3
                box = QtCore.QRectF(area.right() - width - 0.04 * area.width(), y, width, height)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(t["tile"]))
            painter.drawRect(box)
            painter.setPen(QtGui.QColor(t["ink2"]))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    # --- the mouse (PlotView passes it on) -------------------------------------------------------------------------
    def _coordinate(self, pos) -> float:
        """The screen coordinate (along the line's axis) under a viewport point."""
        point = self.view.panel(self.panel_name).box.mapSceneToView(self.view.mapToScene(pos.toPoint()))
        return point.x() if self.axis == "x" else point.y()

    def begin_drag(self, pos):
        self._dragging = True
        self._offset = self.position - self._coordinate(pos)  # the line does not jump to the mouse

    def drag_to(self, pos):
        self.position = self._coordinate(pos) + self._offset
        self._pending = self._value_at(self.position)
        self._timer.start()
        self.view.viewport().update()

    def end_drag(self):
        self._dragging = False
        self._finished()

    def _value_at(self, position):
        value = float(self._transform(self.view.panel(self.panel_name)).inverse(position))
        if self.snap is not None and len(self.snap):
            snap = np.asarray(self.snap, dtype=float)
            value = float(snap[np.nanargmin(np.abs(snap - value))])
        return value

    def _apply(self):
        value, self._pending = self._pending, None
        if value is not None and value != self.value:  # (snapped to the same measured value: nothing to do)
            self.value = value
            self.on_change(value, False)

    def _finished(self):
        self._timer.stop()
        self._pending = None
        self.value = self._value_at(self.position)
        self.redraw()
        self.on_change(self.value, True)
        self.view.drag_finished()


class RangeItem:
    """A draggable range on a panel that shows and sets a Range parameter (stored units)."""

    def __init__(self, view, panel, value, on_change, axis="x"):
        self.view, self.panel_name, self.value, self.on_change, self.axis = view, panel, value, on_change, axis
        self.region = None
        view.cursors.append(self)
        self.attach()

    def attach(self):
        panel_view = self.view.panel(self.panel_name)
        if panel_view is None:
            return
        color = QtGui.QColor(theme.current()["accent"])
        color.setAlphaF(0.15)
        self.region = pg.LinearRegionItem((0, 1), orientation="vertical" if self.axis == "x" else "horizontal",
                                          brush=pg.mkBrush(color))
        self.region.setZValue(-8)
        panel_view.box.addItem(self.region, ignoreBounds=True)
        self.region.sigRegionChangeFinished.connect(self._finished)
        self.redraw()

    def redraw(self):
        if self.region is None:
            return
        panel_view = self.view.panel(self.panel_name)
        t = panel_view.tx if self.axis == "x" else panel_view.ty
        self.region.setVisible(self.value is not None)
        if self.value is not None:
            self.region.blockSignals(True)
            self.region.setRegion([float(t.forward(v)) for v in self.value])
            self.region.blockSignals(False)

    def set_value(self, value):
        self.value = value
        self.redraw()

    def dragging(self) -> bool:
        region = self.region
        return region is not None and (bool(region.moving) or any(line.moving for line in region.lines))

    def hit(self, pos) -> bool:  # (a region is an item of the scene: the scene takes the mouse for it)
        return False

    def paint(self, painter):
        pass

    def _finished(self):
        panel_view = self.view.panel(self.panel_name)
        t = panel_view.tx if self.axis == "x" else panel_view.ty
        low, high = self.region.getRegion()
        self.value = sorted([float(t.inverse(low)), float(t.inverse(high))])
        self.on_change(self.value, True)
        self.view.drag_finished()


class PickHandler:
    """Ctrl + click on a panel adds a point (stored units), Ctrl + right click removes the nearest one."""

    def __init__(self, view, panel, on_add, on_remove):
        self.view, self.panel_name, self.on_add, self.on_remove = view, panel, on_add, on_remove
        view.scene().sigMouseClicked.connect(self._clicked)

    def _clicked(self, event):
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return
        panel_view = self.view.panel(self.panel_name)
        if panel_view is None or not panel_view.box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = panel_view.box.mapSceneToView(event.scenePos())
        x, y = float(panel_view.tx.inverse(point.x())), float(panel_view.ty.inverse(point.y()))
        event.accept()
        (self.on_add if event.button() == Qt.MouseButton.LeftButton else self.on_remove)(x, y)


def render_image(spec, width=640, height=PLOT_HEIGHT, ratio=1.0) -> QtGui.QImage:
    """A plot description as a picture, without a window (check, tests): drawn exactly as in the window."""
    view = PlotView()
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.resize(width, height)
    view.set_spec(spec)
    view.show()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
        view.scene().prepareForPaint()
    image = QtGui.QImage(int(width * ratio), int(height * ratio), QtGui.QImage.Format.Format_ARGB32)
    image.setDevicePixelRatio(ratio)
    image.fill(QtGui.QColor(theme.current()["tile"]))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    view.render(painter, QtCore.QRectF(0, 0, width, height), view.viewport().rect())
    painter.end()
    view.hide()
    view.deleteLater()
    return image
