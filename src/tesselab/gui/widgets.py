"""
Small widgets: a grid that wraps its tiles like text (ReflowGrid), a label that shortens its text only when needed
(ElidedLabel), the (i) with an explanation (InfoIcon), and a plot in its tile (PlotTile).
"""
import math

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab.gui import theme
from tesselab.gui.plots import PLOT_HEIGHT


class ReflowGrid(QtWidgets.QWidget):
    """Widgets in as many columns as fit (each at least cell_width wide), like text that wraps. A widget with
    span=2 (attribute) takes two columns when there are at least two. From the kufner viewer: its minimum size is one
    column (so cards side by side get equal widths), and it arranges itself at once when its number of columns
    changes (otherwise the plots were drawn once more at an intermediate size)."""

    def __init__(self, widgets, cell_width=430):
        super().__init__()
        self.widgets, self.cell_width, self.columns = list(widgets), cell_width, 0
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self._place(1)

    def minimumSizeHint(self):
        width = max((w.minimumSizeHint().expandedTo(w.minimumSize()).width() for w in self.widgets), default=0)
        return QtCore.QSize(width, super().minimumSizeHint().height())

    def sizeHint(self):
        return self.minimumSizeHint()

    def resizeEvent(self, event):
        if self._fit(event.size().width()):
            self.grid.activate()
            self.resize(self.width(), self.minimumSizeHint().height())
        super().resizeEvent(event)

    def _fit(self, width) -> bool:
        columns = max(1, (width + 8) // self.cell_width)
        if columns == self.columns:
            return False
        self._place(columns)
        return True

    def _place(self, columns):
        for widget in self.widgets:
            self.grid.removeWidget(widget)
        row = column = 0
        for widget in self.widgets:
            span = min(getattr(widget, "span", 1), columns)
            if column + span > columns:
                row, column = row + 1, 0
            self.grid.addWidget(widget, row, column, 1, span)
            column += span
            if column >= columns:
                row, column = row + 1, 0
        for c in range(max(columns, self.columns)):
            self.grid.setColumnStretch(c, 1 if c < columns else 0)
        self.columns = columns


class CardTop(QtWidgets.QWidget):
    """The top of a card (#card) with its header. When the card is scrolled up, it stays at the top of the visible area
    (pinned) and draws the card's top as it looks unscrolled (the card's color, the border with the round corners),
    hiding what scrolls by below it. Not pinned, it is transparent: the card draws itself. (From the kufner viewer.)"""

    def __init__(self, parent):
        super().__init__(parent)
        self.pinned = False
        self.active = False  # the card is the active one: its border in the accent color (as the style sheet's)

    def pin(self, pinned):
        if pinned != self.pinned:
            self.pinned = pinned
            self.update()

    def paintEvent(self, event):
        if not self.pinned:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # the page first: outside the round corners, the card's straight edges scroll by underneath
        painter.fillRect(self.rect(), QtGui.QColor(theme.color("page")))
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.color("accent" if self.active else "border")), 1))
        painter.setBrush(QtGui.QColor(theme.color("card")))
        # the border on the pixels where the style sheet draws it; open at the bottom, where the card goes on
        r = theme.CARD_R
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, self.width() - 1, self.height() + r), r, r)
        painter.end()


class ElidedLabel(QtWidgets.QLabel):
    """A one-line label that shortens its text with … only when there is not enough room; the full text as tool tip."""

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self):
        return QtCore.QSize(20, super().minimumSizeHint().height())

    def paintEvent(self, event):
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())
        painter = QtGui.QPainter(self)
        self.style().drawItemText(painter, self.contentsRect(), self.alignment(), self.palette(), self.isEnabled(),
                                  text, self.foregroundRole())
        painter.end()


class InfoIcon(QtWidgets.QLabel):
    """The (i) next to a title: its explanation on hover, or at once on click."""

    def __init__(self, text):
        super().__init__("ⓘ", objectName="info", toolTip=f"<p>{text}</p>")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setVisible(bool(text))

    def mouseReleaseEvent(self, event):
        QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), self.toolTip(), self, self.rect())


class PlotWindow(QtWidgets.QWidget):
    """The window of a plot moved out of its tile (still the same, live plot)."""

    def __init__(self, tile):
        super().__init__(tile, Qt.WindowType.Window)
        self.tile = tile
        self.resize(1100, 750)
        layout = QtWidgets.QVBoxLayout(self)
        self.coordinates = QtWidgets.QLabel(objectName="hint", alignment=Qt.AlignmentFlag.AlignRight)
        tile.view.coordinates.connect(self.coordinates.setText)
        layout.addWidget(self.coordinates)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, tile.put_back)

    def closeEvent(self, event):
        self.tile.put_back()
        super().closeEvent(event)


class PlotTile(QtWidgets.QFrame):
    """A plot in its tile: number and title, the (i), the coordinates under the mouse, an optional row of settings,
    the plot. A double click moves the (same, live) plot into a window of its own; closing it or Esc puts it back."""

    def __init__(self, view, title, info="", settings=(), span=1, window_title=""):
        super().__init__(objectName="tile")
        self.view, self.text, self.number, self.span = view, title, None, span
        self.kept = None  # (dark theme?, what the draw function read, the description it made): see NodeGroup.draw
        self.window_title = window_title
        self.title = QtWidgets.QLabel(title, objectName="heading")
        self.coordinates = ElidedLabel(objectName="hint")
        self.coordinates.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        view.coordinates.connect(self.coordinates.setText)
        self.placeholder = QtWidgets.QPushButton("Shown in its own window: click to put it back here")
        self.placeholder.setFlat(True)
        self.placeholder.setMinimumHeight(PLOT_HEIGHT)
        self.placeholder.setVisible(False)
        self.placeholder.clicked.connect(self.put_back)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.addWidget(self.title, 0, 0)
        layout.addWidget(InfoIcon(info), 0, 1)
        layout.addWidget(self.coordinates, 0, 2)
        layout.setColumnStretch(2, 1)
        self.row = 1
        if settings:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 2)
            for widget in settings:
                row.addWidget(widget)
            row.addStretch()
            layout.addLayout(row, 1, 0, 1, 3)
            self.row = 2
        self.message = QtWidgets.QLabel(objectName="note", wordWrap=True)
        self.message.setVisible(False)
        layout.addWidget(self.message, self.row, 0, 1, 3)
        self.row += 1
        layout.addWidget(view, self.row, 0, 1, 3)
        layout.addWidget(self.placeholder, self.row, 0, 1, 3)
        self.window_ = None
        view.scene().sigMouseClicked.connect(self._clicked)

    def set_number(self, number):
        self.number = number
        self.title.setText(f"{number}. {self.text}" if number else self.text)

    def set_message(self, text, error=False):
        """A warning or error that concerns this plot, above it ("" hides it)."""
        self.message.setObjectName("error" if error else "note")
        self.message.setStyleSheet("")  # the object name changed: let the style sheet apply again
        self.message.setText(text)
        self.message.setVisible(bool(text))

    def _clicked(self, event):
        if event.double() and event.button() == Qt.MouseButton.LeftButton and not event.isAccepted():
            QtCore.QTimer.singleShot(0, self.put_back if self.placeholder.isVisible() else self.pop_out)

    def pop_out(self):
        if self.window_ is None:
            self.window_ = PlotWindow(self)
        self.window_.setWindowTitle(f"{self.window_title}: {self.title.text()}")
        self.window_.layout().addWidget(self.view, 1)
        self.placeholder.setVisible(True)
        self.window_.show()
        self.window_.raise_()
        self.window_.activateWindow()

    def put_back(self):
        if not self.placeholder.isVisible():
            return
        self.layout().addWidget(self.view, self.row, 0, 1, 3)
        self.placeholder.setVisible(False)
        self.window_.hide()


def message_box(parent, title, text, buttons=("OK",), default=0, icon="question") -> int:
    """A simple question; returns the index of the button pressed (-1: closed)."""
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon({"question": QtWidgets.QMessageBox.Icon.Question, "warning": QtWidgets.QMessageBox.Icon.Warning,
                 "info": QtWidgets.QMessageBox.Icon.Information}[icon])
    added = [box.addButton(label, QtWidgets.QMessageBox.ButtonRole.ActionRole) for label in buttons]
    box.setDefaultButton(added[default])
    box.exec()
    clicked = box.clickedButton()
    return added.index(clicked) if clicked in added else -1
