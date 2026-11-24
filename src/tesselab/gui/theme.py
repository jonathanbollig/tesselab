"""
The look: colors, Qt style sheet, line icons, and the colors behind the plot roles ("data", "fit", "series:3", ...).

Two themes come with Tesselab, light and dark; users can add their own as TOML files in the user's config folder
(platformdirs: ~/.config/tesselab/themes/<name>.toml on Linux):

    base = "dark"            # the theme to start from
    accent = "#e07b39"
    [roles]
    fit = "#ffd166"
    [plot]
    line_width = 1.5
    [marks.fit]              # how a kind of mark looks in every module (colors.MARKS; plotspec.MARK_ROLES)
    color = "fit"
    dash = "dot"
    width = 1.5

The quiet grey chrome, the white (or dark) tiles and the data as the only loud part are taken from the kufner viewer,
where they were chosen from rendered variants. Fusion style everywhere, so it looks the same on every desktop.
"""
import atexit
import os
import tomllib

import platformdirs
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from tesselab.colors import THEMES, TAB10, mark_style as _mark_style, role_color as _role_color

_current = dict(THEMES["light"])


def themes_folder() -> str:
    return os.path.join(platformdirs.user_config_dir("tesselab", appauthor=False), "themes")


def available() -> list[str]:
    names = list(THEMES)
    folder = themes_folder()
    if os.path.isdir(folder):
        names += sorted(os.path.splitext(n)[0] for n in os.listdir(folder) if n.endswith(".toml"))
    return names


def load(name) -> dict:
    """A theme by name: built in, or a user's TOML file on top of its base theme."""
    if name in THEMES:
        return dict(THEMES[name])
    path = os.path.join(themes_folder(), f"{name}.toml")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return dict(THEMES["light"])
    theme = dict(THEMES.get(data.get("base", "light"), THEMES["light"]))
    for key, value in data.items():
        if key in ("roles", "plot") and isinstance(value, dict):
            theme[key] = {**theme[key], **value}
        elif key == "marks" and isinstance(value, dict):  # per role, on top of the base theme's look
            theme["marks"] = {role: {**theme["marks"].get(role, {}), **(look if isinstance(look, dict) else {})}
                              for role, look in {**theme["marks"], **value}.items()}
        elif key != "base":
            theme[key] = value
    return theme


def current() -> dict:
    return _current


def color(name) -> str:
    return _current[name]


def role_color(role, index=0) -> str:
    """A color given in a plot description: "#rrggbb", a role, "series:N", or None (the series color of index)."""
    return _role_color(_current, role, index)


def mark_style(style):
    """A mark's Style as the current theme shows it: its role's look, the module's own fields on top."""
    from tesselab.plotspec import Style
    return Style(**_mark_style(_current, style)) if style.role else style


# Corner radii, one scale for the whole window: the larger the thing, the rounder; a thing inside another one is less
# round than its container (panel > card > tile > control), so that the curves run parallel.
PANEL_R, CARD_R, TILE_R, CONTROL_R = 10, 8, 6, 5
GAP = 6  # the chrome around the panels (explorer, tabs' page)


def style_sheet(t) -> str:
    return f"""
QMainWindow {{ background: {t['chrome']}; }}
#page {{ background: {t['page']}; }}
QWidget {{ color: {t['ink']}; }}
QMenuBar {{ background: {t['chrome']}; color: {t['ink']}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 3px 8px; border-radius: {CONTROL_R}px; background: transparent; }}
QMenuBar::item:selected {{ background: {t['tab_hover']}; }}
QMenu {{ background: {t['card']}; color: {t['ink']}; border: 1px solid {t['border']}; border-radius: {CARD_R}px;
         padding: 4px; }}
QMenu::item {{ padding: 4px 22px 4px 22px; border-radius: {CONTROL_R}px; }}
QMenu::item:selected {{ background: {t['accent_light']}; color: {t['ink']}; }}
QMenu::item:disabled {{ color: {t['muted']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
QMenu::indicator {{ width: 12px; height: 12px; left: 6px; }}
QMenu::indicator:checked {{ image: url({t.get('check_icon', '')}); }}
QMenu::right-arrow {{ image: url({t.get('chevron_icon', '')}); width: 12px; height: 12px; right: 6px; }}
QToolBar {{ background: {t['chrome']}; border: none; }}
QStatusBar {{ background: {t['chrome']}; color: {t['ink2']}; }}
QSplitter::handle {{ background: {t['chrome']}; }}
#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: {CARD_R}px; }}
#card[active="true"] {{ border-color: {t['accent']}; }}
#tile {{ background: {t['tile']}; border: 1px solid {t['border']}; border-radius: {TILE_R}px; }}
#label {{ font-size: 16px; font-weight: 600; }}
#heading {{ font-size: 13.5px; font-weight: 600; }}
#field {{ color: {t['ink2']}; }}
#hint {{ color: {t['muted']}; }}
#comment {{ font-style: italic; font-weight: 600; }}
#info {{ color: {t['muted']}; }}
#unsaved {{ color: {t['amber']}; font-size: 15px; }}
#keyresult {{ color: {t['ink']}; font-size: 13px; }}
#section {{ font-size: 11px; font-weight: 600; color: {t['muted']}; letter-spacing: 0.6px; }}
#note {{ background: {'#3a301c' if t['dark'] else '#fdf3e0'}; color: {'#f0d090' if t['dark'] else '#6b4800'};
        border: 1px solid {'#5a4a28' if t['dark'] else '#f0d9a8'}; border-radius: {CONTROL_R}px; padding: 4px 6px; }}
#error {{ background: {'#3d2220' if t['dark'] else '#fbe9e7'}; color: {'#ffb4ab' if t['dark'] else '#8c1d18'};
         border: 1px solid {'#5c2b27' if t['dark'] else '#f3c3bd'}; border-radius: {CONTROL_R}px; padding: 4px 6px; }}
QLineEdit, QPlainTextEdit {{ background: {t['field']}; color: {t['ink']}; border: 1px solid {t['field_border']};
                             border-radius: {CONTROL_R}px; padding: 1px 5px; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {t['accent']}; }}
#sidebar QLineEdit {{ padding: 3px 4px; }}
QComboBox {{ background: {t['field']}; color: {t['ink']}; border: 1px solid {t['field_border']};
             border-radius: {CONTROL_R}px; padding: 1px 6px; }}
QComboBox:hover {{ border-color: {t['plot_axis']}; }}
QComboBox:focus {{ border-color: {t['accent']}; }}
QComboBox:disabled {{ color: {t['muted']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{ image: url({t.get('down_icon', '')}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {t['card']}; color: {t['ink']}; border: 1px solid {t['border']};
                               border-radius: {TILE_R}px; padding: 3px; outline: 0;
                               selection-background-color: {t['accent_light']}; selection-color: {t['ink']}; }}
QComboBox QAbstractItemView::item {{ padding: 3px 6px; border-radius: {CONTROL_R - 1}px; border: none; }}
QComboBox QAbstractItemView::item:selected, QComboBox QAbstractItemView::item:hover {{
    background: {t['accent_light']}; color: {t['ink']}; }}
QCheckBox {{ color: {t['ink']}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t['field_border']}; border-radius: 4px;
                        background: {t['field']}; }}
QCheckBox::indicator:hover {{ border-color: {t['plot_axis']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']};
                                image: url({t.get('check_white_icon', '')}); }}
#project_label {{ font-size: 10.5px; font-weight: 600; color: {t['muted']}; letter-spacing: 0.8px; }}
#project_name {{ font-size: 15px; font-weight: 600; }}
#sidebar QTreeWidget {{ border: none; background: transparent; outline: 0; }}
#sidebar QTreeWidget::item {{ padding: 2px 4px; border-radius: {CONTROL_R}px; }}
#sidebar QTreeWidget::item:hover {{ background: {t['hover']}; }}
#sidebar QTreeWidget::item:selected {{ background: {t['card']}; color: {t['accent']}; }}
QTabBar::tab {{ background: transparent; border: none; border-top-left-radius: {CARD_R}px;
               border-top-right-radius: {CARD_R}px; padding: 6px 10px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {t['page']}; }}
QTabBar::tab:!selected:hover {{ background: {t['tab_hover']}; }}
QTabBar {{ background: transparent; border: none; }}
QScrollArea {{ background: {t['page']}; border: none; }}
#tab_page {{ border-radius: {PANEL_R}px; }}
#tab_page QScrollBar:vertical {{ margin: {PANEL_R}px 0; }}
#tab_page QScrollBar:horizontal {{ margin: 0 {PANEL_R}px; }}
QGraphicsView {{ border: none; }}
QAbstractScrollArea::corner {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle {{ background: {'#4a4a45' if t['dark'] else '#c9c8c0'}; border-radius: 3px; margin: 2px; }}
QScrollBar::handle:vertical {{ min-height: 30px; }}
QScrollBar::handle:horizontal {{ min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
#icon {{ border: none; border-radius: {CONTROL_R}px; padding: 2px; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QTabBar::close-button {{ image: url({t.get('close_icon', '')}); subcontrol-position: right; border-radius: 4px; }}
QTabBar::close-button:hover {{ background: {t['tab_hover']}; }}
#icon:hover {{ background: {t['hover']}; }}
#icon[unsaved="true"] {{ background: {t['amber']}; }}
#icon[unsaved="true"]:hover {{ background: {t['amber']}; }}
QToolTip {{ background: {t['card']}; color: {t['ink']}; border: 1px solid {t['border']}; border-radius: {TILE_R}px;
            padding: 4px; }}
QPushButton {{ background: {t['field']}; color: {t['ink']}; border: 1px solid {t['field_border']};
               border-radius: {CONTROL_R}px; padding: 3px 10px; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:checked {{ background: {t['accent_light']}; color: {t['accent']}; border-color: {t['accent']}; }}
QPushButton:flat {{ border: none; background: transparent; }}
QPushButton:disabled {{ color: {t['muted']}; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}
#group {{ font-size: 14px; font-weight: 600; text-align: left; padding: 3px 4px 3px 0; border: none; background: transparent; }}
#group:hover {{ color: {t['accent']}; }}
#add {{ color: {t['accent']}; border-color: {t['accent']}; }}
#primary {{ background: {t['accent']}; color: white; border-color: {t['accent']}; }}
#amber {{ background: {t['amber']}; color: white; border-color: {t['amber']}; }}
QListWidget {{ background: {t['field']}; border: 1px solid {t['field_border']}; border-radius: {CONTROL_R}px;
               outline: 0; padding: 2px; }}
QListWidget::item {{ padding: 3px 6px; border-radius: {CONTROL_R - 1}px; }}
QListWidget::item:selected {{ background: {t['accent_light']}; color: {t['ink']}; }}
#palette {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: {CARD_R}px; }}
"""


class _RoundPopups(QtCore.QObject):
    """Popups (menus, tool tips, the lists of drop-downs) are windows of their own: their round corners show only if
    the window around them is transparent. Qt makes these windows itself, so they are caught when they are polished,
    which is before their window exists."""
    CLASSES = ("QMenu", "QTipLabel", "QComboBoxPrivateContainer")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Polish and isinstance(obj, QtWidgets.QComboBox) and \
                not isinstance(obj.itemDelegate(), QtWidgets.QStyledItemDelegate):
            # the list's items as the style sheet says: Fusion's own delegate draws them as menu items and drew the
            # current one as a framed white box, like a text field (the user, 2026-11-23)
            obj.setItemDelegate(QtWidgets.QStyledItemDelegate(obj))
        if event.type() == QtCore.QEvent.Type.Polish and obj.metaObject().className() in self.CLASSES:
            obj.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            if obj.metaObject().className() == "QComboBoxPrivateContainer":
                # the list inside draws the box. With a selector: a rule without one applies to the children too,
                # and the list lost its background (the popup was see-through, the user 2026-11-23)
                obj.setStyleSheet("QComboBoxPrivateContainer { background: transparent; border: none; }")
        return False


def apply(app, name="light"):
    """Use the theme `name` for the whole application (Fusion style, palette, style sheet)."""
    global _current
    _current = load(name)
    t = _current
    t["close_icon"] = _icon_file("close", t["muted"])
    t["down_icon"] = _icon_file("chevron_down", t["ink2"])
    t["chevron_icon"] = _icon_file("chevron_right", t["ink2"])
    t["check_icon"] = _icon_file("check", t["ink"])
    t["check_white_icon"] = _icon_file("check", "#ffffff")
    app.setStyle("Fusion")
    if not hasattr(app, "_round_popups"):
        app._round_popups = _RoundPopups(app)
        app.installEventFilter(app._round_popups)
        # Python filters must be gone before Python shuts down: Qt still filters the events of the widgets it
        # deletes at exit, and calling a Python filter then made shiboken allocate memory without end (25 GB)
        atexit.register(app.removeEventFilter, app._round_popups)
    palette = QtGui.QPalette()
    for role, key in ((QtGui.QPalette.ColorRole.Window, "page"), (QtGui.QPalette.ColorRole.Base, "field"),
                      (QtGui.QPalette.ColorRole.AlternateBase, "card"), (QtGui.QPalette.ColorRole.Button, "field"),
                      (QtGui.QPalette.ColorRole.Text, "ink"), (QtGui.QPalette.ColorRole.WindowText, "ink"),
                      (QtGui.QPalette.ColorRole.ButtonText, "ink"), (QtGui.QPalette.ColorRole.ToolTipBase, "card"),
                      (QtGui.QPalette.ColorRole.ToolTipText, "ink"), (QtGui.QPalette.ColorRole.Highlight, "accent"),
                      (QtGui.QPalette.ColorRole.PlaceholderText, "muted")):
        palette.setColor(role, QtGui.QColor(t[key]))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(style_sheet(t))
    import pyqtgraph as pg
    pg.setConfigOptions(background=t["tile"], foreground=t["ticks"], antialias=True)
    _icon_cache.clear()


def _icon_file(name, color) -> str:
    """A line icon as a PNG file (style sheets can only use image files), in the user's cache folder."""
    folder = platformdirs.user_cache_dir("tesselab", appauthor=False)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_{color.lstrip('#')}.png").replace("\\", "/")
    if not os.path.isfile(path):
        icon(name, color).pixmap(48, 48).save(path)
    return path


# --- line icons (grey, drawn in code: Qt's own icons are colorful and differ between systems) -------------------
_icon_cache = {}


def line_icon(draw, color=None, key=None) -> QtGui.QIcon:
    """A line icon: draw(painter) draws on a 24 × 24 grid (in 4× the pixels, sharp on scaled screens)."""
    cache_key = (key, color or _current["ink2"]) if key else None
    if cache_key and cache_key in _icon_cache:
        return _icon_cache[cache_key]
    pixmap = QtGui.QPixmap(96, 96)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.scale(4, 4)
    painter.setPen(QtGui.QPen(QtGui.QColor(color or _current["ink2"]), 1.6, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    draw(painter)
    painter.end()
    icon = QtGui.QIcon(pixmap)
    if cache_key:
        _icon_cache[cache_key] = icon
    return icon


def _poly(p, points, closed=False):
    polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
    (p.drawPolygon if closed else p.drawPolyline)(polygon)


def icon(name, color=None) -> QtGui.QIcon:
    """The icons the window uses, by name."""
    def draw(p):
        if name == "folder":
            _poly(p, ((3, 6), (9, 6), (11, 8), (21, 8), (21, 19), (3, 19)), closed=True)
        elif name == "save":
            p.drawRoundedRect(QtCore.QRectF(4, 4, 16, 16), 1.5, 1.5)
            p.drawRect(QtCore.QRectF(8, 4, 8, 5))
            p.drawRect(QtCore.QRectF(7.5, 13, 9, 7))
        elif name == "save_all":  # two disks, one behind the other
            _poly(p, ((3, 16), (3, 3), (16, 3)))
            p.drawRoundedRect(QtCore.QRectF(7, 7, 14, 14), 1.5, 1.5)
            p.drawRect(QtCore.QRectF(10.5, 7, 7, 4.5))
            p.drawRect(QtCore.QRectF(10, 15, 8, 6))
        elif name == "clock":
            p.drawEllipse(QtCore.QPointF(12, 12), 8, 8)
            _poly(p, ((12, 7.5), (12, 12), (15, 14)))
        elif name == "export":
            _poly(p, ((4, 14), (4, 20), (20, 20), (20, 14)))
            p.drawLine(QtCore.QPointF(12, 4), QtCore.QPointF(12, 15))
            _poly(p, ((8, 8), (12, 4), (16, 8)))
        elif name == "graph":
            p.drawRoundedRect(QtCore.QRectF(2.5, 4, 7, 5), 1.2, 1.2)
            p.drawRoundedRect(QtCore.QRectF(2.5, 15, 7, 5), 1.2, 1.2)
            p.drawRoundedRect(QtCore.QRectF(14.5, 9.5, 7, 5), 1.2, 1.2)
            p.drawLine(QtCore.QPointF(9.5, 6.5), QtCore.QPointF(14.5, 11))
            p.drawLine(QtCore.QPointF(9.5, 17.5), QtCore.QPointF(14.5, 13))
        elif name == "link":
            p.translate(12, 12)
            p.rotate(-45)
            for x in (-4.2, 4.2):
                p.drawRoundedRect(QtCore.QRectF(x - 5, -3, 10, 6), 3, 3)
        elif name == "close":
            p.drawLine(QtCore.QPointF(8, 8), QtCore.QPointF(16, 16))
            p.drawLine(QtCore.QPointF(16, 8), QtCore.QPointF(8, 16))
        elif name == "refresh":
            p.drawArc(QtCore.QRectF(5, 5, 14, 14), 90 * 16, 270 * 16)
            _poly(p, ((16, 11), (19, 8), (22, 11)))
        elif name == "plus":
            p.drawLine(QtCore.QPointF(12, 6), QtCore.QPointF(12, 18))
            p.drawLine(QtCore.QPointF(6, 12), QtCore.QPointF(18, 12))
        elif name == "chevron_down":
            _poly(p, ((8, 10), (12, 14), (16, 10)))
        elif name == "chevron_right":
            _poly(p, ((10, 8), (14, 12), (10, 16)))
        elif name == "check":
            _poly(p, ((6, 12.5), (10, 16.5), (18, 7.5)))
        elif name == "play":
            _poly(p, ((8, 6), (18, 12), (8, 18)), closed=True)
        elif name == "stop":
            p.drawRect(QtCore.QRectF(7, 7, 10, 10))
        elif name == "sidebar":
            p.drawRoundedRect(QtCore.QRectF(3, 5, 18, 14), 2, 2)
            p.drawLine(QtCore.QPointF(9, 5), QtCore.QPointF(9, 19))
        elif name == "trash":
            _poly(p, ((6, 8), (7, 20), (17, 20), (18, 8)))
            p.drawLine(QtCore.QPointF(4, 8), QtCore.QPointF(20, 8))
            _poly(p, ((9, 8), (9, 5), (15, 5), (15, 8)))
        elif name == "search":
            p.drawEllipse(QtCore.QPointF(10.5, 10.5), 5.5, 5.5)
            p.drawLine(QtCore.QPointF(14.6, 14.6), QtCore.QPointF(19.5, 19.5))
        elif name == "menu":
            for y in (7, 12, 17):
                p.drawLine(QtCore.QPointF(5, y), QtCore.QPointF(19, y))
        elif name in ("undo", "redo"):  # an arrow to the left that comes from a turn below; redo mirrored
            if name == "redo":
                p.translate(24, 0)
                p.scale(-1, 1)
            path = QtGui.QPainterPath(QtCore.QPointF(5, 9))
            path.lineTo(14, 9)
            path.arcTo(QtCore.QRectF(9, 9, 10, 10), 90, -180)
            path.lineTo(9, 19)
            p.drawPath(path)
            _poly(p, ((9, 5), (5, 9), (9, 13)))
        elif name == "info":
            p.drawEllipse(QtCore.QPointF(12, 12), 8, 8)
            p.drawLine(QtCore.QPointF(12, 11), QtCore.QPointF(12, 16))
            p.drawPoint(QtCore.QPointF(12, 8))
    return line_icon(draw, color, key=name)


def dot_icon(level) -> QtGui.QIcon:
    """The status dot: green when everything ran without warnings, else the color of the worst level."""
    colors = {None: _current["roles"]["good"], "info": _current["roles"]["good"],
              "warning": _current["roles"]["warning"], "error": _current["roles"]["error"],
              "running": _current["accent"], "waiting": _current["muted"]}
    pixmap = QtGui.QPixmap(48, 48)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(colors.get(level, _current["muted"])))
    if level == "running":
        painter.setPen(QtGui.QPen(QtGui.QColor(colors["running"]), 6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QtCore.QRectF(8, 8, 32, 32), 0, 270 * 16)
    else:
        painter.drawEllipse(QtCore.QRectF(10, 10, 28, 28))
    painter.end()
    return QtGui.QIcon(pixmap)


def app_icon() -> QtGui.QIcon:
    """A small mosaic of tiles (tessera) with a node graph on top."""
    pixmap = QtGui.QPixmap(256, 256)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pixmap)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    colors = ["#1f77b4", "#5aa9e6", "#a9cdee", "#d9d8d0"]
    for i in range(3):
        for j in range(3):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(colors[(i + 2 * j) % 4]))
            p.drawRoundedRect(QtCore.QRectF(22 + i * 72, 22 + j * 72, 64, 64), 12, 12)
    p.end()
    return QtGui.QIcon(pixmap)


def icon_button(name, tooltip, checkable=False) -> QtWidgets.QToolButton:
    """A tool button with a line icon. It keeps the icon's name (property icon_name), so that recolor_icons can draw
    it anew in another theme's colors."""
    button = QtWidgets.QToolButton(toolTip=tooltip, objectName="icon", autoRaise=True, checkable=checkable)
    button.setProperty("icon_name", name)
    button.setIcon(_button_icon(name, checkable))
    button.setIconSize(QtCore.QSize(18, 18))
    return button


def _button_icon(name, checkable) -> QtGui.QIcon:
    """The icon of a button: in ink2, faint when the button is unavailable, in the accent color when switched on.
    The faint one is drawn here: Qt's own (the icon greyed) looked like the usable one on the dark theme's page."""
    result = QtGui.QIcon(icon(name))  # a copy: the cached icon stays as it is
    ink, card = QtGui.QColor(_current["ink2"]), QtGui.QColor(_current["card"])
    faint = QtGui.QColor(*(round(0.35 * a + 0.65 * b) for a, b in zip(ink.getRgb()[:3], card.getRgb()[:3])))
    result.addPixmap(icon(name, faint.name()).pixmap(96, 96), QtGui.QIcon.Mode.Disabled)
    if checkable:
        result.addPixmap(icon(name, _current["accent"]).pixmap(96, 96), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
    return result


def recolor_icons():
    """After a change of theme: every icon button of every window in the new colors. Icons are pixmaps drawn in the
    colors of their day; without this the card headers kept the old theme's ink (dark grey on dark, 2026-11-23).
    Icons that show a state (the save button with unsaved changes) are set again by their owners afterwards."""
    for window in QtWidgets.QApplication.topLevelWidgets():
        for button in window.findChildren(QtWidgets.QToolButton):
            name = button.property("icon_name")
            if name:
                button.setIcon(_button_icon(name, button.isCheckable()))
