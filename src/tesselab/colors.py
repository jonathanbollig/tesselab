"""
The colors of the themes, without Qt: the window (gui/theme.py) and the PDF export (export.py) both use them.
role_color turns a color of a plot description ("#rrggbb", a role, "series:N", None) into "#rrggbb"; mark_style
gives a mark's look: its role's look in the theme ("marks"), with what the module set itself on top.
"""
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# how each kind of mark looks (plotspec.MARK_ROLES), as in the kufner viewer: measured points as blue crosses, a fit
# as a grey dashed line, the result in black, a reference (an expected spectrum) green, unused data grey dashed,
# events (an interrupt of the run) as grey dashed lines. Colors may be roles of the theme ("data", "muted", ...).
MARKS = {
    "measured": {"color": "data", "symbol": "x", "size": 11},
    "fit": {"color": "muted", "width": 2, "dash": "dash"},
    "result": {"color": "ink", "width": 2, "symbol": "_", "size": 14},
    "reference": {"color": "#2ca02c", "width": 2.5},
    "unused": {"color": "muted", "dash": "dash", "alpha": 0.6},
    "event": {"color": "muted", "width": 1.5, "dash": "dash"},
}

THEMES = {
    "light": {
        "dark": False,
        "chrome": "#d9d8d0", "page": "#e5e4de", "card": "#f4f4f1", "tile": "#ffffff",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "border": "#d8d7d0", "field": "#ffffff", "field_border": "#d6d5cf", "hover": "#f0efeb", "tab_hover": "#dfded7",
        "accent": "#1f77b4", "accent_light": "#e3eef8", "amber": "#d18a00",
        "node_reader": "#e9eff6", "node_view": "#eaf3e4",  # the graph's boxes: readers bluish, views greenish
        "plot_axis": "#a3a297", "ticks": "#6b6a64", "zero": "#7d7c75", "grid_alpha": 0.3,
        "roles": {"data": "#1f77b4", "fit": "#d62728", "accent": "#1f77b4", "warning": "#d18a00", "error": "#c62828",
                  "good": "#2e9d48", "muted": "#9a9993", "ink": "#0b0b0b", "positive": "#d62728", "negative": "#1f77b4"},
        "series": TAB10,
        "plot": {"line_width": 1.0, "point_size": 7},
        "marks": MARKS,
    },
    "dark": {
        "dark": True,
        "chrome": "#1d1d1b", "page": "#252523", "card": "#2c2c29", "tile": "#1a1a18",
        "ink": "#ecebe6", "ink2": "#b9b8b1", "muted": "#8a8983",
        "border": "#3b3b37", "field": "#1f1f1d", "field_border": "#45453f", "hover": "#34342f", "tab_hover": "#2f2f2b",
        "accent": "#5aa9e6", "accent_light": "#22384a", "amber": "#e0a030",
        "node_reader": "#1f2731", "node_view": "#212b1e",
        "plot_axis": "#6f6e66", "ticks": "#a7a69e", "zero": "#85847c", "grid_alpha": 0.25,
        "roles": {"data": "#5aa9e6", "fit": "#ff6b6b", "accent": "#5aa9e6", "warning": "#e0a030", "error": "#ff6b6b",
                  "good": "#4cc26a", "muted": "#7d7c75", "ink": "#ecebe6", "positive": "#ff6b6b", "negative": "#5aa9e6"},
        "series": ["#5aa9e6", "#ffa552", "#57c96b", "#ff6b6b", "#b394d9", "#c9907c", "#f09ad2", "#a8a8a8", "#d4d548",
                   "#4fd3e2"],
        "plot": {"line_width": 1.0, "point_size": 7},
        "marks": {**MARKS, "reference": {"color": "#57c96b", "width": 2.5}},
    },
}



def role_color(t, role, index=0) -> str:
    """A color of a plot description in theme t: "#rrggbb", a role ("data", "fit", ...), "series:N", or None (the
    series color number `index`)."""
    if role is None:
        return t["series"][index % len(t["series"])]
    if role.startswith("#"):
        return role
    if role.startswith("series:"):
        return t["series"][int(role[7:]) % len(t["series"])]
    return t["roles"].get(role, t["roles"]["data"])


def black_red(t, n) -> list[str]:
    """n colors from black (light grey on a dark theme) to red: spectra in the order they were measured."""
    start = (200, 200, 200) if t["dark"] else (0, 0, 0)
    end = (255, 0, 0)  # pure red, as the kufner viewer's black-to-red
    out = []
    for i in range(n):
        f = i / max(1, n - 1)
        out.append("#%02x%02x%02x" % tuple(int(a + f * (b - a)) for a, b in zip(start, end)))
    return out


def mark_style(t, style) -> dict:
    """The look of a mark in theme t: its role's look (t["marks"]), with every field the mark's Style sets itself on
    top (a module's own choice wins). Returns the fields of a Style, without the role."""
    look = dict(t.get("marks", {}).get(style.role, {})) if style.role else {}
    look.update({k: v for k, v in style.__dict__.items() if v is not None and k != "role"})
    return look
