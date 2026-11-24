"""
Plot descriptions drawn with matplotlib (the PDF export): the same marks, axes, units and scales as the window.
Imported only for exports (matplotlib takes a while to import).
"""
import numpy as np

from tesselab import units
from tesselab.colors import THEMES, black_red, mark_style, role_color
from tesselab.plotspec import Band, Heatmap, HLine, Line, Lines, Points, Region, Style, Text, VLine

THEME = THEMES["light"]
DASHES = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}
MARKERS = {"o": "o", "s": "s", "t": "^", "d": "D", "+": "+", "x": "x", "_": "_"}


def _color(role, index=0):
    return role_color(THEME, role, index)


def _display(axis):
    return axis.displays[0]


def _values(axis, values):
    display = _display(axis)
    return units.to_display(np.asarray(values, dtype=float), axis.unit, display) if display else \
        np.asarray(values, dtype=float)


def _scale(ax_set_scale, axis):
    if axis.scale == "log":
        ax_set_scale("log")
    elif axis.scale == "linlog":
        threshold = abs(float(units.to_display(axis.linthresh, axis.unit, _display(axis)))) if _display(axis) \
            else axis.linthresh
        ax_set_scale("symlog", linthresh=threshold)


def _cmap(name):
    import matplotlib
    return {"diverging": "RdBu_r", "greys": "Greys", "viridis": "viridis"}.get(name, "viridis")


def draw_panel(fig, ax, panel):
    """Draw one Panel into a matplotlib Axes."""
    ax.set_xlabel(units.label(panel.x.label, panel.x.unit, _display(panel.x)), fontsize=8)
    ax.set_ylabel(units.label(panel.y.label, panel.y.unit, _display(panel.y)), fontsize=8)
    ax.tick_params(labelsize=7)
    _scale(ax.set_xscale, panel.x)
    _scale(ax.set_yscale, panel.y)
    if panel.grid:
        ax.grid(alpha=0.3, linewidth=0.5)
    right = None
    if panel.y_right is not None:
        right = ax.twinx()
        right.set_ylabel(units.label(panel.y_right.label, panel.y_right.unit, _display(panel.y_right)), fontsize=8)
        right.tick_params(labelsize=7)
        _scale(right.set_yscale, panel.y_right)
    if panel.zero_line and panel.y.scale != "log":
        ax.axhline(0, color=THEME["zero"], linewidth=0.6, linestyle=":", zorder=0)
    for i, mark in enumerate(panel.marks):
        target = right if mark.axis == "right" and right is not None else ax
        yaxis = panel.y_right if mark.axis == "right" and panel.y_right is not None else panel.y
        _draw_mark(fig, target, mark, panel.x, yaxis, i)
    if panel.x.invert:
        ax.invert_xaxis()
    if panel.y.invert:
        ax.invert_yaxis()
    if panel.x.limits:
        ax.set_xlim(*_values(panel.x, panel.x.limits))
    if panel.y.limits:
        ax.set_ylim(*_values(panel.y, panel.y.limits))
    if panel.legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles and panel.legend == "below":
            ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        elif handles:
            location = {"top right": "upper right", "top left": "upper left", "bottom right": "lower right",
                        "bottom left": "lower left"}.get(panel.legend, "best")
            ax.legend(fontsize=6, loc=location, frameon=False)
    if panel.note:
        ax.text(0.5, 0.5, panel.note, transform=ax.transAxes, ha="center", va="center", fontsize=8,
                color=THEME["ink2"])


def _draw_mark(fig, ax, mark, xaxis, yaxis, index):
    style = Style(**mark_style(THEME, mark.style))  # its role's look in the theme, the module's own fields on top
    if isinstance(mark, Line):
        x, y = _values(xaxis, mark.x), _values(yaxis, mark.y)
        color = _color(style.color, index)
        if mark.error_y is not None:
            ax.fill_between(x, _values(yaxis, mark.y - mark.error_y), _values(yaxis, mark.y + mark.error_y),
                            color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=color, linewidth=(style.width or 1.0) * 0.8, linestyle=DASHES.get(style.dash, "-"),
                alpha=style.alpha, label=mark.name, drawstyle="steps-post" if mark.step else "default")
    elif isinstance(mark, Points):
        x, y = _values(xaxis, mark.x), _values(yaxis, mark.y)
        color = _color(style.color, index)
        yerr = None
        if mark.error_y is not None:
            yerr = [y - _values(yaxis, mark.y - mark.error_y), _values(yaxis, mark.y + mark.error_y) - y]
        xerr = None
        if mark.error_x is not None:
            xerr = [x - _values(xaxis, mark.x - mark.error_x), _values(xaxis, mark.x + mark.error_x) - x]
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=MARKERS.get(style.symbol, "o"), color=color,
                    markersize=(style.size or 7) * 0.45, elinewidth=0.6, capsize=0, alpha=style.alpha,
                    label=mark.name)
    elif isinstance(mark, Lines):
        x = _values(xaxis, mark.x)
        n = len(mark.y)
        if mark.colors is not None:
            colors = [_color(c, i) for i, c in enumerate(mark.colors)]
        elif style.color is not None:
            colors = [_color(style.color)] * n
        elif mark.colormap == "black-red":
            colors = black_red(THEME, n)
        elif mark.colormap in ("viridis", "greys"):
            import matplotlib
            cmap = matplotlib.colormaps[_cmap(mark.colormap)]
            colors = [cmap(0.1 + 0.8 * i / max(1, n - 1)) for i in range(n)]
        else:
            colors = [_color(None, i) for i in range(n)]
        for i, (row, color) in enumerate(zip(mark.y, colors)):
            label = mark.labels[i] if mark.labels and n <= 12 and i < len(mark.labels) and mark.legend else None
            ax.plot(x, _values(yaxis, row), color=color, linewidth=(style.width or 1.0) * 0.6,
                    linestyle=DASHES.get(style.dash, "-"), alpha=style.alpha, label=label)
    elif isinstance(mark, Band):
        ax.fill_between(_values(xaxis, mark.x), _values(yaxis, mark.low), _values(yaxis, mark.high),
                        color=_color(style.color, index), alpha=style.alpha if style.alpha is not None else 0.2,
                        linewidth=0)
    elif isinstance(mark, VLine):
        x = float(_values(xaxis, [mark.x])[0])
        ax.axvline(x, color=_color(style.color or "muted"), linewidth=0.8, linestyle=DASHES.get(style.dash, "--"))
        if mark.label:
            ax.text(x, 0.97, f" {mark.label}", transform=ax.get_xaxis_transform(), fontsize=6, va="top",
                    color=THEME["ink2"])
    elif isinstance(mark, HLine):
        ax.axhline(float(_values(yaxis, [mark.y])[0]), color=_color(style.color or "muted"), linewidth=0.8,
                   linestyle=DASHES.get(style.dash, "--"))
    elif isinstance(mark, Region):
        for start, stop in zip(_values(xaxis, mark.starts), _values(xaxis, mark.stops)):
            ax.axvspan(start, stop, color=_color(style.color or "muted"),
                       alpha=style.alpha if style.alpha is not None else 0.15, linewidth=0)
    elif isinstance(mark, Heatmap):
        x, y = _values(xaxis, mark.x), _values(yaxis, mark.y)
        order_x, order_y = np.argsort(x), np.argsort(y)
        low, high = mark.color_limits()
        mesh = ax.pcolormesh(_edges(x[order_x]), _edges(y[order_y]), mark.z[np.ix_(order_x, order_y)].T,
                             cmap=_cmap(mark.colormap), vmin=low, vmax=high, shading="flat", rasterized=True)
        if mark.colorbar:
            bar = fig.colorbar(mesh, ax=ax, pad=0.02)
            bar.set_label(mark.colorbar, fontsize=7)
            bar.ax.tick_params(labelsize=6)
    elif isinstance(mark, Text):
        if mark.coords == "tile":
            ax.text(mark.x, mark.y, mark.text, transform=ax.transAxes, fontsize=7, ha=mark.anchor
                    if mark.anchor in ("left", "right", "center") else "left", color=_color(style.color or "ink"))
        else:
            ax.text(float(_values(xaxis, [mark.x])[0]), float(_values(yaxis, [mark.y])[0]), mark.text, fontsize=7,
                    color=_color(style.color or "ink"))


def _edges(centers):
    c = np.asarray(centers, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    middle = (c[:-1] + c[1:]) / 2
    return np.concatenate(([c[0] - (middle[0] - c[0])], middle, [c[-1] + (c[-1] - middle[-1])]))


def draw_spec(fig, subplot_spec, spec):
    """Draw a PlotSpec into a region of a figure (a SubplotSpec), its panels in their mosaic."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    rows, columns = len(spec.mosaic), len(spec.mosaic[0])
    grid = GridSpecFromSubplotSpec(rows, columns, subplot_spec=subplot_spec, width_ratios=spec.sizes[0],
                                   height_ratios=spec.sizes[1], wspace=0.35, hspace=0.45)
    axes = {}
    for name, (row, column, row_span, column_span) in spec.cells().items():
        axes[name] = fig.add_subplot(grid[row:row + row_span, column:column + column_span])
    for group in spec.share_x:
        for name in group[1:]:
            axes[name].sharex(axes[group[0]])
    for group in spec.share_y:
        for name in group[1:]:
            axes[name].sharey(axes[group[0]])
    for name, panel in spec.panels.items():
        draw_panel(fig, axes[name], panel)
    return axes
