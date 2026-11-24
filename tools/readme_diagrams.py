"""
The README's diagrams (docs/images/*.svg) as plain SVG files (no scripts, no styles, no external resources):

    python tools/readme_diagrams.py [output folder, default docs/images]

Each diagram is one function; edit the numbers and texts there and run it again.
"""
import os
import sys
from xml.sax.saxutils import escape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "images")
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans', 'DejaVu Sans', sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, LINE = "#23262b", "#6b7078", "#7b8088"
ACCENT, ORANGE = "#1f6fb5", "#d9480f"
KINDS = {  # fill, stroke (the graph window's shades: readers bluish, computation plain, views greenish)
    "reader": ("#e6edf8", "#7f9cc9"), "compute": ("#ffffff", "#8b9097"), "view": ("#e5f0df", "#86ad7c"),
    "module": ("#fbf1dc", "#d1a34a"), "plain": ("#ffffff", "#aab0b8"), "reused": ("#efeeea", "#c9c7c0"),
    "band": ("#eceef2", "#c3c7ce"),
}


class Svg:
    def __init__(self, width, height, claim):
        self.w, self.h, self.claim, self.parts = width, height, claim, []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, kind="plain", r=8, stroke=None, sw=1.5, dash=None, fill=None):
        f, s = KINDS[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or f}" '
                 f'stroke="{stroke or s}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=14, weight=400, anchor="start", fill=INK, mono=False, italic=False):
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{style}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, arrow=True, color=LINE, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, arrow=True, color=LINE, sw=1.6, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dd}{m}/>')

    def badge(self, x, y, n):
        self.add(f'<circle cx="{x}" cy="{y}" r="11" fill="{ORANGE}"/>')
        self.text(x, y + 4.5, str(n), 13, 700, "middle", "#ffffff")

    def node(self, x, y, w, h, kind, title, sub=None, tsize=13.5):
        self.rect(x, y, w, h, kind, r=7)
        self.text(x + 10, y + 20, title, tsize, 700, fill=INK if kind != "reused" else MUTED)
        if sub:
            self.text(x + 10, y + 37, sub, 11.5, 400, fill=MUTED)

    def save(self, name):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}" role="img" aria-label="{escape(self.claim)}">\n'
               f'<title>{escape(self.claim)}</title>\n'
               '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
               f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{LINE}"/></marker></defs>\n'
               f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="14" fill="#f7f6f3" '
               'stroke="#d8d5cd"/>\n' + body + "\n</svg>\n")
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)


def mini_plot(s, x, y, w, h, shape):
    s.rect(x, y, w, h, "plain", r=4, sw=1)
    pts = {"line": [(0.1, 0.7), (0.3, 0.3), (0.5, 0.4), (0.7, 0.2), (0.9, 0.25)],
           "bumps": [(0.1, 0.8), (0.3, 0.75), (0.45, 0.25), (0.6, 0.7), (0.9, 0.8)],
           "cross": [(0.1, 0.3), (0.3, 0.4), (0.5, 0.5), (0.7, 0.62), (0.9, 0.75)],
           "rise": [(0.1, 0.85), (0.4, 0.8), (0.6, 0.45), (0.8, 0.3), (0.9, 0.28)]}[shape]
    poly = " ".join(f"{x + px * w:.1f},{y + py * h:.1f}" for px, py in pts)
    s.add(f'<polyline points="{poly}" fill="none" stroke="{ACCENT}" stroke-width="1.6"/>')


# --- 1. the big picture ------------------------------------------------------------------------------------------
def big_picture():
    s = Svg(1000, 470, "Modules supply nodes and templates; a template turns a data folder into an analysis, a small "
                       "graph of nodes; Tesselab runs it and shows one card per analysis in a tab of its window.")
    # data
    s.rect(24, 172, 160, 106, "plain", r=10)
    s.text(40, 200, "Your data", 15, 700)
    s.text(40, 222, "instrument files,", 12.5, fill=MUTED)
    s.text(40, 240, "a folder per run", 12.5, fill=MUTED)
    s.text(40, 262, "only ever read", 12.5, fill=MUTED, italic=True)
    s.line(184, 225, 250, 225)
    s.text(217, 215, "template", 12, 700, "middle", MUTED)
    # module
    s.rect(270, 26, 440, 84, "module", r=10)
    s.text(490, 54, "Module", 15, 700, "middle")
    s.text(490, 75, "a folder of Python: nodes and templates", 13, anchor="middle")
    s.text(490, 94, "written by a person or by an LLM, from one protocol file", 12, anchor="middle", fill=MUTED)
    s.line(490, 110, 490, 148)
    s.text(500, 134, "supplies the nodes", 12, 700, fill=MUTED)
    # analysis
    s.rect(250, 150, 468, 210, "plain", r=12, stroke=ACCENT, sw=2)
    s.text(270, 176, "Analysis", 15, 700)
    s.text(350, 176, "a small graph of nodes, saved as one JSON file", 12.5, fill=MUTED)
    rows = (200, 262)
    for y in rows:
        for x, kind, label in ((280, "reader", "reader"), (440, "compute", "compute node"), (600, "view", "view")):
            s.rect(x, y, 100, 34, kind, r=7)
            s.text(x + 50, y + 22, label, 13, 600, "middle")
    for y in rows:
        s.line(380, y + 17, 438, y + 17)
        s.line(540, y + 17, 598, y + 17)
    s.line(380, 279, 438, 221)
    s.line(490, 234, 490, 260)
    s.line(540, 217, 598, 275)
    s.text(484, 323, "Wires join an output to an input of matching shape.", 12.5, anchor="middle", fill=MUTED)
    s.text(484, 342, "Readers feed compute nodes; views show what any node produced.", 12.5, anchor="middle", fill=MUTED)
    # window
    s.rect(800, 150, 176, 210, "plain", r=12)
    s.text(888, 172, "Window", 15, 700, "middle")
    s.rect(808, 182, 56, 17, "plain", r=6, sw=1)
    s.text(836, 194.5, "tab", 11, 400, "middle", MUTED)
    s.rect(868, 182, 56, 17, "reused", r=6, sw=1)
    s.rect(808, 202, 160, 122, "plain", r=8, stroke=ACCENT, sw=1.6)
    s.add(f'<circle cx="820" cy="214" r="3.5" fill="#2e9e4f"/>')
    s.text(828, 218, "card", 11.5, 700, fill=INK)
    for (px, py), shape in (((814, 224), "bumps"), ((892, 224), "cross"), ((814, 276), "rise"), ((892, 276), "line")):
        mini_plot(s, px, py, 70, 42, shape)
    s.text(888, 346, "one card per analysis", 12, anchor="middle", fill=MUTED)
    s.line(718, 255, 798, 255)
    s.text(758, 230, "plots,", 12, 700, "middle", MUTED)
    s.text(758, 244, "parameters", 12, 700, "middle", MUTED)
    # what Tesselab does
    s.rect(24, 396, 952, 54, "band", r=10)
    s.text(500, 418, "Tesselab does everything around the nodes", 14, 700, "middle")
    s.text(500, 438, "forms for the parameters · drawing of the plots · background workers · cache · undo and redo · "
                     "links between analyses · PDF and JSON export", 12.5, anchor="middle", fill=MUTED)
    s.line(490, 396, 490, 364)
    s.text(500, 384, "runs it, draws it", 12, 700, fill=MUTED)
    s.save("big-picture.svg")


# --- 2. the three kinds ------------------------------------------------------------------------------------------
def kinds():
    s = Svg(1000, 330, "Three kinds of nodes: readers read files, compute nodes compute, views show results; "
                       "Datasets travel along the wires and only views have plots.")
    cols = [(90, "reader", "Reader", [("does", "reads a file or folder"), ("takes", "a file path"),
                                      ("gives", "the data, as measured"), ("plots", "none")]),
            (390, "compute", "Compute node", [("does", "corrects, fits, unmixes"),
                                              ("takes", "Datasets, parameters"),
                                              ("gives", "Datasets"), ("plots", "none")]),
            (690, "view", "View", [("does", "shows results"), ("takes", "Datasets of any stage"),
                                   ("gives", "nothing"), ("plots", "yes, with interactions")])]
    for x, kind, title, rows in cols:
        s.rect(x, 50, 236, 176, kind, r=10)
        s.text(x + 14, 78, title, 16, 700)
        s.line(x + 14, 90, x + 222, 90, arrow=False, color=KINDS[kind][1], sw=1)
        for i, (k, v) in enumerate(rows):
            y = 116 + i * 26
            s.text(x + 14, y, k, 12, 400, fill=MUTED)
            s.text(x + 60, y, v, 13)
    s.line(24, 138, 88, 138)
    s.text(56, 128, "files", 12, 700, "middle", MUTED)
    s.line(326, 138, 388, 138)
    s.text(357, 128, "Dataset", 12, 700, "middle", MUTED)
    s.line(626, 138, 688, 138)
    s.text(657, 128, "Datasets", 12, 700, "middle", MUTED)
    s.line(808, 226, 808, 256)
    s.text(820, 246, "plot descriptions", 12, 700, fill=MUTED)
    s.rect(690, 258, 236, 46, "band", r=8)
    s.text(808, 278, "Tesselab draws them", 13, 700, "middle")
    s.text(808, 295, "window · PDF · JSON", 12, anchor="middle", fill=MUTED)
    s.text(90, 262, "A Dataset is a bundle of labelled arrays with units", 13)
    s.text(90, 281, "(xarray). It is the only thing that travels along a wire,", 13)
    s.text(90, 300, "so every output can be cached, exported and previewed.", 13)
    s.save("node-kinds.svg")


# --- 3. drag a cursor: what runs ---------------------------------------------------------------------------------
def loop():
    s = Svg(1000, 500, "Dragging the wavelength line changes one analysis variable; only the nodes that depend on it "
                       "run again, all others are reused, and the plot replaces just the line that moved.")
    # the variable
    s.rect(560, 28, 250, 36, "plain", r=18, stroke="#1f77b4", sw=1.8)
    s.add('<path d="M582,46 l8,-8 l8,8 l-8,8 z" fill="#1f77b4"/>')
    s.text(606, 51, "λ spectrum  270 → 271 nm", 13.5, 700, fill="#1f77b4")
    s.text(560, 84, "an analysis variable", 11.5, fill=MUTED, italic=True)
    # nodes
    ys = 150
    s.node(20, ys, 130, 46, "reused", "Spectra file", "reused")
    s.node(190, ys, 150, 46, "reused", "Spectrum corrections", "reused")
    s.node(380, ys, 140, 46, "reused", "Irradiation steps", "reused")
    s.node(570, ys, 140, 46, "compute", "Quantum yield", "runs again")
    s.rect(570, ys, 140, 46, "compute", r=7, stroke=ORANGE, sw=2.4)
    s.text(580, ys + 20, "Quantum yield", 13.5, 700)
    s.text(580, ys + 37, "runs again", 11.5, 700, fill=ORANGE)
    s.node(570, 240, 140, 46, "reused", "φ₀ vs. λ spectrum", "reused: no λ in it")
    s.rect(770, ys, 190, 136, "view", r=7, stroke=ORANGE, sw=2.4)
    s.text(782, ys + 20, "Quantum yield view", 13.5, 700)
    s.text(782, ys + 37, "redraws only the λ line", 11.5, 700, fill=ORANGE)
    # a tiny plot in the view
    s.rect(782, ys + 50, 166, 76, "plain", r=4, sw=1)
    s.add(f'<polyline points="790,{ys + 108} 815,{ys + 66} 840,{ys + 80} 870,{ys + 112} 900,{ys + 116} 940,{ys + 118}" '
          f'fill="none" stroke="{ACCENT}" stroke-width="1.6"/>')
    s.line(858, ys + 54, 858, ys + 122, arrow=False, color="#c92a2a", sw=2.2)
    s.line(864, ys + 54, 864, ys + 122, arrow=False, color="#c92a2a", sw=1, dash="3 3")
    # wires
    s.line(150, ys + 23, 188, ys + 23)
    s.line(340, ys + 23, 378, ys + 23)
    s.line(520, ys + 23, 568, ys + 23)
    s.path(f"M520,{ys + 30} C545,{ys + 30} 545,263 568,263", color=LINE)
    s.line(710, ys + 23, 768, ys + 23, color=ORANGE, sw=1.8)
    s.path("M710,263 C745,263 745,262 768,262", color=LINE)
    # the variable is bound to Quantum yield and to the view
    s.path("M640,64 L640,148", color="#1f77b4", sw=1.6, dash="5 4")
    s.path("M780,64 C840,70 860,100 862,148", color="#1f77b4", sw=1.6, dash="5 4")
    # the drag
    s.path("M934,148 C934,90 900,50 812,46", color=ORANGE, sw=1.8)
    s.badge(950, 120, 1)
    s.badge(826, 20, 2)
    s.badge(640, 126, 3)
    s.badge(270, 140, 4)
    s.badge(950, 304, 5)
    s.line(20, 208, 520, 208, arrow=False, color="#b9b6ae", sw=1.4)
    s.line(20, 202, 20, 208, arrow=False, color="#b9b6ae", sw=1.4)
    s.line(520, 202, 520, 208, arrow=False, color="#b9b6ae", sw=1.4)
    s.text(270, 228, "same inputs, same parameters: nothing to do", 12.5, 700, "middle", MUTED)
    # the story
    s.line(24, 336, 976, 336, arrow=False, color="#d8d5cd", sw=1)
    story = ["Drag the λ line in a plot: the view changes its own parameter.",
             "That parameter is bound to the analysis variable λ, so every node bound to it gets the new value.",
             "Quantum yield now has different parameters, so a different fingerprint: it runs in a worker process.",
             "The others see the same inputs and parameters as before: their stored results are reused.",
             "The view keeps every mark that did not change and replaces only the one that moved."]
    for i, line in enumerate(story):
        y = 366 + i * 24
        s.badge(40, y - 4, i + 1)
        s.text(62, y, line, 13.5)
    s.save("drag-and-reuse.svg")


# --- 4. the project's files --------------------------------------------------------------------------------------
def files():
    s = Svg(1000, 420, "A project is a folder of your data plus a tesselab folder with analyses and graphs; "
                       "analyses of the same kind share one graph file and keep their own values.")
    tree = [(0, "my_project/", "any folder of yours", None),
            (1, "data/", "", None),
            (2, "experiment_1/", "your measurements,", "data"),
            (2, "experiment_2/", "only read, never changed", "data"),
            (1, "tesselab/", "added by Tesselab", "work"),
            (2, "project.toml", "", None),
            (2, "analyses/", "one small JSON file each", "work"),
            (3, "experiment_1.json", "", None),
            (3, "experiment_2.json", "", None),
            (2, "graphs/", "shared by analyses", "work"),
            (3, "quantum_yield.json", "", None),
            (2, "sessions/", "saved windows: tabs, layout", None),
            (2, "cache/", "results; safe to delete", None)]
    s.rect(24, 24, 470, 372, "plain", r=12)
    for i, (depth, name, note, mark) in enumerate(tree):
        y = 58 + i * 25
        x = 44 + depth * 24
        s.text(x, y, name, 13.5, 600 if name.endswith("/") else 400, mono=True)
        if note:
            s.text(272, y, note, 12.5, fill=MUTED)
    # the shared graph
    s.rect(640, 28, 320, 108, "plain", r=12, stroke=ACCENT, sw=2)
    s.text(660, 56, "Graph file", 15, 700)
    s.text(660, 78, "nodes and how they are wired", 13)
    s.text(660, 97, "where the boxes sit, default values", 13)
    s.text(660, 116, "variables and what they are bound to", 13)
    for x, name in ((530, "experiment_1"), (770, "experiment_2")):
        s.rect(x, 232, 210, 128, "plain", r=12)
        s.text(x + 16, 258, "Analysis", 15, 700)
        s.text(x + 90, 258, name, 13, fill=MUTED, mono=True)
        s.text(x + 16, 282, "its data folder", 13)
        s.text(x + 16, 302, "values set in this analysis", 13)
        s.text(x + 16, 322, "links to other analyses", 13)
        s.text(x + 16, 342, "results, key numbers", 13)
    s.line(635, 232, 700, 138)
    s.line(875, 232, 830, 138)
    s.text(655, 196, "uses", 12, 700, "middle", MUTED)
    s.text(868, 196, "uses", 12, 700, "middle", MUTED)
    s.text(530, 388, "Edit the graph once and every analysis that uses it follows.", 13, 600)
    s.save("project-files.svg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    big_picture()
    kinds()
    loop()
    files()
    print("written to", OUT)
