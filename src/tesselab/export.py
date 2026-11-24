"""
The export of an analysis: a PDF of every plot and, next to it, <name>_data.json with everything the plots show.

The JSON is written from the plot descriptions themselves (marks with their data, axes with units, panels, texts,
the (i) explanations), plus the analysis's parameters, variables, edges, key results, diagnostics and provenance; the
PDF is drawn from the same descriptions with matplotlib, so it shows exactly what the JSON holds. It is the way for
people, scripts and LLMs to look at results without running the analysis. Plots that are own widgets (level 2) are
pictures in the PDF and only named in the JSON.

    {"tesselab_export": 1, "label": ..., "comment": ..., "provenance": {...}, "vars": {...},
     "nodes": [{"instance": "qy", "node": "<module id>/<node id>", "title": ..., "params": {...}, "inputs": {...},
                "status": "done", "seconds": 0.012, "key_results": {"phi0_energy": [value, error]},
                "diagnostics": [...],
                "plots": [{"number": 1, "id": ..., "title": ..., "help": ..., "level": 1, "spec": {...}}]}]}
"""
import json
import os

from tesselab.analysis import Analysis, write_json
from tesselab.engine import Engine, provenance
from tesselab.plotspec import note_only


def collect(engine, analysis, results=None):
    """The export as a dict, and the plot descriptions [(number, title, spec or None, help)] for the PDF."""
    results = results if results is not None else engine.compute(analysis)
    data = {"tesselab_export": 1, "label": analysis.label, "comment": analysis.doc.get("comment", ""),
            "provenance": provenance(engine.registry, analysis), "vars": analysis.vars, "nodes": []}
    figures, number = [], 1
    for name in analysis.doc["view"]["order"]:
        if name not in analysis.nodes:
            continue
        entry = analysis.nodes[name]
        result = results.get(name)
        try:
            cls = engine.node_class(analysis, name)
        except Exception as error:
            data["nodes"].append({"instance": name, "node": entry["node"], "status": "error", "error": str(error)})
            continue
        values, _ = engine.resolve_params(analysis, name, absolute=False)
        node = {"instance": name, "node": entry["node"], "title": entry.get("label") or cls.name, "params": values,
                "inputs": entry.get("inputs", {}), "status": result.status if result else "missing",
                "seconds": result.seconds if result else None,
                "key_results": {k: list(v) for k, v in (result.key_results.items() if result else [])},
                "diagnostics": [{k: v for k, v in d.items() if k != "details"} for d in (result.diagnostics if result
                                                                                        else [])],
                "plots": []}
        for plot in cls.plots:
            info = {"number": number, "id": plot.id, "title": plot.title, "help": plot.help, "level": plot.level}
            spec = None
            if plot.level == 1:
                waiting = engine.missing_inputs(analysis, name, result) if result is not None else []
                if result is None or not result.ok:
                    spec = note_only("No result: " + ((result.error or result.status) if result else "not computed"))
                elif waiting:
                    spec = note_only("No " + ", ".join(waiting))
                else:
                    try:
                        spec = plot.draw(engine.draw_context(analysis, name, result))
                    except Exception as error:
                        spec = note_only(f"The plot failed: {type(error).__name__}: {error}")
                info["spec"] = spec.to_dict()
            node["plots"].append(info)
            figures.append((number, f"{node['title']}: {plot.title}", spec, plot.help, (name, plot.id)))
            number += 1
        data["nodes"].append(node)
    return data, figures


def export_analysis(engine, analysis, pdf_path, results=None, images=None):
    """Write the PDF and <name>_data.json next to it. images: {(instance, plot id): QImage} of level 2 plots."""
    data, figures = collect(engine, analysis, results)
    pdf_path = os.path.abspath(pdf_path)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    json_path = os.path.splitext(pdf_path)[0] + "_data.json"
    write_json(json_path, data)
    write_pdf(pdf_path, data, figures, images or {})
    return pdf_path, json_path


def write_pdf(path, data, figures, images):
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from tesselab.mpl import draw_spec

    with PdfPages(path) as pdf:
        pages = [figures[i:i + 4] for i in range(0, len(figures), 4)] or [[]]
        for page_number, page in enumerate(pages):
            fig = Figure(figsize=(8.27, 11.69))  # A4
            top = 0.97
            if page_number == 0:
                lines = [data["label"]]
                header = fig.text(0.06, top, data["label"], fontsize=14, fontweight="bold", va="top")
                y = top - 0.03
                if data.get("comment"):
                    fig.text(0.06, y, f"“{data['comment']}”", fontsize=9, style="italic", va="top")
                    y -= 0.022
                for node in data["nodes"]:
                    parts = [f"{k} = {v[0]:.4g}" + (f" ± {v[1]:.2g}" if v[1] is not None else "")
                             for k, v in node.get("key_results", {}).items() if v and v[0] is not None]
                    params = ", ".join(f"{k}={_short(v)}" for k, v in node.get("params", {}).items())
                    text = f"{node.get('title', node['instance'])} ({node['instance']}): " + \
                           ("; ".join(parts) + ". " if parts else "") + params
                    fig.text(0.06, y, text[:180], fontsize=6.5, va="top", color="#52514e")
                    y -= 0.016
                    for d in node.get("diagnostics", [])[:3]:
                        fig.text(0.08, y, f"{d['level']}: {d['message']}"[:170], fontsize=6.5, va="top",
                                 color="#c62828" if d["level"] == "error" else "#a06800")
                        y -= 0.016
                p = data["provenance"]
                fig.text(0.06, y, f"Tesselab {p['tesselab']}, saved {p['saved']}; nodes: " +
                         "; ".join(p["nodes"].values())[:200], fontsize=5.5, va="top", color="#898781")
                top = y - 0.03
            grid = GridSpec(2, 2, figure=fig, left=0.08, right=0.95, bottom=0.05, top=top, wspace=0.3, hspace=0.45)
            for i, (number, title, spec, help_text, key) in enumerate(page):
                cell = grid[i // 2, i % 2]
                if spec is not None:
                    axes = draw_spec(fig, cell, spec)
                    first = next(iter(axes.values()))
                    first.set_title(f"{number}. {title}", fontsize=8, loc="left")
                else:
                    ax = fig.add_subplot(cell)
                    ax.set_axis_off()
                    ax.set_title(f"{number}. {title}", fontsize=8, loc="left")
                    image = images.get(key)
                    if image is not None:
                        ax.imshow(_qimage_array(image))
                    else:
                        ax.text(0.5, 0.5, "(an own widget: shown in the window only)", ha="center", fontsize=7)
            pdf.savefig(fig)


def _short(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) < 40 else text[:37] + "…"


def _qimage_array(image):
    import numpy as np
    image = image.convertToFormat(image.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    buffer = image.constBits()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, image.bytesPerLine() // 4, 4)[:, :width]


def export_file(analysis_path, pdf_path=None, extra_modules=()):
    """tesselab export: compute an analysis file here and export it."""
    import sys
    from tesselab.modules import Registry
    from tesselab.project import Project, find_root, trust_state
    root = find_root(analysis_path)
    project = Project(root) if root else None
    registry = Registry(root, extra=extra_modules, trusted=bool(root) and trust_state(root) == "trusted")
    if registry.untrusted:
        print(f"Not using the project's own modules ({', '.join(os.path.basename(f) for f in registry.untrusted)}): "
              f"the project is not trusted. Open it in the window once to decide.", file=sys.stderr)
    engine = Engine(registry, project)
    analysis = Analysis.load(analysis_path)
    if pdf_path is None:
        base = project.root if project else os.path.dirname(os.path.abspath(analysis_path))
        pdf_path = os.path.join(base, "exports", f"{analysis.label}.pdf")
    export_analysis(engine, analysis, pdf_path)
    return pdf_path
