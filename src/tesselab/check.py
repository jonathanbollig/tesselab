"""
tesselab check <module folder>: whether a module works, on its own sample data, without a window. It either passes
or says exactly what is wrong; the plots are rendered to <module>/check/ so that whoever builds the module (a person
or an LLM) can look at them.

1. Manifest: required fields, api supported, requires installed, every template's recognize matches the sample.
2. Import: the node classes import in a fresh process within 2 s, without Qt.
3. Parameters: defaults valid, an analysis file round-trips, migrations run.
4. Templates: each builds an analysis on the sample (a temporary copy), and every node in it runs.
5. Nodes: outputs as declared, picklable, identical on a second run; timing; detect() under 50 ms.
6. Plots: every plot draws; each is rendered to check/<number>_<instance>_<plot>.png at scale 1 and 1.5.
7. Interactions: every cursor, range and pick names a parameter of the right type and existing plots.
8. Export: the JSON and the PDF are written (check/export.pdf).
9. The module's own tests (tests/), if any.
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from tesselab import params as P
from tesselab.node import KINDS, Crosshair, Cursor, Pick, RangeSelect, Toggle


class Report:
    def __init__(self, folder):
        self.folder, self.entries = folder, []

    def add(self, level, message):
        self.entries.append((level, message))

    def ok(self, message):
        self.add("ok", message)

    def warn(self, message):
        self.add("warning", message)

    def error(self, message):
        self.add("error", message)

    @property
    def passed(self):
        return not any(level == "error" for level, _ in self.entries)

    def text(self):
        marks = {"ok": "✓", "warning": "!", "error": "✗", "info": " "}
        lines = [f"{marks[level]} {message}" for level, message in self.entries]
        errors = sum(level == "error" for level, _ in self.entries)
        warnings = sum(level == "warning" for level, _ in self.entries)
        lines.append("")
        lines.append(("PASSED" if self.passed else "FAILED") + f": {errors} errors, {warnings} warnings "
                     f"(pictures and export in {os.path.join(self.folder, 'check')})")
        return "\n".join(lines)


def _datasets_equal(a, b):
    if set(a) != set(b):
        return False
    return all(a[k].identical(b[k]) for k in a)


def check_module(folder, pictures=True) -> Report:
    folder = os.path.abspath(folder)
    report = Report(folder)
    from tesselab.modules import ManifestError, Registry, read_manifest
    # 1. manifest
    try:
        info = read_manifest(folder, "check")
    except (ManifestError, OSError) as error:
        report.error(f"module.toml: {error}")
        return report
    report.ok(f"manifest: {info.id} {info.version}, {len(info.nodes)} nodes, {len(info.templates)} templates")
    for problem in info.problems:
        report.error(f"manifest: {problem}")
    if not info.nodes:
        report.error("manifest: no [[nodes]]")
    if not info.available:
        return report
    out = os.path.join(folder, "check")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    # 2. import, in a fresh process (time, no Qt)
    code = (f"import sys, time; sys.path[:0] = {json.dumps(sys.path)}; start = time.perf_counter();\n"
            f"from tesselab.modules import load_node_class\n"
            + "".join(f"load_node_class({folder!r}, {info.id!r}, {n.class_spec!r})\n" for n in info.nodes.values())
            + "print(round(time.perf_counter() - start, 3), 'PySide6' in sys.modules)")
    try:
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        if run.returncode != 0:
            report.error("import: the node classes do not import:\n" + run.stderr.strip()[-2000:])
            return report
        seconds, qt = run.stdout.split()[-2:]
        (report.ok if float(seconds) < 2 else report.warn)(f"import: {seconds} s"
                                                           + ("" if float(seconds) < 2 else " (more than 2 s)"))
        if qt == "True":
            report.warn("import: importing the nodes imports Qt; the analysis code must not (it runs in worker "
                        "processes and notebooks)")
    except subprocess.TimeoutExpired:
        report.error("import: took more than 60 s")
        return report
    registry = Registry(None, extra=[folder], user=False, builtin=True, entry_points=False)
    if info.id not in registry.modules or registry.modules[info.id].folder != folder:
        registry.modules[info.id] = info  # this folder wins over an installed module with the same id
    classes = {}
    for node in info.nodes.values():
        try:
            classes[node.full_id] = registry.node_class(node.full_id)
        except Exception as error:
            report.error(f"node {node.id}: {error}")
    # 3. parameters, 7. interactions (they need no data)
    for full_id, cls in classes.items():
        _check_declarations(report, full_id, cls)
        for view in getattr(cls, "views", []):
            view_id = view if "/" in view else f"{info.id}/{view}"
            target = classes.get(view_id)
            if target is None and view_id not in registry.nodes():
                report.error(f"node {full_id.rpartition('/')[2]}: its default view {view!r} is no node")
            elif target is not None and getattr(target, "kind", "compute") != "view":
                report.error(f"node {full_id.rpartition('/')[2]}: its default view {view!r} is no view")
    # 4.-6., 8. templates on the sample
    sample = os.path.join(folder, "sample")
    if not os.path.isdir(sample):
        report.warn("no sample/ folder: nothing could be run on data (add a small real dataset)")
        return _module_tests(report, folder)
    if not info.templates:
        report.warn("no template: the nodes were not run on the sample (add a template that builds an analysis)")
    temporary = tempfile.mkdtemp(prefix="tesselab-check-")
    try:
        from tesselab.project import Project, find_data, recognizes
        root = os.path.join(temporary, "project")
        shutil.copytree(sample, os.path.join(root, "sample"))
        project = Project.create(root)
        number = 1
        for template in info.templates.values():
            targets = []
            if template.recognize:
                targets = [f for f, matching in find_data(
                    root, [template], detect=lambda t, f: registry.function(t.module, t.detect)(f))]
                if not targets:
                    report.error(f"template {template.id}: its recognize {template.recognize} matches nothing in sample/")
                    continue
                report.ok(f"template {template.id}: recognizes {', '.join(os.path.relpath(t, root) for t in targets)}")
            else:
                targets = [os.path.join(root, "sample")]
            if template.detect:
                detect = registry.function(info, template.detect)
                start = time.perf_counter()
                found = detect(targets[0])
                seconds = time.perf_counter() - start
                (report.ok if seconds < 0.05 else report.warn)(
                    f"template {template.id}: detect() {'accepts' if found else 'rejects'} the sample in "
                    f"{seconds * 1000:.0f} ms" + ("" if seconds < 0.05 else " (more than 50 ms: the explorer runs it "
                                                                          "for every folder)"))
            number = _check_template(report, registry, project, template, targets[0], out, number, pictures)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return _module_tests(report, folder)


def _check_declarations(report, full_id, cls):
    name = full_id.rpartition("/")[2]
    if not cls.name:
        report.warn(f"node {name}: no name")
    for p in cls.params:
        try:
            p.check(p.default)
        except ValueError as error:
            report.error(f"node {name}: the default of {p.name} is not valid: {error}")
    kind = getattr(cls, "kind", "compute")
    if kind not in KINDS:
        report.error(f"node {name}: kind {kind!r} is not one of {', '.join(KINDS)}")
    if kind != "view" and (cls.plots or cls.interactions):
        report.error(f"node {name}: plots and interactions belong to views (tesselab.View): a {kind} node "
                     f"computes, a view takes its outputs and draws them")
    if kind == "view" and cls.output_names():
        report.error(f"node {name}: a view has no outputs (it only draws its inputs)")
    if kind == "view" and cls.key_results:
        report.error(f"node {name}: key results belong to the compute node whose outputs hold them")
    if kind == "view" and not cls.plots:
        report.warn(f"node {name}: a view without plots shows nothing")
    if kind == "reader" and not any(isinstance(p, (P.Path, P.Paths)) for p in cls.params):
        report.warn(f"node {name}: a reader without a path parameter reads no file")
    plot_ids = {plot.id for plot in cls.plots}
    if len(plot_ids) != len(cls.plots):
        report.error(f"node {name}: two plots have the same id")
    for plot in cls.plots:
        for setting in plot.settings:
            if P.find(cls.params, setting) is None:
                report.error(f"node {name}: plot {plot.id} shows the setting {setting!r}, which is no parameter")
        if plot.level == 2:
            report.warn(f"node {name}: plot {plot.id} is an own widget (level 2): only as stable as the libraries "
                        f"it uses, and only a picture in the export")
    expected = {Cursor: (P.Float,), Crosshair: (P.Float,), RangeSelect: (P.Range, P.Ranges), Pick: (P.Points,),
                Toggle: (P.Ints,)}
    for interaction in cls.interactions:
        for param in interaction.params:
            p = P.find(cls.params, param)
            types = expected[type(interaction)] if param == interaction.params[0] or isinstance(interaction, Crosshair) \
                else (P.Int,)
            if p is None:
                report.error(f"node {name}: the {interaction.kind} names {param!r}, which is no parameter")
            elif not isinstance(p, types):
                report.error(f"node {name}: the {interaction.kind} on {param!r} needs a "
                             f"{' or '.join(t.__name__ for t in types)} parameter, not {type(p).__name__}")
        for plot in interaction.plots:
            if plot not in plot_ids:
                report.error(f"node {name}: the {interaction.kind} on {interaction.params[0]!r} names the plot "
                             f"{plot!r}, which this node does not have")
    report.ok(f"node {name}: {len(cls.params)} parameters, {len(cls.plots)} plots, {len(cls.interactions)} "
              f"interactions declared correctly" if not any(level == "error" and name in m for level, m in
                                                           report.entries) else f"node {name}: declarations checked")


def _check_template(report, registry, project, template, target, out, number, pictures):
    from tesselab.analysis import Analysis
    from tesselab.engine import Engine
    from tesselab.export import export_analysis
    from tesselab.templates import new_analysis
    try:
        analysis = new_analysis(registry, project, template.full_id, target)
    except Exception as error:
        report.error(f"template {template.id}: building the analysis failed: {error}\n{traceback.format_exc()}")
        return number
    # round trip of the analysis file
    analysis.save()
    loaded = Analysis.load(analysis.path)
    if loaded.content() != analysis.content():
        report.error(f"template {template.id}: the analysis file does not read back the same")
    else:
        report.ok(f"template {template.id}: the analysis file round-trips")
    force = [n for n, e in analysis.nodes.items() if registry.node_class(e["node"]).run_on == "button"]
    first = Engine(registry, project).compute(analysis, force=force)
    second = Engine(registry, project).compute(analysis, force=force)
    for name in analysis.order():
        result, again = first[name], second[name]
        cls = registry.node_class(analysis.nodes[name]["node"])
        label = f"node {name} ({cls.name})"
        if not result.ok:
            details = "\n".join(d.get("details", d["message"]) for d in result.diagnostics if d["level"] == "error")
            report.error(f"{label}: {result.error or result.status}\n{details}".strip())
            continue
        missing = [o for o in cls.output_names() if o not in result.outputs]
        paths = [p for p in cls.params if isinstance(p, (P.Path, P.Paths))]
        values = analysis.nodes[name].get("params", {})
        no_files = getattr(cls, "kind", "") == "reader" and paths and not any(values.get(p.name) for p in paths)
        if missing and no_files:  # an optional file the sample does not have (a buffer, the rig's records)
            report.ok(f"{label}: no file given on the sample, so no {', '.join(missing)} (by design)")
        elif missing:
            report.warn(f"{label}: declared outputs not returned: {', '.join(missing)}")
        if isinstance(cls.outputs, dict):
            from tesselab.signature import Signature, fits
            for output, signature in cls.outputs.items():
                if isinstance(signature, Signature) and output in result.outputs and not fits(signature,
                                                                                               result.outputs[output]):
                    report.error(f"{label}: output {output!r} does not fit its declared signature")
        try:
            pickle.dumps(result.outputs)
        except Exception as error:
            report.error(f"{label}: the outputs cannot be pickled (they must be xarray/numpy): {error}")
        if again.ok and not _datasets_equal(result.outputs, again.outputs):
            report.error(f"{label}: a second run gives other results (use a Seed parameter for random numbers)")
        for d in result.diagnostics:
            if d["level"] == "warning":
                report.warn(f"{label}: reports a warning on the sample: {d['message']}")
        for p in cls.params:
            if isinstance(p, P.Seed) and analysis.nodes[name]["params"].get(p.name, p.default) is None:
                report.warn(f"{label}: {p.name} is None: the results are not reproducible")
        report.ok(f"{label}: runs in {result.seconds * 1000:.0f} ms, outputs {', '.join(result.outputs)}")
    number = _check_plots(report, registry, analysis, first, out, number, pictures)
    try:
        engine = Engine(registry, project)
        pdf, js = export_analysis(engine, analysis, os.path.join(out, f"export_{template.id}.pdf"), results=first)
        with open(js) as f:
            json.load(f)
        report.ok(f"export: {os.path.basename(pdf)} and {os.path.basename(js)} written")
    except Exception as error:
        report.error(f"export failed: {error}\n{traceback.format_exc()}")
    return number


def _check_plots(report, registry, analysis, results, out, number, pictures):
    from tesselab.engine import Engine
    from tesselab.plotspec import PlotSpec
    engine = Engine(registry, None)
    app = None
    if pictures:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6 import QtWidgets
            from tesselab.gui import theme
            from tesselab.gui.plots import render_image
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            theme.apply(app, "light")
        except Exception as error:
            report.warn(f"plots: no pictures ({error})")
            pictures = False
    for name in analysis.doc["view"]["order"]:
        cls = registry.node_class(analysis.nodes[name]["node"])
        result = results.get(name)
        for plot in cls.plots:
            tag = f"plot {number} ({name}/{plot.id})"
            number += 1
            if result is None or not result.ok:
                continue
            waiting = engine.missing_inputs(analysis, name, result)
            if waiting:
                report.warn(f"{tag}: not drawn, the view has no {', '.join(waiting)} (their source failed)")
                continue
            ctx = engine.draw_context(analysis, name, result)
            if plot.level == 2:
                if pictures:
                    try:
                        widget = plot.widget()
                        widget.resize(640, 360)
                        widget.update(ctx)
                        widget.grab().save(os.path.join(out, f"{number - 1:02d}_{name}_{plot.id}.png"))
                        report.ok(f"{tag}: own widget drawn")
                    except Exception as error:
                        report.error(f"{tag}: the widget failed: {error}\n{traceback.format_exc()}")
                continue
            try:
                spec = plot.draw(ctx)
            except Exception as error:
                report.error(f"{tag}: draw() failed: {type(error).__name__}: {error}\n{traceback.format_exc()}")
                continue
            if not isinstance(spec, PlotSpec):
                report.error(f"{tag}: draw() must return a PlotSpec, not {type(spec).__name__}")
                continue
            empty = [n for n, panel in spec.panels.items() if not panel.marks and not panel.note]
            if empty:
                report.warn(f"{tag}: panel(s) {', '.join(empty)} have no marks and no note")
            if pictures:
                try:
                    height = 320 * plot.rows
                    width = 640 * plot.span
                    for ratio, suffix in ((1.0, ""), (1.5, "@1.5")):
                        render_image(spec, width, height, ratio).save(
                            os.path.join(out, f"{number - 1:02d}_{name}_{plot.id}{suffix}.png"))
                except Exception as error:
                    report.error(f"{tag}: drawing failed: {error}\n{traceback.format_exc()}")
                    continue
            report.ok(f"{tag}: {plot.title!r} drawn")
    return number


def _module_tests(report, folder):
    tests = os.path.join(folder, "tests")
    if not os.path.isdir(tests):
        return report
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([p for p in sys.path if p] + [folder]))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", tests], capture_output=True, text=True, cwd=folder,
                         env=env, timeout=600)
    last = (run.stdout.strip().splitlines() or [""])[-1]
    if run.returncode == 0:
        report.ok(f"the module's own tests: {last}")
    else:
        report.error(f"the module's own tests fail: {last}\n{run.stdout[-3000:]}")
    return report
