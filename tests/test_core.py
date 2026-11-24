"""The core without a window: parameters, signatures, plot descriptions, modules, analyses, the engine."""
import json
import os

import numpy as np
import pytest
import xarray as xr

from tesselab import P, Signature, Var, PlotSpec, Axis, Line, Lines, Panel, Heatmap
from tesselab import units
from tesselab.analysis import Analysis, CycleError, Source, instantiate
from tesselab.engine import Engine, Key
from tesselab.modules import Registry
from tesselab.signature import bind, bindings


# --- parameters and units ----------------------------------------------------------------------------------------
def test_units():
    assert units.to_display(260e-9, "m", "nm") == pytest.approx(260)
    assert units.from_display(260, "m", "nm") == pytest.approx(260e-9)
    assert units.to_display(0.002, "OD", "mOD") == pytest.approx(2)
    assert units.to_display(120, "s", "min") == pytest.approx(2)
    assert units.to_display(500e-9, "m", "cm-1") == pytest.approx(20000)
    assert units.to_display(500e-9, "m", "eV") == pytest.approx(2.4797, abs=1e-4)
    assert units.from_display(units.to_display(400e-9, "m", "eV"), "m", "eV") == pytest.approx(400e-9)
    with pytest.raises(ValueError):
        units.to_display(1, "m", "kg")


def test_param_text_round_trips():
    lam = P.Float("lambda_spectrum", 260e-9, unit="m", display="nm")
    assert lam.to_text(260e-9) == "260"
    assert lam.from_text("265.5") == pytest.approx(265.5e-9)
    assert P.Ints("ignore").from_text("0-3, 7") == [0, 1, 2, 3, 7]
    assert P.Ints("ignore").to_text([0, 1, 2, 3, 7]) == "0-3, 7"
    r = P.Range("baseline", unit="m", display="nm", optional=True)
    assert r.from_text("385-398") == pytest.approx([385e-9, 398e-9])
    assert r.from_text("") is None
    assert P.Range("delay", unit="s").from_text("-5 - -1") == [-5.0, -1.0]
    ranges = P.Ranges("ranges", unit="m", display="nm").from_text("385-398, 330-340")
    assert np.allclose(ranges, [[385e-9, 398e-9], [330e-9, 340e-9]], rtol=1e-12, atol=0)
    assert P.Paths("files").from_text("powers/*.csv") == "powers/*.csv"
    assert P.Paths("files").from_text("a.csv, b.csv") == ["a.csv", "b.csv"]
    with pytest.raises(ValueError):
        P.Int("n", 5, min=2).check(1)
    with pytest.raises(ValueError):
        P.Float("x", 1.0).check(float("nan"))
    with pytest.raises(ValueError):
        P.Choice("mode", options=["a", "b"]).check("c")


def test_steps_parameter():
    steps = P.Steps("processing", kinds=[P.Step("background", params=[P.Range("window", unit="s", optional=True)]),
                                         P.Step("chirp", params=[P.Points("onsets"), P.Int("order", 3)])])
    value = steps.check([{"kind": "chirp", "onsets": [[1, 2]]}, {"kind": "background", "on": False}])
    assert value == [{"kind": "chirp", "on": True, "onsets": [[1.0, 2.0]], "order": 3},
                     {"kind": "background", "on": False, "window": None}]
    assert P.find([steps], "processing.chirp.onsets").name == "onsets"
    with pytest.raises(ValueError):
        steps.check([{"kind": "unknown"}])


# --- signatures ----------------------------------------------------------------------------------------------------
def uv_like():
    return xr.Dataset({"absorbance": (("spectrum", "wavelength"), np.arange(12.0).reshape(3, 4))},
                      coords={"wavelength": ("wavelength", [1.0, 2, 3, 4], {"units": "m"}),
                              "total_absorbed_energy": ("spectrum", [0.0, 1, 2]),
                              "total_irradiation_time": ("spectrum", [0.0, 10, 20])})


def test_signature_binds_by_shape():
    signature = Signature(matrix=Var("n", "m"), row_pos=Var("n", optional=True), col_pos=Var("m", units="m"))
    options = bindings(signature, uv_like())
    assert all(o["matrix"] == "absorbance" and o["col_pos"] == "wavelength" for o in options)
    assert {o["row_pos"] for o in options} == {"total_absorbed_energy", "total_irradiation_time"}
    bound = bind(signature, uv_like(), {"row_pos": "total_irradiation_time"})
    assert bound.matrix.dims == ("n", "m")
    assert list(bound.row_pos.values) == [0, 10, 20]
    assert list(bound.col_pos.values) == [1, 2, 3, 4]
    assert bound.attrs["tesselab_dims"] == {"n": "spectrum", "m": "wavelength"}


def test_signature_transposes_and_rejects():
    data = uv_like().transpose("wavelength", "spectrum")
    bound = bind(Signature(matrix=Var("n", "m")), data, {"matrix": "absorbance"})
    assert bound.matrix.shape == (4, 3) or bound.matrix.dims == ("n", "m")
    with pytest.raises(ValueError):
        bind(Signature(cube=Var("a", "b", "c")), data)


# --- plot descriptions ---------------------------------------------------------------------------------------------
def test_plotspec_validates_and_exports():
    spec = PlotSpec(x=Axis("Wavelength", "m", display="nm"), y=Axis("Absorbance", "OD"),
                    marks=[Line([1, 2, 3], [4, 5, 6], name="a"), Lines([1, 2], [[1, 2], [3, 4]])])
    data = spec.to_dict()
    json.dumps(data)
    assert data["panels"]["main"]["marks"][0]["y"] == [4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        Line([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        Axis("x", "m", display="kg")
    with pytest.raises(ValueError):
        PlotSpec(panels={"a": Panel()}, mosaic=[["a", "b"]])
    big = Line(np.arange(10000.0), np.sin(np.arange(10000.0)))
    assert len(big.to_dict()["x"]) <= 4000
    heat = Heatmap([1, 2], [1, 2, 3], np.zeros((2, 3)))
    assert heat.color_limits() == (-1.0, 1.0)


# --- modules, analyses, engine ---------------------------------------------------------------------------------------
def test_builtin_modules_are_found():
    registry = Registry(user=False, entry_points=False)
    assert "org.tesselab.table" in registry.modules
    assert "org.tesselab.chemometrics/svd" in registry.nodes()
    assert registry.node_class("org.tesselab.chemometrics/svd").name == "SVD"


def table_and_svd(project):
    analysis = Analysis(path=project.new_analysis_path("mix"))
    analysis.doc["label"] = "mix"
    analysis.add("org.tesselab.table/csv", "table", params={"file": "data/mix.csv", "x_unit": "m"})
    analysis.add("org.tesselab.chemometrics/svd", "svd", inputs={"data": "table.table"})
    return analysis


def test_engine_computes_and_reuses(project):
    registry = Registry(project.root, user=False, entry_points=False)
    engine = Engine(registry, project)
    analysis = table_and_svd(project)
    results = engine.compute(analysis)
    assert results["table"].ok, results["table"].diagnostics
    assert results["svd"].ok, results["svd"].diagnostics
    svd = results["svd"].outputs["svd"]
    assert svd.singular_values.values[2] > 10 * svd.singular_values.values[3]  # three components
    assert results["svd"].key_results["residual"][0] < 1
    # nothing changed: the same results
    again = engine.compute(analysis)
    assert again["svd"] is results["svd"]
    # a parameter of svd changes: svd runs again, the table does not
    analysis.nodes["svd"]["params"]["components"] = 2
    third = engine.compute(analysis)
    assert third["table"] is results["table"] and third["svd"] is not results["svd"]
    # a file changes: everything runs again
    path = project.absolute("data/mix.csv")
    with open(path, "a") as f:
        f.write("401," + ",".join(["0"] * 20) + "\n")
    fourth = engine.compute(analysis)
    assert fourth["table"] is not results["table"]


def test_engine_reports_errors_without_crashing(project):
    engine = Engine(Registry(project.root, user=False, entry_points=False), project)
    analysis = table_and_svd(project)
    analysis.nodes["table"]["params"]["file"] = "data/missing.csv"
    results = engine.compute(analysis)
    assert results["table"].status == "error" and "missing" in results["table"].error
    assert results["svd"].status == "blocked"


def test_cycles_are_refused(project):
    analysis = table_and_svd(project)
    analysis.add("org.tesselab.chemometrics/svd", "svd2", inputs={"data": "svd.svd"})
    with pytest.raises(CycleError):
        analysis.set_input("svd", "data", "svd2.svd")
    assert analysis.nodes["svd"]["inputs"]["data"] == "table.table"  # the refused change was not kept


def test_analysis_file_round_trip(project):
    analysis = table_and_svd(project)
    analysis.vars["k"] = 3
    analysis.nodes["svd"]["params"]["components"] = {"var": "k"}
    analysis.save()
    loaded = Analysis.load(analysis.path)
    assert loaded.content() == analysis.content()
    engine = Engine(Registry(project.root, user=False, entry_points=False), project)
    values, problems = engine.resolve_params(loaded, "svd")
    assert values["components"] == 3 and not problems


def test_links_between_analyses_are_live_with_a_warning(project):
    registry = Registry(project.root, user=False, entry_points=False)
    engine = Engine(registry, project)
    source = table_and_svd(project)
    source.save()
    target = Analysis(path=project.new_analysis_path("target"))
    rel = engine.link_path(target, source.path)
    assert rel == "analyses/mix.json"
    target.add("org.tesselab.chemometrics/svd", "svd", inputs={"data": f"{rel}#table.table"})
    results = engine.compute(target)
    assert results["svd"].ok, results["svd"].diagnostics
    engine.remember_links(target)
    target.save()
    assert not [d for d in engine.compute(target)["svd"].diagnostics if d["code"] == "link-changed"]
    # the source analysis changes so that its output changes: the target warns until it is saved again
    source.nodes["table"]["params"]["y_unit"] = "OD"
    source.save()
    loaded = Analysis.load(target.path)
    assert [d for d in engine.compute(loaded)["svd"].diagnostics if d["code"] == "link-changed"]
    engine.remember_links(loaded)
    assert not [d for d in engine.compute(loaded)["svd"].diagnostics if d["code"] == "link-changed"]


def test_early_cutoff(project):
    """A node whose input data stay the same is not run again, even if the node before it ran again."""
    engine = Engine(Registry(project.root, user=False, entry_points=False), project)
    analysis = table_and_svd(project)
    first = engine.compute(analysis)
    path = project.absolute("data/mix.csv")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # the file is touched: the table runs again ...
    second = engine.compute(analysis)
    assert second["table"] is not first["table"]
    assert second["svd"] is first["svd"]  # ... but gives the same data, so SVD is reused


def test_template_instantiation():
    doc = instantiate({"label": "{name}", "nodes": {"a": {"node": "m/n", "params": {"f": "{file}", "x": "{x}"}}}},
                      {"name": "S1", "file": "data/a.csv", "x": [1, 2]})
    assert doc["label"] == "S1" and doc["nodes"]["a"]["params"] == {"f": "data/a.csv", "x": [1, 2]}
    assert doc["view"]["order"] == ["a"]


def test_source_parsing():
    s = Source.parse({"from": "analyses/b.json#mean.spectrum", "seen": "abc"})
    assert (s.file, s.instance, s.output, s.seen) == ("analyses/b.json", "mean", "spectrum", "abc")
    assert Source.parse("irr.spectra").to_json() == "irr.spectra"


def test_signature_placeholder_that_is_another_dimension():
    """SVD's output has the dims n, m and component; MCR binds its weights (n, component) as (n, m)."""
    data = xr.Dataset({"weights": (("n", "component"), np.ones((4, 2))), "residual": (("n", "m"), np.zeros((4, 3)))})
    bound = bind(Signature(matrix=Var("n", "m")), data, {"matrix": "weights"})
    assert bound.matrix.dims == ("n", "m") and bound.matrix.shape == (4, 2)
    assert bound.residual.dims == ("n", "m_")


# --- found by the audit of 2026-11-22 ---------------------------------------------------------------------------
def test_untrusted_project_modules_never_run(project, tmp_path):
    """A project's own modules run only once the user trusts the project, also from `tesselab export` and
    `tesselab modules`: importing the module here would create the marker file."""
    from tesselab import export
    from tesselab.project import set_trust
    folder = os.path.join(project.root, "tesselab", "modules", "evil")
    os.makedirs(folder)
    marker = tmp_path / "ran"
    with open(os.path.join(folder, "module.toml"), "w") as f:
        f.write('[module]\nid = "x.evil"\nname = "Evil"\nversion = "1"\napi = 1\n\n'
                '[[nodes]]\nid = "n"\nclass = "nodes.py:N"\n')
    with open(os.path.join(folder, "nodes.py"), "w") as f:
        f.write(f"open({str(marker)!r}, 'w').close()\nfrom tesselab import Node\nclass N(Node):\n"
                f"    def run(self, inputs, params):\n        return {{}}\n")
    analysis = Analysis(path=project.new_analysis_path("evil"))
    analysis.add("x.evil/n", "n")
    analysis.save()
    registry = Registry(project.root, user=False, entry_points=False)
    assert "x.evil" not in registry.modules and registry.untrusted == [folder]
    try:
        export.export_file(analysis.path, str(tmp_path / "out.pdf"))
    except Exception:
        pass  # the node's module is missing: an error is fine, running its code is not
    assert not marker.exists()
    set_trust(project.root, True)
    from tesselab.project import trust_state
    assert "x.evil" in Registry(project.root, trusted=trust_state(project.root) == "trusted",
                                user=False, entry_points=False).modules


def test_float_range_messages_with_one_open_end():
    with pytest.raises(ValueError, match=r"-1 is outside 0 … ∞"):
        P.Float("x", 1.0, min=0).check(-1)
    with pytest.raises(ValueError, match=r"3 is outside -∞ … 2"):
        P.Float("x", 1.0, max=2).check(3)
    with pytest.raises(ValueError, match=r"100 nm is outside 200 … 360 nm"):
        P.Float("wl", 2.6e-7, unit="m", display="nm", min=2e-7, max=3.6e-7).check(1e-7)


def test_result_memory_is_bounded():
    from tesselab.engine import NodeResult, ResultMemory
    memory = ResultMemory(max_bytes=1000, max_count=3)
    result = lambda n: NodeResult("", {"out": xr.Dataset({"a": ("i", np.zeros(n))})})
    for fp in "abc":
        memory[fp] = result(10)  # 80 bytes each
    memory["a"]  # used: kept
    memory["d"] = result(10)
    assert "b" not in memory and {"a", "c", "d"} <= {fp for fp in "abcd" if fp in memory}
    memory["e"] = result(100)  # 800 bytes: the oldest go until it fits
    assert memory.nbytes <= 1000 and "e" in memory and "c" not in memory
    memory["f"] = result(1000)  # alone bigger than the budget: still kept (it is the result in use)
    assert len(memory) == 1 and "f" in memory
    memory.clear()
    assert len(memory) == 0 and memory.nbytes == 0


def test_disk_cache_keeps_types_and_hashes(tmp_path):
    from tesselab import cache
    from tesselab.engine import content_hash
    ds = xr.Dataset({"a": (("t", "c"), np.arange(6, dtype=np.int64).reshape(3, 2) * 2 ** 40, {"n": 3})},
                    coords={"t": [0.1, 0.2, 0.3], "c": np.array(["x", "y"]), "label": ("t", np.array(["p", "q", "r"]))},
                    attrs={"k": np.int64(2), "flag": True})
    assert cache.store(str(tmp_path), "ab" * 16, {"outputs": {"m": ds}, "seconds": 1.0},
                       hashes={"m": content_hash(ds)})
    back = cache.load(str(tmp_path), "ab" * 16)
    m = back["outputs"]["m"]
    assert m["a"].dtype == np.int64 and m["c"].dtype.kind == "U" and m.identical(ds)
    assert back["hashes"]["m"] == content_hash(ds)
    objects = xr.Dataset({"o": ("i", np.array([{"a": 1}], dtype=object))})
    assert not cache.store(str(tmp_path), "cd" * 16, {"outputs": {"o": objects}})
    assert cache.load(str(tmp_path), "cd" * 16) is None


def test_saving_notices_changes_made_elsewhere(project):
    analysis = table_and_svd(project)
    assert not analysis.changed_on_disk()  # not saved yet
    analysis.save()
    assert not analysis.changed_on_disk()
    other = Analysis.load(analysis.path)
    other.doc["comment"] = "edited in another window"
    other.save()
    assert analysis.changed_on_disk()
    analysis.save()
    assert not analysis.changed_on_disk()


def test_write_json_leaves_no_temporary_file(tmp_path):
    from tesselab.analysis import write_json
    with pytest.raises(TypeError):
        write_json(str(tmp_path / "x.json"), {"bad": object()})
    assert os.listdir(tmp_path) == []


def test_graph_files_keep_the_sample_apart():
    """split() puts nodes, connections, bindings, positions and default values into the graph, and the sample's
    own values, links and view into the analysis file; merge() puts them together again."""
    from tesselab.graph import merge, own_from_template, split
    doc = {"label": "x", "vars": {"lam": 2.6e-7}, "view": {"order": ["a", "b"], "open": ["b"]}, "results": {},
           "nodes": {"a": {"node": "m/a", "inputs": {}, "pos": [0, 0],
                           "params": {"file": "d/1.csv", "k": 3, "w": {"var": "lam"}}},
                     "b": {"node": "m/b", "inputs": {"x": "a.out", "y": "analyses/o.json#a.out"}, "params": {"n": 1}}},
           "own": {"a": ["file"]}}
    graph, sample = split(doc)
    assert graph["nodes"]["a"]["params"] == {"k": 3, "w": {"var": "lam"}} and graph["nodes"]["a"]["pos"] == [0, 0]
    assert graph["nodes"]["b"]["inputs"] == {"x": "a.out", "y": None}  # a link is the sample's
    assert sample["params"] == {"a": {"file": "d/1.csv"}} and sample["inputs"] == {"b": {"y": "analyses/o.json#a.out"}}
    assert sample["view"] == {"open": ["b"]} and graph["order"] == ["a", "b"] and graph["vars"] == {"lam": 2.6e-7}
    whole = merge(graph, sample)
    assert whole["nodes"] == doc["nodes"] and whole["own"] == {"a": ["file"]}
    # a value changed without being marked counts as the sample's own: it is never pushed into the graph
    whole["nodes"]["b"]["params"]["n"] = 2
    graph2, sample2 = split(whole, graph)
    assert graph2["nodes"]["b"]["params"]["n"] == 1 and sample2["params"]["b"] == {"n": 2}
    # another sample on the same graph: its own values, the graph's structure and values
    other = merge(graph2, {"label": "y", "params": {"a": {"file": "d/2.csv"}}, "vars": {"lam": 3e-7}})
    assert other["nodes"]["a"]["params"]["file"] == "d/2.csv" and other["vars"] == {"lam": 3e-7}
    assert other["nodes"]["b"]["params"]["n"] == 1 and other["nodes"]["b"]["inputs"]["y"] is None
    assert own_from_template({"nodes": {"a": {"params": {"file": "{file}", "k": 3}}}}) == {"a": ["file"]}


def test_analyses_from_one_template_share_a_graph_file(project):
    from tesselab.graph import GraphStore
    from tesselab.modules import Registry
    from tesselab.templates import new_analysis
    import shutil
    shutil.copytree(project.absolute("data"), project.absolute("data2"))
    registry = Registry(project.root, user=False, entry_points=False)
    store = GraphStore()
    a = new_analysis(registry, project, "org.tesselab.table/table", project.absolute("data"), graphs=store)
    b = new_analysis(registry, project, "org.tesselab.table/table", project.absolute("data2"), graphs=store)
    assert a.graph is b.graph and a.is_own("table", "file")
    a.add("org.tesselab.chemometrics/svd", "svd", inputs={"data": "table.table"}, pos=(14, 0))
    assert a.push_graph()
    b.pull_graph()
    assert b.nodes["svd"]["inputs"]["data"] == "table.table" and b.nodes["table"]["params"]["file"] == "data2/mix.csv"
    a.save()
    b.save()
    with open(b.path) as f:
        saved = json.load(f)
    assert "nodes" not in saved and saved["params"] == {"table": {"file": "data2/mix.csv"}}
    loaded = Analysis.load(b.path)
    assert loaded.content() == b.content() and loaded.nodes["svd"]["pos"] == [14, 0]
    assert os.path.basename(loaded.graph.path) == "table.json"


# --- views and previews --------------------------------------------------------------------------------------------
def test_views_hold_their_inputs_and_draw_what_there_is(project):
    """A view is done in the main process: its result holds its bound inputs (no worker, no copy); a source without a
    result leaves the input out (a warning), never blocks it; a view is never offered as a source."""
    registry = Registry(project.root, user=False, entry_points=False)
    analysis = Analysis(path=project.new_analysis_path("v"))
    analysis.add("org.tesselab.table/csv", "table", params={"file": "data/mix.csv"})
    analysis.add("org.tesselab.chemometrics/svd", "svd", inputs={"data": "table.table"})
    analysis.add("org.tesselab.chemometrics/svd-view", "view", inputs={"svd": "svd.svd"})
    engine = Engine(registry, project)
    results = engine.compute(analysis)
    assert results["view"].ok and results["view"].holds_inputs
    assert results["view"].outputs["svd"].singular_values.values.tobytes() == \
        results["svd"].outputs["svd"].singular_values.values.tobytes()
    ctx = engine.draw_context(analysis, "view", results["view"])
    assert "svd" in ctx.inputs and not ctx.out
    choices = engine.input_choices(analysis, "svd", "data", results)
    assert [s.instance for s, _ in choices] == ["table"]  # not the view, whose result holds the SVD
    analysis.nodes["table"]["params"]["file"] = "data/nothing.csv"
    results = engine.compute(analysis)
    assert results["svd"].status == "blocked"
    assert results["view"].ok and "svd" not in results["view"].outputs
    assert any(d["level"] == "warning" for d in results["view"].diagnostics)


def test_previews_follow_the_shape():
    from tesselab import preview
    data = xr.Dataset({"spectrum": ("wavelength", np.ones(200)),
                       "few": (("nucleoside", "wavelength"), np.ones((4, 200))),
                       "map": (("delay", "wavelength"), np.ones((300, 200))),
                       "phi0": ((), 0.0123, {"units": ""})},
                      coords={"wavelength": ("wavelength", np.linspace(200e-9, 400e-9, 200), {"units": "m"}),
                              "nucleoside": list("ACGU")})
    assert preview.variables(data) == ["spectrum", "few", "map"]
    assert preview.scalars(data) == [("phi0", "0.0123")]
    line = preview.preview_spec(data, "spectrum").panels["main"]
    assert line.x.display == "nm" and line.marks[0].kind == "line"
    lines = preview.preview_spec(data, "few").panels["main"]
    assert lines.marks[0].kind == "lines" and len(lines.marks[0].y) == 4
    assert preview.preview_spec(data, "map").panels["main"].marks[0].kind == "heatmap"


def test_kinds_are_checked():
    """Plots belong to views; a view has no outputs."""
    import tesselab
    from tesselab.check import Report, _check_declarations

    class Bad(tesselab.Node):
        name = "Bad"
        outputs = ["x"]
        plots = [tesselab.Plot("p", "P", draw=lambda ctx: None)]

    class BadView(tesselab.View):
        name = "Bad view"
        outputs = ["x"]

    report = Report("x")
    _check_declarations(report, "m/bad", Bad)
    _check_declarations(report, "m/bad-view", BadView)
    errors = [m for level, m in report.entries if level == "error"]
    assert any("plots and interactions belong to views" in m for m in errors)
    assert any("a view has no outputs" in m for m in errors)


def test_mark_roles_come_from_the_theme():
    """A mark's role gives its look in the theme; what the module sets itself wins; a user's theme can change a role."""
    from tesselab import Style
    from tesselab.colors import THEMES, mark_style
    assert mark_style(THEMES["light"], Style(role="fit")) == {"color": "muted", "width": 2, "dash": "dash"}
    assert mark_style(THEMES["light"], Style(role="fit", width=3))["width"] == 3
    assert mark_style(THEMES["dark"], Style(role="reference"))["color"] != THEMES["light"]["marks"]["reference"]["color"]
    assert mark_style(THEMES["light"], Style(color="data")) == {"color": "data"}
    with pytest.raises(ValueError):
        Style(role="loud")


def test_a_task_made_for_older_module_code_does_not_run(tmp_path):
    """A module edited after a task was made for it (an editor or an LLM saving in quick steps): the worker does not
    run the task with the new code, the result is not kept under the old fingerprint, and once the registry has
    read the module again (refresh) the task for the new code runs."""
    from tesselab import worker
    from tesselab.scaffold import new_module
    folder = new_module(str(tmp_path / "curve"))
    registry = Registry(None, extra=[folder], user=False, builtin=False, entry_points=False)
    engine = Engine(registry)
    analysis = Analysis()
    analysis.add("my.curve/read", "read", params={"file": os.path.join(folder, "sample", "example", "curve.csv")})
    key = Key(analysis, "read")
    state, task = engine.prepare(key, {})
    assert state == "run"
    with open(os.path.join(folder, "analysis.py"), "a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    raw = worker.run_task(task)
    assert raw.get("stale")
    assert engine.finish(key, task, raw).status == "error" and task["fingerprint"] not in engine.memory
    assert registry.refresh() == ["my.curve"]
    state, again = engine.prepare(key, {})
    assert again["fingerprint"] != task["fingerprint"]
    result = engine.finish(key, again, worker.run_task(again))
    assert result.ok and again["fingerprint"] in engine.memory
    assert registry.refresh() == []  # nothing changed since


def test_cli_help_and_version(capsys):
    """--help and --version print and return; an unknown option is an error, not a path that opens the window."""
    from tesselab import __version__
    from tesselab.cli import main
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert main(["--help"]) == 0
    assert "tesselab check" in capsys.readouterr().out
    assert main(["--no-such-option"]) == 2
    assert "unknown option" in capsys.readouterr().err
