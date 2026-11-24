"""
The engine: computes analyses. It knows, for every node instance, everything its result depends on and turns that
into a fingerprint; a result is reused as long as its fingerprint stays the same.

The fingerprint of an instance covers: the node (module id, version, code hash, schema), its parameters (resolved:
analysis variables filled in, without the redraw-only ones), the files its path parameters point to (path, size,
modification time) and the content of the data its inputs receive (a hash of each upstream output). Because inputs
count by their content, not by the upstream node's state, a node whose input data stay the same is not run again
even when something upstream ran again ("early cutoff"): dragging λ reruns the quantum yield, but φ0(λ), which gets
the same data, is reused. So:
- changing a parameter reruns that node and whatever receives different data from it, nothing before it;
- a result is never shown for a state it was not computed for (the window asks for the current fingerprint);
- a link to another analysis is "live": the linked output's content is part of the fingerprint, and the analysis
  file remembers the content hash it last saw (Source.seen), so a changed source gives a warning until the analysis
  is saved again.

A fingerprint can only be computed once the node's inputs have results (their content is part of it); the runner
and compute() go through the nodes in dependency order. Results live in memory (all nodes) and on disk (nodes with
cache = "disk", see cache.py). Node runs happen in run_task (worker.py): here directly (compute), or in worker
processes (the window's runner, gui/runner.py, which uses prepare/finish).
"""
import collections
import glob
import hashlib
import json
import os
import time

from tesselab import cache, params as P
from tesselab.analysis import Analysis, AnalysisError, CycleError, Source, sources_of
from tesselab.graph import GraphStore
from tesselab.node import Bunch, DrawContext
from tesselab.project import natural_key
from tesselab.signature import bind, bindings
from tesselab.worker import run_task


class NodeResult:
    """The result of one node instance: outputs (dict of Datasets), diagnostics, seconds, error, fingerprint,
    key results {name: (value, error)}, status ("done", "error", "waiting": a run-on-button node not run yet,
    "blocked": an input is missing or failed), cached: None, "memory" or "disk"."""

    def __init__(self, fingerprint, outputs=None, diagnostics=(), seconds=0.0, error=None, status="done",
                 cached=None, load_seconds=0.0):
        self.fingerprint, self.outputs = fingerprint, outputs or {}
        self.diagnostics, self.seconds, self.error = list(diagnostics), seconds, error
        self.load_seconds = load_seconds  # importing the module in a fresh worker (not part of the computation)
        self.status, self.cached = status, cached
        self.key_results = {}
        self._hashes = {}
        self.holds_inputs = False  # a view's result: its outputs are its bound inputs (Engine.prepare)

    def output_hash(self, output) -> str | None:
        """A hash of an output's content (cached): what a node that takes this output depends on."""
        if output not in self.outputs:
            return None
        if output not in self._hashes:
            self._hashes[output] = content_hash(self.outputs[output])
        return self._hashes[output]

    @property
    def ok(self):
        return self.status == "done" and self.error is None

    def level(self) -> str | None:
        """The worst diagnostic: None, "info", "warning", "error"."""
        levels = [d["level"] for d in self.diagnostics] + (["error"] if self.error or self.status == "blocked" else [])
        for level in ("error", "warning", "info"):
            if level in levels:
                return level
        return None


class Key:
    """A node instance in an analysis (hashable: analyses by identity)."""
    __slots__ = ("analysis", "name")

    def __init__(self, analysis, name):
        self.analysis, self.name = analysis, name

    def __hash__(self):
        return hash((id(self.analysis), self.name))

    def __eq__(self, other):
        return isinstance(other, Key) and other.analysis is self.analysis and other.name == self.name

    def __repr__(self):
        return f"Key({self.analysis.label!r}, {self.name!r})"


def file_stamp(path):
    """What changes when a file or a folder's files change: path, size, modification time."""
    if not path:
        return None
    if os.path.isdir(path):
        stamps = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                full = os.path.join(root, name)
                try:
                    s = os.stat(full)
                    stamps.append((os.path.relpath(full, path), s.st_size, s.st_mtime_ns))
                except OSError:
                    pass
        return ["dir", path, stamps]
    try:
        s = os.stat(path)
        return [path, s.st_size, s.st_mtime_ns]
    except OSError:
        return [path, None]


def content_hash(dataset) -> str:
    """A hash of everything in a Dataset: names, dimensions, values, coordinates, attributes."""
    import numpy as np
    digest = hashlib.blake2b(digest_size=16)
    for name in sorted(map(str, dataset.variables)):
        variable = dataset.variables[name]
        digest.update(json.dumps([name, list(variable.dims), str(variable.dtype), list(variable.shape),
                                  variable.attrs], sort_keys=True, default=repr).encode())
        values = variable.values
        if values.dtype.kind == "O":
            digest.update(json.dumps(values.tolist(), default=repr).encode())
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(json.dumps(dataset.attrs, sort_keys=True, default=repr).encode())
    return digest.hexdigest()


class ResultMemory:
    """Results by fingerprint. Old results are kept (an undo finds them again), but not without end: beyond
    max_bytes of output data or max_count results, the least recently used are dropped (a result in use is looked up
    at every request, so it stays). A dropped result is computed again when it is needed."""

    def __init__(self, max_bytes=1024 ** 3, max_count=1000):
        self.max_bytes, self.max_count = max_bytes, max_count
        self._results: collections.OrderedDict[str, NodeResult] = collections.OrderedDict()
        self._sizes: dict[str, int] = {}
        self.nbytes = 0

    def __contains__(self, fingerprint):
        return fingerprint in self._results

    def __len__(self):
        return len(self._results)

    def __getitem__(self, fingerprint):
        self._results.move_to_end(fingerprint)
        return self._results[fingerprint]

    def __setitem__(self, fingerprint, result):
        self.pop(fingerprint)
        # a view's result holds its inputs, which belong to other results already: they do not count twice
        size = 0 if result.holds_inputs else sum(int(getattr(d, "nbytes", 0)) for d in result.outputs.values())
        self._results[fingerprint], self._sizes[fingerprint] = result, size
        self.nbytes += size
        while len(self._results) > 1 and (self.nbytes > self.max_bytes or len(self._results) > self.max_count):
            self.pop(next(iter(self._results)))

    def pop(self, fingerprint):
        if self._results.pop(fingerprint, None) is not None:
            self.nbytes -= self._sizes.pop(fingerprint)

    def clear(self):
        self._results.clear()
        self._sizes.clear()
        self.nbytes = 0


class Engine:
    def __init__(self, registry, project=None):
        self.registry, self.project = registry, project
        self.memory = ResultMemory()
        self.open: dict[str, Analysis] = {}      # analyses open in the window, by absolute path (links use them)
        self.graphs = GraphStore()               # their graph files, shared by the analyses that use the same one
        self._loaded: dict[str, tuple] = {}       # analyses read for links: path -> (mtime, Analysis)
        self._bound = collections.OrderedDict()   # bound() by data, signature and choice (the newest 64)
        self._stamps: dict[str, tuple] = {}       # stamp(): path -> (when it was read, its file stamp)

    # --- analyses --------------------------------------------------------------------------------------------
    def register(self, analysis):
        """An analysis open in the window: links from other analyses use this (live) object."""
        if analysis.path:
            self.open[os.path.abspath(analysis.path)] = analysis
        if analysis.graph is not None and analysis.graph.path:
            self.graphs.add(analysis.graph)

    def unregister(self, analysis):
        if analysis.path and self.open.get(os.path.abspath(analysis.path)) is analysis:
            del self.open[os.path.abspath(analysis.path)]
        graph = analysis.graph
        if graph is not None and not any(a.graph is graph for a in self.open.values()):
            self.graphs.drop(graph)  # opened again later: read from its file

    def sharing(self, analysis) -> list:
        """The other open analyses that use the same graph file."""
        if analysis.graph is None:
            return []
        return [a for a in self.open.values() if a is not analysis and a.graph is analysis.graph]

    def base_folder(self, analysis) -> str:
        """Relative paths in an analysis are relative to the project (or the analysis file's folder)."""
        if self.project:
            return self.project.root
        return os.path.dirname(analysis.path) if analysis.path else os.getcwd()

    def link_folder(self, analysis) -> str:
        """Links to other analyses are relative to the project's tesselab/ folder ("analyses/buffer.json")."""
        if self.project:
            return self.project.folder()
        return os.path.dirname(os.path.dirname(analysis.path)) if analysis.path else os.getcwd()

    def link_path(self, analysis, other_path) -> str:
        """How `analysis` writes a link to the analysis file at other_path."""
        return os.path.relpath(other_path, self.link_folder(analysis)).replace("\\", "/")

    def linked(self, analysis, source) -> Analysis:
        """The analysis a link points to (open in the window, or read from its file)."""
        path = os.path.normpath(os.path.join(self.link_folder(analysis), source.file))
        if path in self.open:
            return self.open[path]
        if not os.path.isfile(path):
            raise AnalysisError(f"The linked analysis {source.file} does not exist.")
        cached = self._loaded.get(path)
        if cached is None or cached[0] != _stamps(path, cached[1]):
            analysis = Analysis.load(path)  # its graph file read from disk too: a closed analysis is its files
            cached = (_stamps(path, analysis), analysis)
            self._loaded[path] = cached
        return cached[1]

    # --- nodes -----------------------------------------------------------------------------------------------
    def node_class(self, analysis, name):
        return self.registry.node_class(analysis.nodes[name]["node"])

    def node_info(self, analysis, name):
        return self.registry.node(analysis.nodes[name]["node"])

    def resolve_params(self, analysis, name, absolute=True) -> tuple[dict, list]:
        """The values run() gets: defaults, the saved values (migrated if saved with an older schema), analysis
        variables filled in, paths absolute. Returns (values, problems)."""
        cls = self.node_class(analysis, name)
        entry = analysis.nodes[name]
        saved = dict(entry.get("params", {}))
        problems = []
        schema = entry.get("schema", cls.schema)
        if schema < cls.schema:
            try:
                saved = cls.migrate(saved, schema)
            except Exception as error:
                problems.append(f"Parameters could not be migrated from schema {schema}: {error}")
        elif schema > cls.schema:
            problems.append(f"Saved with a newer version of the node (schema {schema}); shown read-only.")
        values = {}
        base = self.base_folder(analysis)
        for p in cls.params:
            value = saved.get(p.name, p.default_value())
            if isinstance(value, dict) and set(value) == {"var"}:
                if value["var"] not in analysis.vars:
                    problems.append(f"{p.name}: the variable {value['var']!r} does not exist.")
                    value = p.default_value()
                else:
                    value = analysis.vars[value["var"]]
            try:
                value = p.check(value)
            except ValueError as error:
                problems.append(str(error))
                value = p.default_value()
            if absolute and value is not None and isinstance(p, (P.Path, P.Paths)):
                value = self.absolute_paths(p, value, base)
            values[p.name] = value
        return values, problems

    @staticmethod
    def absolute_paths(p, value, base):
        if isinstance(p, P.Path):
            return os.path.normpath(os.path.join(base, value))
        if isinstance(value, str):  # a pattern
            return sorted(glob.glob(os.path.join(base, value)), key=lambda s: natural_key(os.path.basename(s)))
        return [os.path.normpath(os.path.join(base, v)) for v in value]

    def stamp(self, path):
        """file_stamp(), remembered for a second (forget_stamps clears it). A fingerprint needs the stamps of every
        file a node reads and each step of dragging λ asks for many fingerprints, a folder of power meter files
        among them: walking it again every time was a tenth of a step (2026-11-24)."""
        now = time.monotonic()
        hit = self._stamps.get(path)
        if hit is not None and now - hit[0] < 1.0:
            return hit[1]
        value = file_stamp(path)
        self._stamps[path] = (now, value)
        return value

    def forget_stamps(self):
        """The next fingerprints read the files again (something on disk may have changed)."""
        self._stamps.clear()

    def fingerprint(self, key, results) -> str:
        """The fingerprint of a node instance; results: the results of (at least) the instances it takes inputs
        from (their outputs' content is part of it). KeyError if one of them has no result yet."""
        analysis, name = key.analysis, key.name
        entry = analysis.nodes[name]
        info = self.registry.node(entry["node"])
        cls = self.registry.node_class(entry["node"])
        values, _ = self.resolve_params(analysis, name)
        redraw = {p.name for p in cls.params if p.redraw}
        files = {p.name: [self.stamp(v) for v in (values[p.name] if isinstance(values[p.name], list)
                                                  else [values[p.name]])]
                 for p in cls.params if isinstance(p, (P.Path, P.Paths)) and values.get(p.name)}
        upstream = {}
        for input_name, sources in analysis.inputs_of(name).items():
            parts = []
            for s in sources:
                if not self._resolvable(analysis, s):
                    parts.append(["missing", s.ref()])
                    continue
                dep = results[self.source_key(analysis, s)]
                content = dep.output_hash(s.output) if dep.ok else None
                parts.append([content or [dep.status, dep.fingerprint], s.output, s.bind])
            upstream[input_name] = parts
        data = {"node": entry["node"], "version": info.module.version, "code": info.module.code_hash,
                "schema": cls.schema, "params": {k: v for k, v in values.items() if k not in redraw},
                "files": files, "inputs": upstream}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=repr).encode()).hexdigest()[:32]

    def _resolvable(self, analysis, source):
        try:
            key = self.source_key(analysis, source)
            return key.name in key.analysis.nodes
        except (AnalysisError, OSError):
            return False

    def source_key(self, analysis, source) -> Key:
        if source.file:
            return Key(self.linked(analysis, source), source.instance)
        return Key(analysis, source.instance)

    def dependencies(self, key) -> list[Key]:
        found = []
        for sources in key.analysis.inputs_of(key.name).values():
            for source in sources:
                if self._resolvable(key.analysis, source):
                    dep = self.source_key(key.analysis, source)
                    if dep not in found:
                        found.append(dep)
        return found

    def needed(self, keys) -> list[Key]:
        """The keys and everything they depend on, in an order where dependencies come first."""
        result, visiting = [], set()

        def visit(key, path):
            if key in result:
                return
            if key in visiting:
                raise CycleError("The graph loops: " + " → ".join(f"{k.analysis.label}:{k.name}" for k in path + [key]))
            visiting.add(key)
            for dep in self.dependencies(key):
                visit(dep, path + [key])
            visiting.discard(key)
            result.append(key)

        for key in keys:
            visit(key, [])
        return result

    # --- running ---------------------------------------------------------------------------------------------
    def lookup(self, key, fingerprint) -> NodeResult | None:
        """A result for this fingerprint of `key`, if there is one (memory, then disk)."""
        if fingerprint in self.memory:
            return self.memory[fingerprint]
        cls = self.node_class(key.analysis, key.name)
        if cls.cache == "disk" and self.project:
            raw = cache.load(self.project.root, fingerprint)
            if raw is not None:
                return self._store(key, fingerprint, raw, cached="disk")
        return None

    def bound(self, signature, dataset, binding):
        """bind(), remembered. The same data (the very object: what a reused result gives, even after the node before
        it ran again) bound in the same way give the same bound dataset, so that a view and its plots can tell that an
        input has not changed. Binding took ~8 ms per input and a view has eight: dragging λ prepared two views, and
        that was the largest part of a step (2026-11-24). The bound datasets share the data of their sources."""
        key = (id(dataset), id(signature), repr(sorted((binding or {}).items())))
        hit = self._bound.get(key)
        if hit is not None and hit[0] is dataset and hit[1] is signature:  # (the objects are kept: ids stay unique)
            self._bound.move_to_end(key)
            return hit[2]
        value = bind(signature, dataset, binding)
        self._bound[key] = (dataset, signature, value)
        while len(self._bound) > 64:
            self._bound.popitem(last=False)
        return value

    def prepare(self, key, results, force=False, fingerprint=None):
        """For one key whose dependencies have results: ("done", NodeResult) if there is nothing to run, or
        ("run", task) with the task for run_task. A view is done here at once: its result holds its bound inputs
        (nothing is sent to a worker process), and an input whose source has no result is left out (the view
        draws what there is; the source's own status says what went wrong)."""
        fingerprint = fingerprint or self.fingerprint(key, results)
        existing = self.lookup(key, fingerprint)
        if existing is not None and not (force and existing.status == "waiting"):
            return "done", existing
        analysis, name = key.analysis, key.name
        cls = self.node_class(analysis, name)
        info = self.node_info(analysis, name)
        values, problems = self.resolve_params(analysis, name)
        diagnostics = [{"level": "warning", "message": p, "plots": [], "code": "params"} for p in problems]
        view = is_view(cls)
        inputs, blocked, missing = {}, [], []
        saved_inputs = analysis.inputs_of(name)
        for input_name, signature in cls.inputs.items():
            sources = saved_inputs.get(input_name, [])
            if not sources:
                if not signature.optional:
                    (missing if view else blocked).append(f"The input {input_name!r} is not connected.")
                continue
            datasets = []
            for source in sources:
                if not self._resolvable(analysis, source):
                    (missing if view else blocked).append(f"{input_name}: {source.ref()} does not exist.")
                    continue
                upstream = results.get(self.source_key(analysis, source))
                if upstream is None or not upstream.ok:
                    (missing if view else blocked).append(
                        f"{input_name}: {source.ref()} has no result"
                        + (f" ({upstream.error or upstream.status})." if upstream else "."))
                    continue
                if source.output not in upstream.outputs:
                    if not signature.optional:  # an optional input stays empty, e.g. from a reader without a file
                        blocked.append(f"{input_name}: {source.instance} has no output {source.output!r}.")
                    continue
                try:
                    datasets.append(self.bound(signature, upstream.outputs[source.output], source.bind))
                except ValueError as error:
                    blocked.append(f"{input_name}: {error}")
            if datasets:
                inputs[input_name] = datasets if signature.many else datasets[0]
        if view:
            raw = {"outputs": inputs, "diagnostics": diagnostics + [
                {"level": "warning", "message": m, "plots": [], "code": "input"} for m in missing]}
            return "done", self._store(key, fingerprint, raw)
        if blocked:
            result = NodeResult(fingerprint, diagnostics=diagnostics + [
                {"level": "error", "message": m, "plots": [], "code": "input"} for m in blocked], status="blocked")
            return "done", result  # not stored: it changes as soon as the inputs do
        if cls.run_on == "button" and not force:
            return "done", NodeResult(fingerprint, diagnostics=diagnostics, status="waiting")
        task = {"folder": info.module.folder, "module_id": info.module.id, "code_hash": info.module.code_hash,
                "class_spec": info.class_spec, "node": analysis.nodes[name]["node"], "params": values,
                "inputs": inputs, "fingerprint": fingerprint, "diagnostics": diagnostics}
        return "run", task

    def finish(self, key, task, raw) -> NodeResult:
        """Store what run_task returned for the task (not when the module changed on disk before it ran: then
        nothing ran, and the error is returned without being kept)."""
        if raw.get("stale"):
            return NodeResult(task["fingerprint"], error=raw["error"], status="error", diagnostics=[
                {"level": "error", "message": raw["error"], "plots": [], "code": "module-changed"}])
        raw = dict(raw)
        raw["diagnostics"] = task.get("diagnostics", []) + raw.get("diagnostics", [])
        result = self._store(key, task["fingerprint"], raw)
        cls = self.node_class(key.analysis, key.name)
        if cls.cache == "disk" and self.project and result.ok:
            cache.store(self.project.root, task["fingerprint"], raw,
                        hashes={name: result.output_hash(name) for name in result.outputs})
        return result

    def _store(self, key, fingerprint, raw, cached=None) -> NodeResult:
        result = NodeResult(fingerprint, raw.get("outputs"), raw.get("diagnostics", []), raw.get("seconds", 0.0),
                            raw.get("error"), "error" if raw.get("error") else "done", cached,
                            raw.get("load_seconds", 0.0))
        cls = self.registry.node_class(key.analysis.nodes[key.name]["node"])
        result.holds_inputs = is_view(cls)
        for r in cls.key_results:
            result.key_results[r.name] = r.read(result.outputs)
        result._hashes.update(raw.get("hashes", {}))  # from the disk cache: the hashes of the original outputs
        self.memory[fingerprint] = result
        return result

    def compute(self, analysis, names=None, force=(), runner=run_task) -> dict[str, NodeResult]:
        """Compute an analysis (or the named instances and what they need) here, one node after the other. force:
        instances to run even if they run only on a button. Returns {instance name: NodeResult} of this analysis."""
        results = self.compute_keys([Key(analysis, n) for n in (names or analysis.order())],
                                    {Key(analysis, n) for n in force}, runner)
        out = {key.name: result for key, result in results.items() if key.analysis is analysis}
        for name, result in out.items():
            result.diagnostics = [d for d in result.diagnostics if d.get("code") != "link-changed"] + \
                self.link_warnings(analysis, name, results)
        return out

    def compute_keys(self, keys, force=(), runner=run_task) -> dict:
        """{Key: NodeResult} of the keys and everything they need, computed here in dependency order."""
        self.forget_stamps()  # (a computation of its own looks at the files as they are now)
        results = {}
        for key in self.needed(keys):
            state, value = self.prepare(key, results, force=key in force)
            results[key] = value if state == "done" else self.finish(key, value, runner(value))
        return results

    def link_warnings(self, analysis, name, results) -> list[dict]:
        """A warning for every link of this node whose source's content changed since the analysis was saved.
        results: the current results, including the linked instances."""
        warnings = []
        for input_name, sources in analysis.inputs_of(name).items():
            for source in sources:
                if not source.file or not source.seen or not self._resolvable(analysis, source):
                    continue
                upstream = results.get(self.source_key(analysis, source))
                if upstream is None or not upstream.ok:
                    continue
                if upstream.output_hash(source.output) != source.seen:
                    warnings.append({"level": "warning", "plots": [], "code": "link-changed",
                                     "message": f"{source.file} has changed since this analysis was saved "
                                                f"({input_name}); key results may differ. Save to accept."})
        return warnings

    def remember_links(self, analysis):
        """Before saving: remember the current content hash of every linked result (Source.seen)."""
        for name, input_name, _ in analysis.links():
            value = analysis.nodes[name]["inputs"][input_name]
            updated = []
            for source in sources_of(value):
                if source.file and self._resolvable(analysis, source):
                    try:
                        key = self.source_key(analysis, source)
                        upstream = self.compute_keys([key])[key]
                        if upstream.ok:
                            source.seen = upstream.output_hash(source.output)
                    except (AnalysisError, KeyError, RuntimeError):
                        pass
                updated.append(source.to_json())
            analysis.nodes[name]["inputs"][input_name] = updated if isinstance(value, list) else updated[0]

    def store_results(self, analysis, results):
        """Key results into the analysis file (written on save; they never make an analysis unsaved)."""
        saved = {}
        for name, result in results.items():
            values = {}
            for key_name, (value, error) in result.key_results.items():
                if value is not None:
                    values[key_name] = value
                    if error is not None:
                        values[key_name + "_err"] = error
            if values:
                saved[name] = values
        analysis.doc["results"] = saved

    # --- drawing ---------------------------------------------------------------------------------------------
    def draw_context(self, analysis, name, result, others=()) -> DrawContext:
        """What a plot's draw function gets. A view's result holds its bound inputs: they are ctx.inputs."""
        values, _ = self.resolve_params(analysis, name)
        data = result.outputs if result else {}
        view = is_view(self.node_class(analysis, name))
        return DrawContext({} if view else data, data if view else {}, values, analysis.label, name, analysis.vars,
                           result.diagnostics if result else [], others)

    def missing_inputs(self, analysis, name, result) -> list[str]:
        """A view's needed (not optional) inputs that its result does not hold: their sources have no result (yet).
        Its plots then say so instead of being drawn: a draw function may count on its needed inputs."""
        cls = self.node_class(analysis, name)
        if not is_view(cls) or result is None:
            return []
        return [k for k, signature in cls.inputs.items() if not signature.optional and k not in result.outputs]

    def input_choices(self, analysis, name, input_name, results) -> list[tuple[Source, list]]:
        """For the window's input dropdown and the graph window: [(source, bindings), ...] of every output in this
        analysis that fits the input's signature and would make no cycle (links to other analyses are chosen in the
        LinkDialog)."""
        cls = self.node_class(analysis, name)
        signature = cls.inputs[input_name]
        downstream = set(analysis.downstream(name))
        choices = []
        for other_name in analysis.order():
            if other_name in downstream:
                continue  # would loop
            try:
                if is_view(self.node_class(analysis, other_name)):
                    continue  # a view's result holds its inputs, not outputs of its own
            except Exception:
                continue
            result = results.get(other_name)
            if result is None or not result.ok:
                continue
            for output, dataset in result.outputs.items():
                options = bindings(signature, dataset)
                if options:
                    choices.append((Source(other_name, output), options))
        return choices


def is_view(cls) -> bool:
    return getattr(cls, "kind", "compute") == "view"


def node_kind(cls) -> str:
    """"reader", "compute" or "view"."""
    kind = getattr(cls, "kind", "compute")
    return kind if kind in ("reader", "compute", "view") else "compute"


def _stamps(path, analysis):
    """When an analysis file and its graph file were last changed (a linked analysis is read again then)."""
    files = [path] + ([analysis.graph.path] if analysis is not None and analysis.graph is not None else [])
    return tuple(os.stat(f).st_mtime_ns if os.path.exists(f) else None for f in files)


def provenance(registry, analysis) -> dict:
    """What a saved result was made with."""
    import platform
    import time
    from tesselab import __version__
    nodes = {}
    for name, entry in analysis.nodes.items():
        try:
            info = registry.node(entry["node"])
            nodes[name] = f"{entry['node']} {info.module.version} {info.module.code_hash}"
        except KeyError:
            nodes[name] = f"{entry['node']} (module missing)"
    found = {"tesselab": __version__, "nodes": nodes, "python": platform.python_version(),
             "saved": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    template = analysis.doc.get("provenance", {}).get("template")
    if template:
        found["template"] = template  # the template it was made from (what comes from the data, graph.py)
    return found
