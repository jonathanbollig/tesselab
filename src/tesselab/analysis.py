"""
Analyses: a graph of node instances, saved as one JSON file in <project>/tesselab/analyses/.

    {"tesselab": 1, "label": "5_UC@2", "comment": "", "vars": {"lambda": 2.6e-7},
     "nodes": {"qy": {"node": "<module id>/<node id>", "version": "1.2.0", "schema": 3,
                      "inputs": {"spectra": "irr.spectra"},
                      "params": {"lambda_spectrum": {"var": "lambda"}, "fit_num": 5}}},
     "results": {...}, "view": {"order": [...], "open": [...]}, "provenance": {...}}

An input is a source: "<instance>.<output>"; or {"from": source, "bind": {signature variable: data variable},
"seen": fingerprint}; a link into another analysis of the project is "analyses/<file>.json#<instance>.<output>"
(its "seen" is the fingerprint of that result when this file was last saved, see engine.py); an input with
many=True takes a list of sources. A parameter is a value or {"var": name}, an analysis variable. A node may have a
"pos": [x, y], its place in the graph window (grid units).
The graph has no cycles, also not through links to other analyses. Fields Tesselab does not know are kept.

The graph can live in a file of its own, shared by many analyses (graph.py): the analysis file then names it
("graph": "../graphs/x.json") and keeps only its sample. In memory such an analysis is whole as well (doc as above,
plus doc["own"]: the parameters it sets itself); Analysis.graph is the shared Graph, Analysis.base the graph the
document was last merged with. push_graph() hands an edit's graph part to the Graph, pull_graph() takes the Graph's
changes (made in another analysis) over.
"""
import json
import os
import re

from tesselab.graph import Graph, GraphError, clone, is_var, merge, split

FORMAT = 1
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_NONE = object()


class AnalysisError(ValueError):
    pass


class CycleError(AnalysisError):
    pass


class Source:
    """Where an input comes from: an output of an instance, in this analysis (file None) or in another one."""

    def __init__(self, instance, output, file=None, bind=None, seen=None):
        self.instance, self.output, self.file, self.bind, self.seen = instance, output, file, bind or {}, seen

    @classmethod
    def parse(cls, value) -> "Source":
        if isinstance(value, Source):
            return value
        bind = seen = None
        if isinstance(value, dict):
            bind, seen, value = value.get("bind"), value.get("seen"), value.get("from")
        if not isinstance(value, str):
            raise AnalysisError(f"An input source must be text like 'steps.spectra', not {value!r}.")
        file, _, ref = value.rpartition("#")
        instance, _, output = ref.partition(".")
        if not instance or not output:
            raise AnalysisError(f"An input source looks like '<node>.<output>', not {value!r}.")
        return cls(instance, output, file or None, bind, seen)

    def ref(self) -> str:
        return (f"{self.file}#" if self.file else "") + f"{self.instance}.{self.output}"

    def to_json(self):
        if not self.bind and not self.seen:
            return self.ref()
        data = {"from": self.ref()}
        if self.bind:
            data["bind"] = dict(self.bind)
        if self.seen:
            data["seen"] = self.seen
        return data

    def __repr__(self):
        return f"Source({self.ref()!r})"


def sources_of(value) -> list[Source]:
    """The sources of one input (a list for many=True inputs); none for an unconnected input."""
    if value is None:
        return []
    if isinstance(value, list):
        return [Source.parse(v) for v in value]
    return [Source.parse(value)]


class Analysis:
    def __init__(self, doc=None, path=None, graph=None):
        self.doc = doc if doc is not None else new_doc("New analysis")
        self.path = os.path.abspath(path) if path else None
        self.graph = graph  # a shared graph file (Graph), or None: the graph is in this file
        self.base = clone(graph.doc) if graph else None
        if graph is not None:
            self.doc.setdefault("own", {})
        self.doc.setdefault("tesselab", FORMAT)
        self.doc.setdefault("label", "")
        self.doc.setdefault("comment", "")
        self.doc.setdefault("vars", {})
        self.doc.setdefault("nodes", {})
        self.doc.setdefault("results", {})
        view = self.doc.setdefault("view", {})
        view.setdefault("order", list(self.doc["nodes"]))
        view.setdefault("open", [])
        if self.doc["tesselab"] > FORMAT:
            self.read_only = True  # written by a newer Tesselab: shown, not saved over
        else:
            self.read_only = False
        self.disk_stamp = _stamp(self.path)  # the file as read or last saved (changed_on_disk)

    # --- files -------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path, graphs=None) -> "Analysis":
        """graphs: a GraphStore (the window's), so that open analyses share their graph file; without one the graph
        file is read for this analysis alone."""
        with open(path, encoding="utf-8") as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError as error:
                raise AnalysisError(f"{path}: not a valid analysis file ({error}).") from None
        if not isinstance(doc, dict) or ("nodes" not in doc and "graph" not in doc):
            raise AnalysisError(f"{path}: not an analysis file (no 'nodes').")
        if "graph" not in doc:
            return cls(doc, path)
        graph_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), doc["graph"]))
        try:
            graph = graphs.get(graph_path) if graphs is not None else Graph.load(graph_path)
        except GraphError as error:
            raise AnalysisError(f"{path}: {error}") from None
        analysis = cls(merge(graph.doc, doc), path, graph)
        analysis.read_only = analysis.read_only or graph.read_only
        return analysis

    def save(self, path=None):
        """Write the analysis file, and its graph file when the graph has unsaved changes."""
        path = os.path.abspath(path or self.path)
        if self.read_only:
            raise AnalysisError(f"{path} was written by a newer Tesselab and is shown read-only.")
        if self.graph is None:
            write_json(path, self.doc)
        else:
            self.push_graph()
            if self.graph.is_dirty() or not os.path.exists(self.graph.path):
                self.graph.save()
            sample = split(self.doc, self.base)[1]
            sample["graph"] = self.graph_ref(path)
            write_json(path, sample)
        self.path = path
        self.disk_stamp = _stamp(path)

    def graph_ref(self, path=None) -> str:
        """How the analysis file names its graph file: relative to the analysis file's folder."""
        folder = os.path.dirname(os.path.abspath(path or self.path))
        return os.path.relpath(self.graph.path, folder).replace("\\", "/")

    def changed_on_disk(self) -> bool:
        """Whether the file (or its graph file) was changed by someone else (another window, a sync client) since it
        was read or last saved here: saving would overwrite that."""
        mine = self.path is not None and _stamp(self.path) not in (None, self.disk_stamp)
        return mine or (self.graph is not None and self.graph.changed_on_disk())

    # --- the shared graph --------------------------------------------------------------------------------------
    def push_graph(self) -> bool:
        """Hand the graph part of this analysis to its Graph (after an edit here); True if the graph changed (the
        other open analyses using it then pull_graph())."""
        if self.graph is None:
            return False
        graph, _ = split(self.doc, self.base)
        changed = graph != self.graph.doc
        self.graph.doc = graph
        self.base = clone(graph)
        return changed

    def pull_graph(self):
        """Take over the Graph's changes, made in another analysis: this analysis keeps its sample."""
        if self.graph is None or self.base == self.graph.doc:
            return
        _, sample = split(self.doc, self.base)
        self.doc = merge(self.graph.doc, sample)
        self.base = clone(self.graph.doc)

    def state(self):
        """What makes this analysis unsaved when it changes: its content, or with a graph file its sample (the
        graph's own changes count through Graph.is_dirty)."""
        content = self.content()
        return content if self.graph is None else split(content, self.base)[1]

    def is_own(self, name, key) -> bool:
        """Whether the analysis sets this parameter itself (always without a graph file)."""
        return self.graph is None or key in self.doc.get("own", {}).get(name, [])

    def set_own(self, name, key, own=True):
        if self.graph is None:
            return
        keys = self.doc.setdefault("own", {}).setdefault(name, [])
        if own and key not in keys:
            keys.append(key)
        elif not own and key in keys:
            keys.remove(key)
        if not keys:
            del self.doc["own"][name]

    def graph_value(self, name, key):
        """(found, value): the graph's value of a parameter (not found: the module's default applies)."""
        params = (self.base or {}).get("nodes", {}).get(name, {}).get("params", {})
        return (key in params and not is_var(params[key])), params.get(key)

    def reset_param(self, name, key):
        """The parameter takes the graph's value again."""
        found, value = self.graph_value(name, key)
        self.set_own(name, key, False)
        if found:
            self.nodes[name]["params"][key] = clone(value)
        else:
            self.nodes[name].get("params", {}).pop(key, None)

    def promote_param(self, name, key):
        """This analysis's value becomes the graph's value (for every analysis that does not set its own)."""
        value = self.nodes[name].get("params", {}).get(key)
        if self.base is not None and name in self.base["nodes"]:
            self.base["nodes"][name]["params"][key] = clone(value)
        self.set_own(name, key, False)

    def use_graph(self, graph, own=None):
        """Use another graph file: a new shared one made from this analysis's graph (sharing it, or a separate copy),
        or an existing one, whose nodes and connections then replace this analysis's. own: the parameters this
        analysis sets itself (default: those it set so far, and every value that differs from the new graph's)."""
        doc = clone(self.doc)
        if own is not None:
            doc["own"] = own
        _, sample = split(doc, self.base)
        for name, entry in doc["nodes"].items():  # values that differ from the new graph's stay this analysis's
            target = graph.doc["nodes"].get(name)
            if target is None or target.get("node") != entry.get("node"):
                continue
            for key, value in entry.get("params", {}).items():
                theirs = target["params"].get(key, _NONE)
                if not is_var(value) and not is_var(theirs) and theirs != value:
                    sample["params"].setdefault(name, {})[key] = clone(value)
        self.graph = graph
        self.doc = merge(graph.doc, sample)
        self.base = clone(graph.doc)

    @property
    def label(self) -> str:
        return self.doc.get("label") or (os.path.splitext(os.path.basename(self.path))[0] if self.path else "")

    @property
    def nodes(self) -> dict:
        return self.doc["nodes"]

    @property
    def vars(self) -> dict:
        return self.doc["vars"]

    def content(self) -> dict:
        """What makes the analysis what it is (for undo and 'unsaved'): everything but results, view, provenance and
        the fingerprints of linked results."""
        def without_seen(value):
            if value is None:
                return None
            if isinstance(value, list):
                return [without_seen(v) for v in value]
            source = Source.parse(value)
            return Source(source.instance, source.output, source.file, source.bind).to_json()

        doc = clone({k: v for k, v in self.doc.items() if k not in ("results", "view", "provenance")})
        for instance in doc["nodes"].values():
            for name, value in list(instance.get("inputs", {}).items()):
                instance["inputs"][name] = without_seen(value)
        return doc

    def restore(self, content, content_graph=None, graph=None):
        """Undo/redo: put back a state from content(). With a graph file: content was taken while the graph was
        content_graph; this analysis's sample from then is put on `graph` (the Graph's document is set to it)."""
        keep = {k: v for k, v in self.doc.items() if k in ("results", "view", "provenance")}
        # the fingerprints of linked results seen at the last save stay (they are no part of the content)
        seen = {s.ref(): s.seen for _, _, s in self.links() if s.seen}
        if self.graph is None:
            self.doc = clone(content)
        else:
            _, sample = split(content, content_graph)
            self.graph.doc = clone(graph)
            self.doc = merge(graph, sample)
            self.base = clone(graph)
            keep["view"] = dict(keep.get("view", {}), order=self.doc["view"]["order"])
        self.doc.update(keep)
        for name, key, source in self.links():
            if source.ref() in seen:
                value = self.nodes[name]["inputs"][key]
                if isinstance(value, list):
                    self.nodes[name]["inputs"][key] = [
                        Source(s.instance, s.output, s.file, s.bind, seen.get(s.ref())).to_json()
                        for s in sources_of(value)]
                else:
                    self.nodes[name]["inputs"][key] = Source(source.instance, source.output, source.file,
                                                             source.bind, seen[source.ref()]).to_json()
        order = self.doc["view"].setdefault("order", [])
        self.doc["view"]["order"] = [n for n in order if n in self.nodes] + [n for n in self.nodes if n not in order]

    # --- editing -----------------------------------------------------------------------------------------------
    def add(self, node_id, name=None, inputs=None, params=None, version="", schema=1, label=None, pos=None) -> str:
        base = name or node_id.rpartition("/")[2].replace("-", "_")
        base = re.sub(r"[^A-Za-z0-9_]", "_", base) or "node"
        if not NAME.fullmatch(base):
            base = "n_" + base
        name, i = base, 2
        while name in self.nodes:
            name, i = f"{base}{i}", i + 1
        entry = {"node": node_id, "version": version, "schema": schema,
                 "inputs": {k: (v.to_json() if isinstance(v, Source) else v) for k, v in (inputs or {}).items()},
                 "params": dict(params or {})}
        if label:
            entry["label"] = label
        if pos is not None:
            entry["pos"] = [int(pos[0]), int(pos[1])]
        self.nodes[name] = entry
        self.doc["view"]["order"].append(name)
        return name

    def remove(self, name):
        """Remove an instance; inputs that came from it become unconnected."""
        del self.nodes[name]
        self.doc.get("own", {}).pop(name, None)
        self.doc["view"]["order"] = [n for n in self.doc["view"]["order"] if n != name]
        self.doc["view"]["open"] = [n for n in self.doc["view"]["open"] if n != name]
        self.doc["results"].pop(name, None)
        for instance in self.nodes.values():
            for key, value in list(instance.get("inputs", {}).items()):
                kept = [s for s in sources_of(value) if s.file or s.instance != name]
                if isinstance(value, list):
                    instance["inputs"][key] = [s.to_json() for s in kept]
                elif not kept:
                    instance["inputs"][key] = None

    def set_input(self, name, input_name, source):
        """source: a Source, a source string, a list of them (many=True), or None (unconnected)."""
        if isinstance(source, list):
            value = [Source.parse(s).to_json() for s in source]
        elif source is None:
            value = None
        else:
            value = Source.parse(source).to_json()
        old = self.nodes[name].setdefault("inputs", {}).get(input_name)
        self.nodes[name]["inputs"][input_name] = value
        try:
            self.order()
        except CycleError:
            self.nodes[name]["inputs"][input_name] = old
            raise

    def inputs_of(self, name) -> dict[str, list[Source]]:
        return {key: sources_of(value) for key, value in self.nodes[name].get("inputs", {}).items()}

    def dependencies(self, name) -> list[str]:
        """Instances of this analysis that `name` takes inputs from."""
        found = []
        for sources in self.inputs_of(name).values():
            for source in sources:
                if source.file is None and source.instance not in found:
                    found.append(source.instance)
        return found

    def downstream(self, name) -> list[str]:
        """`name` and every instance that depends on it, directly or not."""
        result, changed = [name], True
        while changed:
            changed = False
            for other in self.nodes:
                if other not in result and any(d in result for d in self.dependencies(other)):
                    result.append(other)
                    changed = True
        return result

    def order(self) -> list[str]:
        """The instances so that each comes after those it takes inputs from (CycleError if impossible)."""
        done, visiting, result = set(), set(), []

        def visit(name, path):
            if name in done:
                return
            if name in visiting:
                raise CycleError("The graph would loop: " + " → ".join(path + [name]))
            visiting.add(name)
            for dependency in self.dependencies(name):
                if dependency in self.nodes:
                    visit(dependency, path + [name])
            visiting.discard(name)
            done.add(name)
            result.append(name)

        for name in self.doc["view"].get("order", []) + list(self.nodes):
            if name in self.nodes:
                visit(name, [])
        return result

    def links(self) -> list[tuple[str, str, Source]]:
        """(instance, input, source) of every input that comes from another analysis."""
        return [(name, key, source) for name in self.nodes for key, sources in self.inputs_of(name).items()
                for source in sources if source.file]


def new_doc(label) -> dict:
    return {"tesselab": FORMAT, "label": label, "comment": "", "vars": {}, "nodes": {}, "results": {},
            "view": {"order": [], "open": []}}


def _stamp(path):
    """(modification time, size) of a file, None if there is none."""
    try:
        info = os.stat(path)
    except (OSError, TypeError):
        return None
    return info.st_mtime_ns, info.st_size


def write_json(path, doc):
    """Write through a temporary file and a rename: a file that is open elsewhere (Windows, sync clients) is never
    left half-written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)  # no half-written leftovers next to the user's files
        raise


def instantiate(template, values) -> dict:
    """A template (an analysis document with {placeholders}) with the placeholders filled in. A string that is only
    a placeholder takes the value with its type (a list, a number, None); inside a longer string it is formatted."""
    def fill(item):
        if isinstance(item, str):
            whole = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", item)
            if whole and whole.group(1) in values:
                return clone(values[whole.group(1)])
            return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                          lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), item)
        if isinstance(item, list):
            return [fill(v) for v in item]
        if isinstance(item, dict):
            return {k: fill(v) for k, v in item.items()}
        return item
    doc = fill(clone(template))
    for key, value in new_doc("").items():
        doc.setdefault(key, clone(value))
    if not doc["view"].get("order"):
        doc["view"]["order"] = list(doc["nodes"])
    return doc
