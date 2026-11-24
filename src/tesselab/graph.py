"""
Graph files: what an analysis is made of (its nodes, their connections, which parameters follow an analysis
variable, the parameters' values, where the nodes sit in the graph window) in a file of its own, so that the analyses
of many experiments share one graph. A module's template makes the project's graph file when the first analysis is
made from it (tesselab/graphs/<template>.json); later analyses from that template use the same file.

    {"tesselab_graph": 1, "label": "Quantum yield", "comment": "",
     "vars": {"lambda": 2.6e-7}, "var_pos": {"lambda": [11, -5]},
     "nodes": {"qy": {"node": "<module id>/<node id>", "version": "1.2.0", "schema": 3, "pos": [22, 0],
                      "inputs": {"spectra": "steps.spectra"},
                      "params": {"lambda_spectrum": {"var": "lambda"}, "fit_num": 5}}},
     "order": ["spectra", "powers", "steps", "qy"], "provenance": {"template": "<module id>/<template id>"}}

An analysis that uses a graph file keeps only its sample there:

    {"tesselab": 1, "label": "5_UC@2", "graph": "../graphs/quantum_yield.json",
     "params": {"spectra": {"folder": "data/5_UC@2"}, "qy": {"fit_num": 4}}, "vars": {"lambda": 2.55e-7},
     "inputs": {"steps": {"spectra": "analyses/buffer.json#spectra.spectra"}}, "results": {...}, "view": {...}}

Its "params" are the values it sets itself, its own parameters: the values a template fills in from the data (data
paths among them) and every value changed in this analysis; any other parameter takes the graph's value, so changing
the graph's value changes every analysis that does not set its own. Variables' values and links to other analyses are
always the analysis's own. Positions are in grid units (GRID pixels in the graph window).

In memory an analysis is always whole, as if it had no graph file (merge() puts graph and sample together), so the
engine and the nodes see no difference; split() takes it apart again. A parameter bound to a variable in the graph
stays bound in every analysis.
"""
import copy
import json
import os

FORMAT = 1
GRID = 20
NODE_KEYS = ("node", "version", "schema", "label", "pos", "off")  # a node's entry in the graph, besides inputs and
# params; off: a view switched off (not shown)
_MISSING = object()


class GraphError(ValueError):
    pass


def clone(value):
    """A copy of a document (dicts, lists, numbers, strings: what a JSON file holds), about six times faster than
    copy.deepcopy, which every edit used a dozen times (dragging λ spent a tenth of each step on it, 2026-11-24).
    Anything else is copied by deepcopy."""
    kind = type(value)
    if kind is dict:
        return {key: clone(item) for key, item in value.items()}
    if kind is list:
        return [clone(item) for item in value]
    if value is None or kind in (str, int, float, bool):
        return value
    return copy.deepcopy(value)


def is_var(value) -> bool:
    return isinstance(value, dict) and set(value) == {"var"}


def is_link(value) -> bool:
    """Whether an input (one source or a list) comes from another analysis ("analyses/x.json#node.output")."""
    values = value if isinstance(value, list) else [value]
    return any("#" in (v.get("from", "") if isinstance(v, dict) else v or "") for v in values)


def normal(doc) -> dict:
    """A graph document with every key, each node with inputs and params, order only of existing nodes (then the
    others): two graphs compare equal exactly when they mean the same."""
    doc = clone(doc or {})
    doc.setdefault("tesselab_graph", FORMAT)
    doc.setdefault("label", "")
    doc.setdefault("comment", "")
    doc.setdefault("vars", {})
    doc["var_pos"] = {k: v for k, v in doc.get("var_pos", {}).items() if k in doc["vars"]}
    nodes = doc.setdefault("nodes", {})
    for entry in nodes.values():
        entry.setdefault("inputs", {})
        entry.setdefault("params", {})
    order = doc.get("order", [])
    doc["order"] = [n for n in order if n in nodes] + [n for n in nodes if n not in order]
    return doc


def merge(graph, sample) -> dict:
    """The whole analysis document from a graph and an analysis file's sample. doc["own"] lists the parameters the
    analysis sets itself ({node: [parameter, ...]})."""
    graph = normal(graph)
    doc = {k: clone(v) for k, v in sample.items() if k not in ("params", "inputs", "vars", "own")}
    nodes, own = {}, {}
    for name, g in graph["nodes"].items():
        entry = clone(g)
        for key, value in sample.get("inputs", {}).get(name, {}).items():
            entry["inputs"][key] = clone(value)
        for key, value in sample.get("params", {}).get(name, {}).items():
            if not is_var(entry["params"].get(key)):  # bound to a variable in the graph: bound everywhere
                entry["params"][key] = clone(value)
                own.setdefault(name, []).append(key)
        nodes[name] = entry
    doc["nodes"], doc["own"] = nodes, own
    doc["vars"] = {name: clone(sample.get("vars", {}).get(name, value)) for name, value in graph["vars"].items()}
    doc["var_pos"] = clone(graph["var_pos"])
    view = doc.setdefault("view", {})
    view["order"] = list(graph["order"])
    view.setdefault("open", [])
    for key, value in (("label", ""), ("comment", ""), ("results", {})):
        doc.setdefault(key, value)
    return doc


def split(doc, base=None) -> tuple[dict, dict]:
    """(graph, sample) of a whole analysis document. base: the graph the document was made from (its parameter values
    stay the graph's where the analysis sets its own). A parameter not marked as the analysis's own but changed from
    the graph's value counts as its own too: a sample's value is never pushed into the graph by mistake."""
    base = normal(base)
    own = doc.get("own", {})
    nodes, params, inputs = {}, {}, {}
    for name, entry in doc["nodes"].items():
        old = base["nodes"].get(name, {})
        same = old.get("node") == entry.get("node")
        old_params = old.get("params", {}) if same else {}
        g = {k: clone(entry[k]) for k in NODE_KEYS if k in entry}
        g["inputs"], g["params"] = {}, {}
        for key, value in entry.get("inputs", {}).items():
            if is_link(value):  # the analysis's own; the graph keeps the connection it had
                inputs.setdefault(name, {})[key] = clone(value)
                kept = old.get("inputs", {}).get(key) if same else None
                g["inputs"][key] = None if is_link(kept) else clone(kept)
            else:
                g["inputs"][key] = clone(value)
        for key, value in entry.get("params", {}).items():
            default = old_params.get(key, _MISSING)
            graph_value = default is not _MISSING and not is_var(default)
            if is_var(value):
                g["params"][key] = clone(value)
            elif key in own.get(name, ()) or (graph_value and default != value):
                params.setdefault(name, {})[key] = clone(value)
                if graph_value:
                    g["params"][key] = clone(default)
            else:
                g["params"][key] = clone(value)
        nodes[name] = g
    graph = {k: clone(v) for k, v in base.items() if k not in ("nodes", "vars", "var_pos", "order")}
    graph["nodes"] = nodes
    variables = doc.get("vars", {})
    graph["vars"] = {k: clone(base["vars"][k] if k in base["vars"] else v) for k, v in variables.items()}
    graph["var_pos"] = clone(doc.get("var_pos", {}))
    graph["order"] = list(doc.get("view", {}).get("order", []))
    sample = {k: clone(v) for k, v in doc.items() if k not in ("nodes", "own", "var_pos")}
    sample["view"] = {k: v for k, v in sample.get("view", {}).items() if k != "order"}
    sample["params"], sample["inputs"], sample["vars"] = params, inputs, clone(variables)
    return normal(graph), sample


def own_from_template(template) -> dict:
    """{node: [parameters]} whose value in a template is filled in from the data (a {placeholder}): the sample's."""
    own = {}
    for name, entry in template.get("nodes", {}).items():
        for key, value in entry.get("params", {}).items():
            if "{" in json.dumps(value):
                own.setdefault(name, []).append(key)
    return own


class Graph:
    """A graph file in memory. The open analyses that use the same file share one Graph (GraphStore)."""

    def __init__(self, doc=None, path=None):
        self.doc = normal(doc)
        self.path = os.path.abspath(path) if path else None
        self.saved = None  # the document as last read or written (None: not written yet)
        self.read_only = self.doc["tesselab_graph"] > FORMAT
        self.disk_stamp = None

    @classmethod
    def load(cls, path) -> "Graph":
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            raise GraphError(f"The graph file {path} does not exist.") from None
        except json.JSONDecodeError as error:
            raise GraphError(f"{path}: not a valid graph file ({error}).") from None
        if not isinstance(doc, dict) or "nodes" not in doc:
            raise GraphError(f"{path}: not a graph file (no 'nodes').")
        graph = cls(doc, path)
        graph.saved = clone(graph.doc)
        graph.disk_stamp = _stamp(graph.path)
        return graph

    @property
    def label(self) -> str:
        return self.doc.get("label") or (os.path.splitext(os.path.basename(self.path))[0] if self.path else "")

    def is_dirty(self) -> bool:
        return self.doc != self.saved

    def changed_on_disk(self) -> bool:
        return self.path is not None and self.saved is not None and _stamp(self.path) not in (None, self.disk_stamp)

    def save(self):
        from tesselab.analysis import write_json
        if self.read_only:
            raise GraphError(f"{self.path} was written by a newer Tesselab and is shown read-only.")
        write_json(self.path, self.doc)
        self.saved = clone(self.doc)
        self.disk_stamp = _stamp(self.path)


class GraphStore:
    """The graph files of the analyses open in the window, by path, so that they share one Graph each."""

    def __init__(self):
        self.graphs: dict[str, Graph] = {}

    def get(self, path) -> Graph:
        path = os.path.abspath(path)
        if path not in self.graphs:
            self.graphs[path] = Graph.load(path)
        return self.graphs[path]

    def add(self, graph):
        self.graphs[graph.path] = graph

    def find(self, path):
        return self.graphs.get(os.path.abspath(path))

    def drop(self, graph):
        if graph.path and self.graphs.get(graph.path) is graph:
            del self.graphs[graph.path]


def _stamp(path):
    try:
        info = os.stat(path)
    except (OSError, TypeError):
        return None
    return info.st_mtime_ns, info.st_size
