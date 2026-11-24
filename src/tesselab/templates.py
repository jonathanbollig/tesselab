"""
New analyses from a module's template: the template file (an analysis with {placeholders}) filled in for a data
folder. The values come from the template's fill function (fill = "nodes.py:function" in the manifest:
function(folder, project_root) -> dict), or by default: name (the folder's name) and folder (relative to the
project).

In a project the analyses made from one template share a graph file (graph.py): the first one makes
tesselab/graphs/<template file name>.json, the later ones use it, with the values filled in from their data (the
template's placeholders) as their own.
"""
import copy
import os

from tesselab.analysis import Analysis, instantiate
from tesselab.graph import Graph, GraphError, merge, own_from_template, split
from tesselab.modules import read_template


def new_analysis(registry, project, template_id, folder, label=None, graphs=None) -> Analysis:
    """graphs: the window's GraphStore (a graph file made by an analysis that is not saved yet is used too)."""
    template = registry.template(template_id)
    module = template.module
    root = project.root if project else os.path.dirname(os.path.abspath(folder))
    values = {"name": os.path.basename(os.path.normpath(folder)),
              "folder": os.path.relpath(folder, root).replace("\\", "/")}
    if template.fill:
        values.update(registry.function(module, template.fill)(folder, root))
    raw = read_template(module, template)
    doc = instantiate(raw, values)
    doc.setdefault("provenance", {})["template"] = template.full_id
    if label:
        doc["label"] = label
    elif not doc.get("label"):
        doc["label"] = values["name"]
    for entry in doc["nodes"].values():  # the version of the nodes the analysis is made with
        try:
            info = registry.node(entry["node"])
            entry.setdefault("version", info.module.version)
            entry.setdefault("schema", registry.node_class(entry["node"]).schema)
            entry.setdefault("inputs", {})
            entry.setdefault("params", {})
        except KeyError:
            pass
    if project is None:
        return Analysis(doc, None)  # no project: no place for a shared graph, the graph stays in the analysis
    path = project.new_analysis_path(doc["label"])
    graph_path = project.folder("graphs", os.path.splitext(os.path.basename(template.file))[0] + ".json")
    doc["own"] = own_from_template(raw)
    graph = graphs.find(graph_path) if graphs is not None else None
    if graph is None and os.path.exists(graph_path):
        try:
            graph = graphs.get(graph_path) if graphs is not None else Graph.load(graph_path)
        except GraphError:  # a broken graph file is neither used nor overwritten: the graph stays in the analysis
            return Analysis(doc_without_own(doc), path)
    made, sample = split(doc)
    if graph is None:
        made["label"] = template.name
        made["provenance"] = {"template": template.full_id}
        graph = Graph(made, graph_path)
        if graphs is not None:
            graphs.add(graph)
    whole = merge(graph.doc, sample)
    analysis = Analysis(whole, path, graph)
    analysis.doc["graph"] = analysis.graph_ref()
    return analysis


def doc_without_own(doc):
    doc = copy.deepcopy(doc)
    doc.pop("own", None)
    return doc


def empty_analysis(project, label, graphs=None) -> Analysis:
    """A new analysis without nodes; in a project with a graph file of its own (graphs/<label>.json)."""
    analysis = Analysis(path=project.new_analysis_path(label) if project else None)
    analysis.doc["label"] = label
    if project is not None:
        graph = Graph({"label": label}, project.new_graph_path(label))
        if graphs is not None:
            graphs.add(graph)
        analysis.use_graph(graph, own={})
        analysis.doc["graph"] = analysis.graph_ref()
    return analysis
