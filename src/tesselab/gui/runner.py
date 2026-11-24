"""
Runs the nodes of open analyses in worker processes, so the window stays responsive.

request(analysis) asks for the current state of an analysis: every node whose result is not known for its current
fingerprint is run as soon as its inputs are there; independent nodes run at the same time. updated(analysis) is
emitted whenever a result arrives; results(analysis) gives the results for the current state (a node still running
is missing). A result of an older state is kept (under its own fingerprint, it may be needed again after an undo)
but never shown for the new one: that is how a newer result always wins.

inline=True runs everything right here, one node after the other (tests, export).
"""
import traceback

from PySide6 import QtCore

from tesselab import worker
from tesselab.analysis import AnalysisError
from tesselab.engine import Key, NodeResult


class Runner(QtCore.QObject):
    updated = QtCore.Signal(object)       # an analysis whose results changed
    _finished = QtCore.Signal(str)        # from the thread that collects worker results

    def __init__(self, engine, inline=False):
        super().__init__()
        self.engine, self.inline = engine, inline
        self.pending = {}     # fingerprint -> (key, task, future, analyses waiting for it)
        self.current = {}     # analysis -> {name: NodeResult}
        self.running = {}     # analysis -> set of names running
        self.forced = set()   # (analysis, name) of run-on-button nodes asked to run (dropped by forget)
        self.stale = {}       # fingerprint -> NodeResult: sent to run, but its module changed on disk before (cleared
                              # when modules are reloaded; kept apart so that it is not sent again and again)
        self._finished.connect(self._collect, QtCore.Qt.ConnectionType.QueuedConnection)

    def request(self, analysis, run=(), fresh=True):
        """Compute what the analysis needs; run: names of run-on-button nodes to run now. fresh=False: files are not
        read again if they were looked at a moment ago (an edit, such as a dragged λ, changes no file)."""
        if fresh:
            self.engine.forget_stamps()
        for name in run:
            self.forced.add((analysis, name))
        self._advance(analysis)

    def warm(self, registry):
        """Start the worker processes and import every available module's nodes in them, in the background (when
        a project opens), so that the first computations do not wait for process starts and imports."""
        if self.inline:
            return
        specs = [{"folder": node.module.folder, "module_id": node.module.id, "code_hash": node.module.code_hash,
                  "class_spec": node.class_spec} for node in registry.nodes().values()]
        pool = worker.pool()
        for _ in range(pool._max_workers):
            pool.submit(worker.warm, specs)

    def forget(self, analysis):
        self.current.pop(analysis, None)
        self.running.pop(analysis, None)
        self.forced = {(a, name) for a, name in self.forced if a is not analysis}

    def drop_pending(self):
        """Forget the tasks sent to the workers (Reload modules shuts the pool down: their late callbacks then find
        nothing, instead of storing a cancelled task as the node's result)."""
        self.pending.clear()
        self.stale.clear()
        self.running = {analysis: set() for analysis in self.running}

    def results(self, analysis) -> dict:
        return self.current.get(analysis, {})

    def is_running(self, analysis) -> set:
        return self.running.get(analysis, set())

    def _advance(self, analysis):
        engine = self.engine
        results, running = {}, set()
        try:
            keys = engine.needed([Key(analysis, name) for name in analysis.order()])
        except (AnalysisError, KeyError, RuntimeError) as error:
            self.current[analysis] = {name: NodeResult("", status="blocked", diagnostics=[
                {"level": "error", "message": str(error), "plots": [], "code": "graph"}]) for name in analysis.nodes}
            self.running[analysis] = set()
            self.updated.emit(analysis)
            return
        submitted = []
        for key in keys:
            if any(dep in running or dep not in results for dep in engine.dependencies(key)):
                running.add(key)  # waits for its inputs (their content is part of its fingerprint)
                continue
            try:
                fingerprint = engine.fingerprint(key, results)
            except Exception as error:  # e.g. the node's module is missing
                results[key] = NodeResult("", status="error", error=str(error), diagnostics=[
                    {"level": "error", "message": str(error), "plots": [], "code": "module"}])
                continue
            if fingerprint in self.pending:
                self.pending[fingerprint][3].add(analysis)
                running.add(key)
                continue
            if fingerprint in self.stale:
                results[key] = self.stale[fingerprint]
                continue
            forced = (key.analysis, key.name) in self.forced
            try:
                state, value = engine.prepare(key, results, force=forced, fingerprint=fingerprint)
            except Exception as error:
                results[key] = NodeResult(fingerprint, status="error", error=str(error), diagnostics=[
                    {"level": "error", "message": str(error), "plots": [], "code": "prepare",
                     "details": traceback.format_exc()}])
                continue
            if state == "done":
                results[key] = value
                continue
            self.forced.discard((key.analysis, key.name))
            if self.inline:
                results[key] = engine.finish(key, value, worker.run_task(value))
                continue
            future = worker.pool().submit(worker.run_task, value)
            self.pending[fingerprint] = (key, value, future, {analysis})
            running.add(key)
            submitted.append((fingerprint, future))
        own = {key.name: result for key, result in results.items() if key.analysis is analysis}
        for name, result in own.items():
            result.diagnostics = [d for d in result.diagnostics if d.get("code") != "link-changed"] + \
                engine.link_warnings(analysis, name, results)
        self.current[analysis] = own
        self.running[analysis] = {key.name for key in running if key.analysis is analysis}
        self.updated.emit(analysis)
        for fingerprint, future in submitted:
            future.add_done_callback(lambda f, fp=fingerprint: self._finished.emit(fp))

    def _collect(self, fingerprint):
        entry = self.pending.pop(fingerprint, None)
        if entry is None:
            return
        key, task, future, waiting = entry
        if future.cancelled():
            return  # the pool was shut down; never a result
        try:
            raw = future.result()
        except Exception as error:  # the worker process died
            raw = {"outputs": {}, "diagnostics": [], "seconds": 0.0, "error": f"The worker stopped: {error}"}
        try:
            result = self.engine.finish(key, task, raw)
            if raw.get("stale"):
                self.stale[fingerprint] = result
        except Exception:
            traceback.print_exc()
        for analysis in waiting:
            if analysis in self.current or analysis is key.analysis:
                self._advance(analysis)
