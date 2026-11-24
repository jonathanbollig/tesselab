"""
Running a node's run() — in a worker process (the window's way, see gui/runner.py) or right here (check, export,
tests). Kept free of Qt: worker processes import only this, the node's module and its analysis code.

run_task(task) gets everything as plain data (task: see Engine.task), imports the node class from its module folder
(again only if the module's code changed), runs it with its diagnostics collected, and returns the outputs.
"""
import ctypes
import multiprocessing
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

_classes = {}  # (folder, code hash, class spec) -> class, per process
_imported = {}  # folder -> the code hash of what this process imported from it
_pool = None
STALE = ("The module changed on disk while this node was waiting to run. It runs with the new code once the module "
         "is reloaded (at once if the window follows changes on disk, else Modules → Reload modules).")


class ModuleChanged(RuntimeError):
    pass


def node_class(task):
    """The node class of a task, imported from the module folder once per version of the module's code. The code
    imported must be the code the task's fingerprint names: a module edited again after the task was made (an editor
    or an LLM saving in quick steps) raises ModuleChanged, or a result of the new code would be kept under the old
    fingerprint."""
    folder, version = task["folder"], task["code_hash"]
    key = (folder, version, task["class_spec"])
    if key not in _classes:
        from tesselab.modules import code_hash, load_node_class
        verified = _imported.get(folder) == version
        if folder in _imported and not verified:  # another version of the module's code: import its files again
            _forget(task)
        cls = load_node_class(folder, task["module_id"], task["class_spec"])
        if not verified:
            if code_hash(folder) != version:
                _forget(task)
                raise ModuleChanged(STALE)
            _imported[folder] = version
        _classes[key] = cls
    return _classes[key]


def _forget(task):
    from tesselab.modules import forget_imports
    forget_imports(task["module_id"])
    for key in [k for k in _classes if k[0] == task["folder"]]:
        del _classes[key]
    _imported.pop(task["folder"], None)


def run_task(task) -> dict:
    """{'outputs': {name: Dataset}, 'diagnostics': [...], 'seconds': float (the computation), 'load_seconds':
    float (importing the node's module, only the first time in a process), 'error': str or None, 'stale': True if
    the module's code on disk is no longer the task's (nothing ran)}"""
    import xarray as xr
    from tesselab.node import Bunch, collecting
    start = time.perf_counter()
    outputs, error, loading, stale = {}, None, 0.0, None
    with collecting() as diagnostics:
        try:
            cls = node_class(task)
            loading = time.perf_counter() - start  # importing the module: once per worker process
            start = time.perf_counter()
            inputs = Bunch(task["inputs"])
            result = cls().run(inputs, Bunch(task["params"]))
            if result is None:
                result = {}
            if not isinstance(result, dict):
                raise TypeError(f"run() must return a dict of xarray Datasets, not {type(result).__name__}.")
            for name, value in result.items():
                if isinstance(value, xr.DataArray):
                    value = value.to_dataset(name=value.name or name)
                if not isinstance(value, xr.Dataset):
                    raise TypeError(f"run() returned {type(value).__name__} for output {name!r}; outputs are xarray "
                                    f"Datasets (or DataArrays).")
                outputs[name] = value
        except ModuleChanged as changed:
            stale = str(changed)
        except Exception as exception:
            error = "".join(traceback.format_exception_only(type(exception), exception)).strip()
            details = traceback.format_exc()
            diagnostics.append({"level": "error", "message": error, "plots": [], "code": "exception",
                                "details": details})
    if stale:  # nothing ran: not a result of this node, the runner does not keep it
        return {"outputs": {}, "diagnostics": [], "seconds": 0.0, "load_seconds": 0.0, "error": stale, "stale": True}
    return {"outputs": outputs, "diagnostics": list(diagnostics), "seconds": time.perf_counter() - start,
            "load_seconds": loading, "error": error}


def warm(specs):
    """Import node classes in this worker ahead of time (the window does this when a project opens), so that the
    first computation does not wait for imports. specs: dicts with folder, module_id, code_hash, class_spec."""
    import importlib
    # what scientific code imports on its first call (xarray's interp imports scipy.interpolate, ~0.3 s): ahead too
    for name in ("numpy", "xarray", "pandas", "scipy.interpolate", "scipy.optimize"):
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    for spec in specs:
        try:
            node_class(spec)
        except Exception:
            pass  # a broken module reports its error when it is used


def run_action(task, action_id, context):
    """An action with job=True, in a worker: the node class's action `action_id`, called with the draw context."""
    cls = node_class(task)
    action = next(a for a in cls.actions if a.id == action_id)
    return action.run(context)


def _init_process():
    """In each worker: Ctrl+C is meant for the window; and the worker ends when the window ends (Linux:
    PR_SET_PDEATHSIG; elsewhere a thread that watches the parent)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        parent = os.getppid()

        def watch():
            while True:
                time.sleep(2)
                if os.getppid() != parent:
                    os._exit(0)
        threading.Thread(target=watch, daemon=True).start()


def pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is not None and getattr(_pool, "_broken", False):
        _pool = None
    if _pool is None:
        # "spawn": the workers are children of the window itself and never import Qt
        _pool = ProcessPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 1)), initializer=_init_process,
                                    mp_context=multiprocessing.get_context("spawn"))
    return _pool


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
