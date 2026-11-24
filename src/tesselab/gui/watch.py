"""
Following changes on disk (Modules → Follow changes on disk): when a module's code changes, the window reloads that
module and computes again; when a file that an open analysis reads changes, that analysis is computed again. So an
edit in an editor (or by an LLM) shows at once, and so does a measurement that is still being written.

One QFileSystemWatcher watches
- the module folders (the files modules.code_files lists: what a module's code hash is made of) and the folders
  modules are searched in (a module added);
- the files and folders the path parameters of the open analyses name: a folder with every file inside it (a file
  written in place is seen only by a watch on the file itself), a file with its folder (an editor that saves by
  writing a new file and renaming it replaces the watched file; a file that does not exist yet appears there).

Changes are collected until none came for DELAY ms (editors and the rig save in several steps), then handled at once:
the modules are read again (Registry.refresh: only those whose code hash changed count), and the analyses concerned
are requested. Nothing more is needed: fingerprints hold the module's code hash and the stamps of the files a node
reads, so only the nodes a change concerns run again (and what depends on their results, if those changed). The
project's own tesselab/ folder (analyses, cache) is never watched for data: the window writes there itself.
"""
import os
import re

from PySide6 import QtCore

from tesselab import params as P
from tesselab.modules import code_files, module_folders

DELAY = 500        # ms without a change before acting
MAX_PATHS = 20000  # at most this many watched paths (each takes one of the user's inotify watches)


def _ignored(name):
    """Temporary and hidden files: lock files of office programs, editor backups and swap files, caches."""
    return name.startswith((".", "~$")) or name.endswith(("~", ".swp", ".swx", ".tmp")) or name == "__pycache__"


def _under(path, folder):
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def data_roots(engine, analysis) -> list[str]:
    """The files and folders that the path parameters of an analysis name, absolute (a pattern: the files it matches
    and the folder it searches, where new files may appear)."""
    roots = []
    base = engine.base_folder(analysis)
    for name in analysis.nodes:
        try:
            cls = engine.node_class(analysis, name)
            values, _ = engine.resolve_params(analysis, name, absolute=False)
        except Exception:  # a module that cannot be imported: its nodes read nothing
            continue
        for p in cls.params:
            value = values.get(p.name)
            if not value or not isinstance(p, (P.Path, P.Paths)):
                continue
            if isinstance(p, P.Paths) and isinstance(value, str):
                fixed = re.split(r"[*?\[]", value, maxsplit=1)[0]
                roots.append(os.path.normpath(os.path.join(base, os.path.dirname(fixed))))
            found = engine.absolute_paths(p, value, base)
            roots += [found] if isinstance(found, str) else found
    return roots


class Follow(QtCore.QObject):
    """Watches the modules and the data of the open analyses for the window, and tells it what changed."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.on = False
        self.watcher = QtCore.QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._seen)
        self.watcher.directoryChanged.connect(self._seen)
        self.changes = set()
        self.timer = QtCore.QTimer(self, singleShot=True, interval=DELAY)
        self.timer.timeout.connect(self._act)
        self.update_timer = QtCore.QTimer(self, singleShot=True, interval=DELAY)  # the watched paths, after edits
        self.update_timer.timeout.connect(self.update)
        self.roots = {}  # id(analysis) -> (analysis, its data roots)

    def set_on(self, on):
        self.on = on
        self.changes.clear()
        self.timer.stop()
        if on:
            self.update()
        else:
            self._watch(set())
            self.roots = {}

    def schedule(self):
        """Analyses were opened, closed or edited (a path may have changed): update the watched paths a moment
        later, once after a series of edits (a dragged cursor edits many times)."""
        if self.on:
            self.update_timer.start()

    # --- what is watched ---------------------------------------------------------------------------------------------
    def update(self):
        if not self.on:
            return
        paths = set()
        for base, _ in self.window.registry.bases():
            base = os.path.abspath(base)  # --modules may be relative; changes are matched by absolute paths
            paths.add(base)
            for folder in module_folders(base):
                for subfolder, files in code_files(folder):
                    paths.add(subfolder)
                    paths.update(files)
        own = self.window.project.folder() if self.window.project else None
        self.roots = {}
        for card in self.window.cards():
            analysis = card.analysis
            if id(analysis) not in self.roots:
                roots = [r for r in data_roots(self.window.engine, analysis) if not (own and _under(r, own))]
                self.roots[id(analysis)] = (analysis, roots)
                for root in roots:
                    paths |= self._data_paths(root, own, len(paths))
        self._watch(paths)

    @staticmethod
    def _data_paths(root, own, count):
        """What to watch for one data root: a folder with its subfolders and files, a file with its folder."""
        if os.path.isdir(root):
            found = set()
            for folder, dirs, files in os.walk(root):
                if count + len(found) >= MAX_PATHS:  # a folder parameter set to / or a home folder: watch a part
                    break
                dirs[:] = [d for d in dirs if not _ignored(d) and not (own and _under(os.path.join(folder, d), own))]
                found.add(folder)
                found.update(os.path.join(folder, f) for f in files if not _ignored(f))
            return found
        return {root, os.path.dirname(root)}  # the file (if it exists, _watch checks) and its folder

    def _watch(self, paths):
        """Watch exactly these paths (those that exist). A file that a save replaced (written anew and renamed over
        the old one) is no longer watched by Qt: it is not in Qt's lists any more, so it is added again here."""
        watched = set(self.watcher.files()) | set(self.watcher.directories())
        gone = watched - paths
        if gone:
            self.watcher.removePaths(sorted(gone))
        new = sorted(p for p in paths - watched if os.path.exists(p))
        if new:
            self.watcher.addPaths(new)

    # --- changes -------------------------------------------------------------------------------------------------
    def _seen(self, path):
        self.changes.add(path)
        self.timer.start()  # again: act once the saving is over

    def _act(self):
        changes, self.changes = self.changes, set()
        if not self.on or not changes:
            return
        window = self.window
        registry = window.registry
        module_paths = [os.path.abspath(base) for base, _ in registry.bases()]
        messages = []
        if any(_under(path, base) for path in changes for base in module_paths):
            changed = registry.refresh()
            if changed:
                names = [registry.modules[i].name if i in registry.modules else i for i in changed]
                files = sorted({os.path.basename(p) for p in changes if os.path.isfile(p)
                                and any(_under(p, base) for base in module_paths)})
                messages.append(f"{', '.join(files) or 'Module files'} changed: {', '.join(names)} reloaded"
                                + (" (a module cannot be read: Modules → Modules… shows why)" if registry.errors
                                   else ""))
                window.modules_changed(changed)  # computes every open analysis again
        concerned, hits = [], set()
        for analysis, roots in self.roots.values():
            matched = {path for path in changes for root in roots
                       if path == os.path.dirname(root) or _under(path, root)}
            if matched:
                concerned.append(analysis)
                hits |= matched
        concerned = window.data_changed(concerned)  # those still open, computed again
        if concerned:
            names = sorted({os.path.basename(p) + ("/" if os.path.isdir(p) else "") for p in hits})
            messages.append(f"{', '.join(names[:3])}{' …' if len(names) > 3 else ''} changed: "
                            f"{', '.join(a.label for a in concerned)} computed again")
        if messages:
            window.statusBar().showMessage("; ".join(messages), 6000)
        self.update()  # new files in watched folders, and files a save replaced
