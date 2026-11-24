"""
Projects: a folder with your data and a tesselab/ folder holding your work.

    <project>/
        tesselab/
            project.toml      marks the project: name, settings
            analyses/         one JSON file per analysis
            sessions/         saved windows (*.tesselab-session)
            snapshots/        frozen links to other analyses
            modules/          frozen copies of modules (optional)
            cache/            derived results, safe to delete
        ... your data, anywhere in the project ...

A project is recognized by tesselab/project.toml, not by the folder name alone. Whether the modules inside a project
may run (they are code that came with the folder) is the user's decision, stored in the user's own settings, never
in the project.
"""
import fnmatch
import json
import os
import re
import tomllib

import platformdirs

from tesselab.analysis import write_json

FOLDER = "tesselab"
SKIP = {FOLDER, "__pycache__", "node_modules"}


def find_root(path) -> str | None:
    """The project that contains `path` (itself or a parent), None if there is none."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    while True:
        if os.path.isfile(os.path.join(path, FOLDER, "project.toml")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"Cannot write {value!r} to project.toml.")


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.settings = {}
        path = os.path.join(self.root, FOLDER, "project.toml")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                self.settings = tomllib.load(f)

    @classmethod
    def create(cls, root, name=None) -> "Project":
        """Make `root` a project (if it is not one yet)."""
        root = os.path.abspath(root)
        os.makedirs(os.path.join(root, FOLDER, "analyses"), exist_ok=True)
        path = os.path.join(root, FOLDER, "project.toml")
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("# A Tesselab project: your analyses are in analyses/, the cache in cache/ can be deleted.\n")
                f.write(f"[project]\nname = {_toml_value(name or os.path.basename(root))}\n")
        return cls(root)

    @property
    def name(self) -> str:
        return self.settings.get("project", {}).get("name") or os.path.basename(self.root)

    def folder(self, *parts) -> str:
        return os.path.join(self.root, FOLDER, *parts)

    def analyses(self) -> list[str]:
        """Paths of the analysis files, sorted naturally (2_UU before 12_UA)."""
        folder = self.folder("analyses")
        if not os.path.isdir(folder):
            return []
        return [os.path.join(folder, name) for name in sorted(os.listdir(folder), key=natural_key)
                if name.endswith(".json")]

    def sessions(self) -> list[str]:
        folder = self.folder("sessions")
        if not os.path.isdir(folder):
            return []
        return [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith(".tesselab-session")]

    def relative(self, path) -> str:
        """A path as written into project files: relative to the project with /, absolute if it is outside."""
        path = os.path.abspath(path)
        relative = os.path.relpath(path, self.root) if not _other_drive(path, self.root) else None
        if relative is None or relative.startswith(".."):
            return path.replace("\\", "/")
        return relative.replace("\\", "/")

    def absolute(self, path) -> str:
        return os.path.normpath(os.path.join(self.root, path)) if path and not os.path.isabs(path) else path

    def new_analysis_path(self, label) -> str:
        return _free_path(self.folder("analyses"), label)

    def new_graph_path(self, label) -> str:
        return _free_path(self.folder("graphs"), label)

    def graphs(self) -> list[str]:
        """Paths of the graph files (tesselab/graphs/*.json), sorted naturally."""
        folder = self.folder("graphs")
        if not os.path.isdir(folder):
            return []
        return [os.path.join(folder, n) for n in sorted(os.listdir(folder), key=natural_key) if n.endswith(".json")]


def _free_path(folder, base) -> str:
    base = re.sub(r'[<>:"/\\|?*]', "_", base).strip() or "analysis"
    path, i = os.path.join(folder, f"{base}.json"), 2
    while os.path.exists(path):
        path, i = os.path.join(folder, f"{base} ({i}).json"), i + 1
    return path


def _other_drive(a, b):
    return os.path.splitdrive(a)[0].lower() != os.path.splitdrive(b)[0].lower()


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


# --- trust: whether a project's own modules may run --------------------------------------------------------------
def _trust_file():
    return os.path.join(platformdirs.user_config_dir("tesselab", appauthor=False), "trust.json")


def _read_trust():
    try:
        with open(_trust_file(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"trusted": [], "declined": []}


def trust_state(root) -> str | None:
    """'trusted', 'declined' or None (not asked yet)."""
    data, root = _read_trust(), os.path.abspath(root)
    if root in data.get("trusted", []):
        return "trusted"
    if root in data.get("declined", []):
        return "declined"
    return None


def set_trust(root, trusted: bool):
    data, root = _read_trust(), os.path.abspath(root)
    for key in ("trusted", "declined"):
        data[key] = [r for r in data.get(key, []) if r != root]
    data["trusted" if trusted else "declined"].append(root)
    write_json(_trust_file(), data)


def has_own_modules(root) -> bool:
    folder = os.path.join(root, FOLDER, "modules")
    return os.path.isdir(folder) and any(os.path.isfile(os.path.join(folder, n, "module.toml"))
                                         for n in os.listdir(folder))


# --- data recognized by templates --------------------------------------------------------------------------------
def _matches(folder, pattern, names_cache):
    """Whether some file or folder in `folder` matches a glob relative to it (e.g. 'powers/*.csv', 'data/*.csv')."""
    directory, _, name_pattern = pattern.rpartition("/")
    path = os.path.join(folder, directory) if directory else folder
    if path not in names_cache:
        try:
            names_cache[path] = os.listdir(path)
        except OSError:
            names_cache[path] = []
    if name_pattern.endswith("/"):
        name_pattern = name_pattern[:-1]
    return any(fnmatch.fnmatch(n.lower(), name_pattern.lower()) for n in names_cache[path])


def recognizes(template, folder, names_cache=None) -> bool:
    rules = template.recognize
    if not rules:
        return False
    cache = {} if names_cache is None else names_cache
    if "all" in rules and not all(_matches(folder, p, cache) for p in rules["all"]):
        return False
    if "any" in rules and not any(_matches(folder, p, cache) for p in rules["any"]):
        return False
    return "all" in rules or "any" in rules


def find_data(root, templates, max_depth=6, detect=None) -> list[tuple[str, list]]:
    """Folders in the project that templates recognize: [(folder, [template, ...]), ...]. The folders inside a
    recognized one are not searched further; hidden folders and tesselab/ are skipped. detect(template, folder):
    the closer look of templates that have a detect function (it imports module code, so it is given only for
    modules that may run)."""
    found = []
    templates = [t for t in templates if t.recognize]
    if not templates:
        return found
    root = os.path.abspath(root)
    for folder, dirs, _ in os.walk(root):
        depth = folder[len(root):].count(os.sep)
        cache = {}
        matching = [t for t in templates if recognizes(t, folder, cache)
                    and (not t.detect or detect is None or _safe(detect, t, folder))]
        if matching and folder != root:
            found.append((folder, matching))
            dirs.clear()
            continue
        dirs[:] = sorted((d for d in dirs if d not in SKIP and not d.startswith(".")), key=natural_key) \
            if depth < max_depth else []
    return found


def _safe(detect, template, folder) -> bool:
    try:
        return bool(detect(template, folder))
    except Exception:
        return False
