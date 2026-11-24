"""
Modules: installed folders of Python code that supply nodes and templates.

A module is a folder with a manifest, module.toml (see AGENTS.md), and the files it names. Tesselab finds modules in
these places, in this order; the first module with a given id wins:

1. the project: <project>/tesselab/modules/<folder>/ (archived copies; their code came with the folder, so the user
   is asked once per project whether to run it, see project.py)
2. the user's module folder (platformdirs: ~/.local/share/tesselab/modules on Linux), where modules are built
3. the modules that come with Tesselab (tesselab/builtin_modules)
4. installed packages that register a module folder under the entry point group "tesselab.modules"
5. extra folders given on the command line or in the environment variable TESSELAB_MODULE_PATH (os.pathsep-separated)

Manifests are read without importing any module code; a node class is imported on first use. Each module folder is
imported as a package of its own (tesselab_modules.<id>), so its files can import each other with relative imports
(from . import analysis). load_node_class does the same in worker processes, from the folder's path.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import tomllib
import types

import platformdirs

API_VERSION = 1
BUILTIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builtin_modules")
PACKAGE_PREFIX = "tesselab_modules"


def user_module_folder() -> str:
    return os.path.join(platformdirs.user_data_dir("tesselab", appauthor=False), "modules")


class ManifestError(ValueError):
    pass


class NodeInfo:
    """A node as the manifest describes it; the class is imported only when needed (Registry.node_class)."""

    def __init__(self, module, id, class_spec, files=()):
        self.module, self.id, self.class_spec, self.files = module, id, class_spec, list(files)

    @property
    def full_id(self):
        return f"{self.module.id}/{self.id}"


class TemplateInfo:
    """A ready-made analysis graph: file (JSON with {placeholders}), recognize (globs that a data folder must match:
    all = every pattern matches a file, any = at least one does), detect: "file.py:function" (optional, a closer look:
    function(folder) -> bool), fill: "file.py:function" (optional, function(folder, project_folder) -> dict of
    placeholder values; default: folder, name)."""

    def __init__(self, module, id, name, file, recognize=None, detect=None, fill=None, help=""):
        self.module, self.id, self.name, self.file = module, id, name, file
        self.recognize, self.detect, self.fill, self.help = recognize or {}, detect, fill, help

    @property
    def full_id(self):
        return f"{self.module.id}/{self.id}"


class ModuleInfo:
    def __init__(self, folder, source, manifest):
        self.folder, self.source = os.path.abspath(folder), source
        module = manifest.get("module")
        if not isinstance(module, dict):
            raise ManifestError(f"{folder}: module.toml needs a [module] table.")
        for key in ("id", "name", "version", "api"):
            if key not in module:
                raise ManifestError(f"{folder}: module.toml needs {key} in [module].")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-_]*", str(module["id"])):
            raise ManifestError(f"{folder}: id {module['id']!r}: letters, digits, '.', '-' and '_' only.")
        self.id, self.name, self.version = str(module["id"]), str(module["name"]), str(module["version"])
        self.api = module["api"]
        self.description = module.get("description", "")
        self.authors, self.license = module.get("authors", []), module.get("license", "")
        self.requires = list(module.get("requires", []))
        self.platforms = module.get("platforms")
        self.nodes = {}
        for entry in manifest.get("nodes", []):
            for key in ("id", "class"):
                if key not in entry:
                    raise ManifestError(f"{folder}: every [[nodes]] needs {key}.")
            if entry["id"] in self.nodes:
                raise ManifestError(f"{folder}: node id {entry['id']!r} twice.")
            self.nodes[entry["id"]] = NodeInfo(self, entry["id"], entry["class"], entry.get("files", []))
        self.templates = {}
        for entry in manifest.get("templates", []):
            for key in ("id", "name", "file"):
                if key not in entry:
                    raise ManifestError(f"{folder}: every [[templates]] needs {key}.")
            self.templates[entry["id"]] = TemplateInfo(self, entry["id"], entry["name"], entry["file"],
                                                       entry.get("recognize"), entry.get("detect"),
                                                       entry.get("fill"), entry.get("help", ""))
        self.problems = []  # why it cannot be used (api too new, missing packages, wrong platform)
        if not isinstance(self.api, int) or self.api > API_VERSION:
            self.problems.append(f"written for protocol api {self.api}; this Tesselab knows api {API_VERSION}")
        missing = [req for req in self.requires if not requirement_installed(req)]
        if missing:
            self.problems.append("missing Python packages: " + ", ".join(missing)
                                 + f"  (install with: pip install {' '.join(repr(m) for m in missing)})")
        if self.platforms and platform_name() not in self.platforms:
            self.problems.append(f"works on {', '.join(self.platforms)} only")
        self._code_hash = None

    @property
    def available(self) -> bool:
        return not self.problems

    @property
    def package(self) -> str:
        return package_name(self.id)

    @property
    def code_hash(self) -> str:
        if self._code_hash is None:
            self._code_hash = code_hash(self.folder)
        return self._code_hash


def platform_name():
    return {"linux": "linux", "win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)


def package_name(module_id) -> str:
    return f"{PACKAGE_PREFIX}.{re.sub(r'[^A-Za-z0-9_]', '_', module_id)}"


def requirement_installed(requirement: str) -> bool:
    """Whether a requirement such as 'scipy>=1.10' or 'kufner_uv_analysis @ git+...' is installed (by its
    distribution name; versions are not compared)."""
    name = re.split(r"[\s<>=!~@\[;]", requirement.strip(), maxsplit=1)[0]
    for candidate in {name, name.replace("_", "-"), name.replace("-", "_")}:
        try:
            importlib.metadata.distribution(candidate)
            return True
        except importlib.metadata.PackageNotFoundError:
            continue
    return importlib.util.find_spec(name.replace("-", "_")) is not None


def code_files(folder):
    """The files that make a module's code: its .py, .toml and .json files, not the sample data, the tests or what
    check writes. Yields (folder, [files in it]) for the module folder and each subfolder, in a fixed order (code_hash
    reads them, the window watches them)."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in ("sample", "tests", "__pycache__", "check") and not d.startswith("."))
        yield root, [os.path.join(root, name) for name in sorted(files) if name.endswith((".py", ".toml", ".json"))]


def code_hash(folder) -> str:
    """A fingerprint of a module's code (code_files)."""
    digest = hashlib.sha256()
    for _, paths in code_files(folder):
        for path in paths:
            digest.update(os.path.relpath(path, folder).encode())
            with open(path, "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()[:16]


def forget_imports(module_id):
    """Forget what this process imported of a module, so that its files are imported again (a module changed on
    disk). Also forgets the import system's cached folder listings (a file added since)."""
    name = package_name(module_id)
    for old in [n for n in sys.modules if n == name or n.startswith(name + ".")]:
        del sys.modules[old]
    importlib.invalidate_caches()


def read_manifest(folder, source) -> ModuleInfo:
    path = os.path.join(folder, "module.toml")
    with open(path, "rb") as f:
        try:
            manifest = tomllib.load(f)
        except tomllib.TOMLDecodeError as error:
            raise ManifestError(f"{path}: {error}") from None
    return ModuleInfo(folder, source, manifest)


def module_folders(base):
    """Module folders directly in `base` (or base itself if it is one)."""
    if not os.path.isdir(base):
        return []
    if os.path.isfile(os.path.join(base, "module.toml")):
        return [base]
    return [os.path.join(base, name) for name in sorted(os.listdir(base))
            if os.path.isfile(os.path.join(base, name, "module.toml"))]


def _entry_point_folders():
    folders = []
    try:
        points = importlib.metadata.entry_points(group="tesselab.modules")
    except Exception:
        return folders
    for point in points:
        try:
            loaded = point.load()
            folder = loaded() if callable(loaded) else os.path.dirname(loaded.__file__)
            folders.append(folder)
        except Exception as error:  # a broken package must not stop Tesselab
            print(f"tesselab: module entry point {point.name!r} failed: {error}", file=sys.stderr)
    return folders


class Registry:
    """The modules Tesselab knows. project: a project folder (its tesselab/modules are used only if trusted, which
    is never the default: a project someone sent must not run its code unasked); extra: more folders to search (each
    a module folder or a folder of module folders)."""

    def __init__(self, project=None, extra=(), trusted=False, user=True, builtin=True, entry_points=True):
        self.project, self.extra, self.trusted = project, list(extra), trusted
        self.use_user, self.use_builtin, self.use_entry_points = user, builtin, entry_points
        self.modules: dict[str, ModuleInfo] = {}
        self.errors: list[str] = []  # manifests that could not be read
        self.untrusted: list[str] = []  # project module folders not loaded (not trusted)
        self._classes = {}
        self.scan()

    def bases(self):
        """The folders searched for modules, in order, except installed packages: [(folder, source)], each a module
        folder or a folder of module folders. The window watches them (a module added or changed)."""
        found = []
        if self.project and self.trusted:
            found.append((os.path.join(self.project, "tesselab", "modules"), "project"))
        if self.use_user:
            found.append((user_module_folder(), "user"))
        if self.use_builtin:
            found.append((BUILTIN, "builtin"))
        for base in self.extra + [p for p in os.environ.get("TESSELAB_MODULE_PATH", "").split(os.pathsep) if p]:
            found.append((base, "extra"))
        return found

    def sources(self):
        if self.project and not self.trusted:
            self.untrusted = module_folders(os.path.join(self.project, "tesselab", "modules"))
        found = [(f, source) for base, source in self.bases() for f in module_folders(base)]
        if self.use_entry_points:  # after the built-in modules, before the extra folders
            at = next((i for i, (_, source) in enumerate(found) if source == "extra"), len(found))
            found[at:at] = [(f, "installed") for f in _entry_point_folders()]
        return found

    def scan(self):
        self.modules, self.errors = {}, []
        for folder, source in self.sources():
            try:
                info = read_manifest(folder, source)
                info.code_hash  # now, not on first use: refresh() compares it with the code on disk later
            except (ManifestError, OSError) as error:
                self.errors.append(str(error))
                continue
            if info.id not in self.modules:
                self.modules[info.id] = info

    def reload(self):
        """Read the manifests again and forget imported node classes (Reload modules)."""
        for name in [n for n in sys.modules if n == PACKAGE_PREFIX or n.startswith(PACKAGE_PREFIX + ".")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        self._classes = {}
        self.scan()

    def refresh(self) -> list[str]:
        """Read the manifests again and forget the imports of the modules whose code changed (the window follows
        changes on disk); the other modules keep theirs. Returns the ids of the modules that are new, gone or
        changed."""
        old = self.modules
        self.scan()
        changed = sorted(i for i in set(old) | set(self.modules)
                         if i not in old or i not in self.modules or old[i].folder != self.modules[i].folder
                         or old[i].code_hash != self.modules[i].code_hash or old[i].problems != self.modules[i].problems)
        for module_id in changed:
            forget_imports(module_id)
            self._classes = {k: v for k, v in self._classes.items() if k.rpartition("/")[0] != module_id}
        return changed

    def nodes(self):
        """All nodes of the available modules: {full id: NodeInfo}."""
        return {node.full_id: node for module in self.modules.values() if module.available
                for node in module.nodes.values()}

    def node(self, full_id) -> NodeInfo:
        module_id, _, node_id = full_id.rpartition("/")
        module = self.modules.get(module_id)
        if module is None:
            raise KeyError(f"No module {module_id!r} (needed for node {full_id!r}).")
        if node_id not in module.nodes:
            raise KeyError(f"Module {module_id!r} has no node {node_id!r}.")
        return module.nodes[node_id]

    def node_class(self, full_id):
        if full_id not in self._classes:
            info = self.node(full_id)
            if not info.module.available:
                raise RuntimeError(f"Module {info.module.id!r} cannot be used: {'; '.join(info.module.problems)}")
            self._classes[full_id] = load_node_class(info.module.folder, info.module.id, info.class_spec)
        return self._classes[full_id]

    def templates(self):
        return {t.full_id: t for module in self.modules.values() if module.available
                for t in module.templates.values()}

    def template(self, full_id) -> TemplateInfo:
        module_id, _, template_id = full_id.rpartition("/")
        return self.modules[module_id].templates[template_id]

    def function(self, module, spec):
        """A function named in a manifest as 'file.py:function'."""
        return load_attribute(module.folder, module.id, spec)


def _ensure_package(folder, module_id):
    """Import the module folder as the package tesselab_modules.<id> (once per process)."""
    if PACKAGE_PREFIX not in sys.modules:
        root = types.ModuleType(PACKAGE_PREFIX)
        root.__path__ = []
        sys.modules[PACKAGE_PREFIX] = root
    name = package_name(module_id)
    package = sys.modules.get(name)
    if package is None or getattr(package, "__tesselab_folder__", None) != os.path.abspath(folder):
        for old in [n for n in sys.modules if n == name or n.startswith(name + ".")]:
            del sys.modules[old]
        spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [os.path.abspath(folder)]
        package.__tesselab_folder__ = os.path.abspath(folder)
        sys.modules[name] = package
    return name


def load_attribute(folder, module_id, spec):
    """'nodes.py:QuantumYield' (or 'nodes:QuantumYield') from the module folder."""
    file, _, attribute = spec.partition(":")
    if not attribute:
        raise ManifestError(f"{spec!r}: expected 'file.py:Name'.")
    package = _ensure_package(folder, module_id)
    module_name = file[:-3] if file.endswith(".py") else file
    # without writing .pyc files: Python checks them by the source's size and modification time in whole seconds,
    # so a quick edit that keeps the size (5 -> 3) would be imported from the old bytecode
    writing, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        module = importlib.import_module(f"{package}.{module_name.replace('/', '.')}")
    finally:
        sys.dont_write_bytecode = writing
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ManifestError(f"{module_id}: {file} has no {attribute!r}.") from None


def load_node_class(folder, module_id, class_spec):
    cls = load_attribute(folder, module_id, class_spec)
    from tesselab.node import Node
    if not (isinstance(cls, type) and issubclass(cls, Node)):
        raise ManifestError(f"{module_id}: {class_spec} is no subclass of tesselab.Node.")
    return cls


def read_template(module, template) -> dict:
    with open(os.path.join(module.folder, template.file)) as f:
        return json.load(f)
