"""
The command line:

    tesselab [path]                       the window (a project folder, an analysis file or a saved window)
    tesselab check <module folder>        check a module on its sample data (see check.py)
    tesselab new module <folder>          a new, working minimal module to start from
    tesselab export <analysis.json> [-o out.pdf]   PDF + JSON of an analysis, without a window
    tesselab init <folder>                make a folder a project
    tesselab modules                      list the modules Tesselab finds
    tesselab --help | --version

--modules <folder> (repeatable, before the command) adds folders to look for modules in.
"""
import argparse
import os
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    while "--modules" in argv:
        i = argv.index("--modules")
        extra.append(os.path.expanduser(argv[i + 1]))
        del argv[i:i + 2]
    command = argv[0] if argv else None
    if command in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if command in ("-V", "--version"):
        from tesselab import __version__
        print(f"tesselab {__version__}")
        return 0
    if command and command.startswith("-"):  # not a path: an option nobody knows (it would open the window)
        print(f"tesselab: unknown option {command} (tesselab --help lists the commands)", file=sys.stderr)
        return 2
    if command == "check":
        parser = argparse.ArgumentParser(prog="tesselab check")
        parser.add_argument("folder")
        parser.add_argument("--no-pictures", action="store_true", help="skip rendering the plots")
        args = parser.parse_args(argv[1:])
        from tesselab.check import check_module
        report = check_module(args.folder, pictures=not args.no_pictures)
        print(report.text())
        return 0 if report.passed else 1
    if command == "new":
        parser = argparse.ArgumentParser(prog="tesselab new")
        parser.add_argument("kind", choices=["module"])
        parser.add_argument("folder")
        parser.add_argument("--id", default=None, help="the module id (default: from the folder name)")
        args = parser.parse_args(argv[1:])
        from tesselab.scaffold import new_module
        try:
            path = new_module(args.folder, args.id)
        except (OSError, ValueError) as error:
            print(f"tesselab new: {error}", file=sys.stderr)
            return 1
        print(f"A new module is in {path}. Next: edit it (see its AGENTS.md), then: tesselab check {path}")
        return 0
    if command == "export":
        parser = argparse.ArgumentParser(prog="tesselab export")
        parser.add_argument("analysis")
        parser.add_argument("-o", "--output", default=None)
        args = parser.parse_args(argv[1:])
        from tesselab.analysis import AnalysisError
        from tesselab.export import export_file
        try:
            pdf = export_file(args.analysis, args.output, extra_modules=extra)
        except (OSError, AnalysisError) as error:
            print(f"tesselab export: {error}", file=sys.stderr)
            return 1
        print(f"Wrote {pdf} and its JSON")
        return 0
    if command == "init":
        from tesselab.project import Project
        project = Project.create(argv[1] if len(argv) > 1 else ".")
        print(f"{project.root} is a Tesselab project now.")
        return 0
    if command == "modules":
        from tesselab.modules import Registry
        from tesselab.project import find_root, trust_state
        root = find_root(os.getcwd())
        registry = Registry(root, extra=extra, trusted=bool(root) and trust_state(root) == "trusted")
        for module in sorted(registry.modules.values(), key=lambda m: m.name):
            state = "ok" if module.available else "; ".join(module.problems)
            print(f"{module.id} {module.version} ({module.source}, {state})\n    {module.folder}")
            for node in module.nodes.values():
                print(f"    node {node.full_id}")
            for template in module.templates.values():
                print(f"    template {template.full_id}: {template.name}")
        for error in registry.errors:
            print(f"(not read) {error}")
        for folder in registry.untrusted:
            print(f"(project not trusted, not used) {folder}")
        return 0
    from tesselab.gui.app import run
    return run(command, extra)


if __name__ == "__main__":
    sys.exit(main())
