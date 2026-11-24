"""Every action of every menu: the window's menus, each card's ⋯ menu, each node's ⋯ menu. Modal dialogs and popups
that open are rejected by a timer, so nothing blocks; Quit is left out."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios_common import *  # noqa

if __name__ == "__main__":
    from PySide6 import QtCore, QtWidgets

    proj = sys.argv[1]
    session = os.path.join(proj, "tesselab", "sessions", "two experiments.tesselab-session")
    app, window = setup(session)
    settle(app, 1)
    seen_modal = []

    def kill_modals():
        for w in (QtWidgets.QApplication.activeModalWidget(), QtWidgets.QApplication.activePopupWidget()):
            if w is not None:
                seen_modal.append(type(w).__name__ + ":" + (w.windowTitle() or ""))
                if isinstance(w, QtWidgets.QDialog):
                    w.reject()
                else:
                    w.close()
    killer = QtCore.QTimer(interval=150, timeout=kill_modals)
    killer.start()

    def actions_of(menu, prefix=""):
        found = []
        for action in menu.actions():
            if action.menu() is not None:
                found += actions_of(action.menu(), prefix + action.text() + " › ")
            elif not action.isSeparator() and action.text():
                found.append((prefix + action.text(), action))
        return found

    skip = ("quit", "exit")
    window_actions = []
    for top in window.menus.actions():
        if top.menu() is not None:
            top.menu().aboutToShow.emit()  # menus filled when shown (themes, recent)
            window_actions += actions_of(top.menu(), top.text() + " › ")
    print(len(window_actions), "window actions")
    ANSWER["value"] = 1  # "Discard"/"Cancel" where asked
    for name, action in window_actions:
        if any(s in name.lower() for s in skip) or not action.isEnabled():
            continue
        before = len(errors)
        print("trigger", name, flush=True)
        try:
            action.trigger()
        except Exception as error:
            errors.append(f"{name}: {error}")
        settle(app, 0.4)
        if len(errors) > before:
            print("  -> problem in", name)
        if not list(window.cards()):  # an action closed everything: open the session again
            window.open_session(session)
            settle(app, 1.5)
    # card and node ⋯ menus
    ANSWER["value"] = 0  # "Remove" etc.
    import shiboken6
    for card in list(window.cards()):
        n_buttons = len(card.findChildren(QtWidgets.QToolButton))
        for b in range(n_buttons):
            if card not in list(window.cards()):
                break
            buttons = card.findChildren(QtWidgets.QToolButton)  # fresh: an action may rebuild the groups
            if b >= len(buttons) or buttons[b].menu() is None:
                continue
            buttons[b].menu().aboutToShow.emit()
            for i in range(len(actions_of(buttons[b].menu()))):
                buttons = card.findChildren(QtWidgets.QToolButton)
                if card not in list(window.cards()) or b >= len(buttons) or buttons[b].menu() is None:
                    break
                listed = actions_of(buttons[b].menu())
                if i >= len(listed):
                    break
                name, action = listed[i]
                if not shiboken6.isValid(action) or not action.isEnabled():
                    continue
                print("card action", card.analysis.label, name, flush=True)
                steps = len(card.undo_steps)
                try:
                    action.trigger()
                except Exception as error:
                    errors.append(f"card {name}: {error}")
                settle(app, 0.3)
                if card in list(window.cards()) and len(card.undo_steps) > steps:
                    card.undo()
                    settle(app, 0.2)
    print("modal widgets rejected:", sorted(set(seen_modal)))
    killer.stop()
    finish(window)
