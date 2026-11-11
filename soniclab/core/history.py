"""Undo stack of already-applied edits.

Commands are pushed after the model has changed, so `push` does not call redo.
"""


class Edit:
    def __init__(self, undo, redo):
        self._undo = undo
        self._redo = redo

    def undo(self) -> None:
        self._undo()

    def redo(self) -> None:
        self._redo()


class History:
    def __init__(self, on_change=None, limit: int = 100):
        self._undo = []
        self._redo = []
        self._limit = int(limit)
        self._on_change = on_change

    def push(self, edit: Edit) -> None:
        self._undo.append(edit)
        del self._undo[: max(0, len(self._undo) - self._limit)]
        self._redo.clear()
        self._notify()

    def undo(self) -> bool:
        if not self._undo:
            return False
        edit = self._undo.pop()
        edit.undo()
        self._redo.append(edit)
        self._notify()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        edit = self._redo.pop()
        edit.redo()
        self._undo.append(edit)
        self._notify()
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()
