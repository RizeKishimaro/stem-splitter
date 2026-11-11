import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget


class MediaBrowser(QWidget):
    """Session contents and recent projects. Library shelves are marked unavailable."""

    track_activated = Signal(int)
    recent_activated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setAnimated(False)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(self._opened)

    def set_session(self, project_name: str, project_path: str | None, tracks, recent):
        self.tree.blockSignals(True)
        self.tree.clear()
        session = self._section("Session")
        if project_path:
            title = f"{project_name}  —  {os.path.basename(project_path)}"
        else:
            title = f"{project_name}  —  unsaved"
        holder = QTreeWidgetItem([title])
        holder.setFlags(Qt.ItemFlag.ItemIsEnabled)
        holder.setToolTip(0, project_path or "This session has not been saved")
        session.addChild(holder)
        if not tracks:
            empty = QTreeWidgetItem(["No tracks"])
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            holder.addChild(empty)
        for index, track in enumerate(tracks):
            item = QTreeWidgetItem([f"{index + 1:02d}   {track.get('name', 'Track')}"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("track", index))
            path = track.get("path")
            duration = float(track.get("duration", 0.0))
            start = float(track.get("start", 0.0))
            if path and not os.path.exists(path):
                item.setText(0, item.text(0) + "   missing")
            item.setToolTip(
                0,
                f"Start {start:.3f} s\nDuration {duration:.3f} s\n{path or 'No source file'}",
            )
            holder.addChild(item)

        imported = self._section("Imported audio")
        sources = [
            (index, track.get("path"))
            for index, track in enumerate(tracks)
            if track.get("path")
        ]
        if not sources:
            none = QTreeWidgetItem(["Nothing imported"])
            none.setFlags(Qt.ItemFlag.NoItemFlags)
            imported.addChild(none)
        for index, path in sources:
            name = os.path.basename(path)
            if not os.path.exists(path):
                name += "   missing"
            item = QTreeWidgetItem([name])
            item.setData(0, Qt.ItemDataRole.UserRole, ("track", index))
            item.setToolTip(0, path)
            imported.addChild(item)

        recent_root = self._section("Recent")
        if not recent:
            none = QTreeWidgetItem(["No recent projects"])
            none.setFlags(Qt.ItemFlag.NoItemFlags)
            recent_root.addChild(none)
        for path in recent:
            label = os.path.basename(path)
            if not os.path.exists(path):
                label += "   missing"
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, ("recent", path))
            item.setToolTip(0, path)
            recent_root.addChild(item)

        shelves = []
        for title in ("Library", "Samples"):
            section = self._section(title)
            note = QTreeWidgetItem(["Not available yet"])
            note.setFlags(Qt.ItemFlag.NoItemFlags)
            note.setToolTip(0, f"The {title.lower()} is not available yet")
            section.addChild(note)
            shelves.append(section)

        self.tree.addTopLevelItem(session)
        self.tree.addTopLevelItem(imported)
        self.tree.addTopLevelItem(recent_root)
        for section in shelves:
            self.tree.addTopLevelItem(section)
        self.tree.expandAll()
        self.tree.blockSignals(False)

    def _section(self, text: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = QFont(item.font(0))
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QColor("#8B939F"))
        return item

    def _clicked(self, item, _column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and data and data[0] == "track":
            self.track_activated.emit(int(data[1]))

    def _opened(self, item, _column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and data and data[0] == "recent":
            self.recent_activated.emit(str(data[1]))
