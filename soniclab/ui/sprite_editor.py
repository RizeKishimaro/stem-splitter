"""Sprite animation editor. A separate window from the arrangement.

The left side browses the sheet, the center previews it, the bottom holds
the frame sequence, and the right side edits timing and beat triggers.
Drum triggers never change the BPM of the base animation.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QByteArray, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from soniclab.core.sprite_project import (
    GRIDS,
    LAYERS,
    LOOP_MODES,
    POSE_CATEGORIES,
    TRIGGERS,
    Animation,
    EditorSession,
    SpriteProject,
    Sync,
    base_frame_id,
    display_source,
    frame_id_at,
    guess_frame_size,
    heart_dancer_project,
    shown_frame_id,
    unique_name,
    validate,
)

MIME = "application/x-soniclab-frames"
_COUNTS = ("1", "e", "&", "a")


def _checker(painter: QPainter, width: int, height: int, cell: int = 8) -> None:
    light, dark = QColor("#1A1E24"), QColor("#12151A")
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            painter.fillRect(x, y, cell, cell, light if ((x // cell) + (y // cell)) % 2 == 0 else dark)


class _FrameList(QListWidget):
    """Sheet frames. Drag them to reorder, or onto the timeline to use them."""

    dropped = Signal(list, int)

    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(72, 72))
        self.setGridSize(QSize(96, 118))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Snap)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSpacing(4)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setAcceptDrops(True)

    def mimeTypes(self):
        return [MIME]

    def mimeData(self, items):
        from PySide6.QtCore import QMimeData

        payload = {"ids": [int(item.data(Qt.ItemDataRole.UserRole)) for item in items], "source": "frames"}
        mime = QMimeData()
        mime.setData(MIME, QByteArray(json.dumps(payload).encode()))
        return mime

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(MIME):
            event.ignore()
            return
        payload = json.loads(bytes(event.mimeData().data(MIME)).decode())
        if payload.get("source") != "frames":
            event.ignore()
            return
        row = self.indexAt(event.position().toPoint()).row()
        if row < 0:
            row = self.count()
        self.dropped.emit([int(frame_id) for frame_id in payload.get("ids", [])], row)
        event.acceptProposedAction()


class _Timeline(QListWidget):
    """Frames of the current animation, in order."""

    dropped = Signal(list, int, str)

    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(56, 56))
        self.setGridSize(QSize(68, 78))
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSpacing(4)
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)

    def mimeTypes(self):
        return [MIME]

    def mimeData(self, items):
        from PySide6.QtCore import QMimeData

        payload = {
            "ids": [int(item.data(Qt.ItemDataRole.UserRole)) for item in items],
            "positions": [int(item.data(Qt.ItemDataRole.UserRole + 1)) for item in items],
            "source": "timeline",
        }
        mime = QMimeData()
        mime.setData(MIME, QByteArray(json.dumps(payload).encode()))
        return mime

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(MIME):
            event.ignore()
            return
        payload = json.loads(bytes(event.mimeData().data(MIME)).decode())
        row = self.indexAt(event.position().toPoint()).row()
        if row < 0:
            row = self.count()
        ids = payload.get("positions") if payload.get("source") == "timeline" else payload.get("ids", [])
        self.dropped.emit([int(value) for value in ids], row, str(payload.get("source", "")))
        event.acceptProposedAction()


class _SheetView(QWidget):
    picked = Signal(int, bool)

    def __init__(self, owner: SpriteEditorWindow):
        super().__init__()
        self.owner = owner
        self.setMinimumSize(220, 160)
        self._fit = (0, 0, 1, 1, 1.0)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0C0D10"))
        if self.owner.show_checker:
            _checker(painter, self.width(), self.height())
        image = self.owner.image()
        project = self.owner.session.project
        if image is None or image.isNull():
            painter.setPen(QColor("#9AA49C"))
            painter.drawText(8, self.height() // 2, "Import a PNG sprite sheet")
            painter.end()
            return
        fitted = _fit(image.width(), image.height(), self.width(), self.height())
        self._fit = fitted
        left, top, width, height, _scale = fitted
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(QRect(left, top, width, height), image)
        sheet = project.sheet
        if self.owner.show_grid and sheet.columns and sheet.rows:
            painter.setPen(QPen(QColor(215, 179, 106, 90), 1))
            cell_w = width / sheet.columns
            cell_h = height / sheet.rows
            for column in range(sheet.columns + 1):
                x = int(left + column * cell_w)
                painter.drawLine(x, top, x, top + height)
            for row in range(sheet.rows + 1):
                y = int(top + row * cell_h)
                painter.drawLine(left, y, left + width, y)
        selected = set(self.owner.selected_frame_ids())
        if self.owner.show_verify and image.width() > 0 and image.height() > 0:
            scale_x = width / image.width()
            scale_y = height / image.height()
            painter.setFont(QFont("DejaVu Sans", 8))
            for item in project.frames:
                rx, ry, rw, rh = project.rect_of(item.id)
                view = QRect(int(left + rx * scale_x), int(top + ry * scale_y), max(1, int(rw * scale_x)), max(1, int(rh * scale_y)))
                painter.setPen(QPen(QColor("#D7B36A") if item.id in selected else QColor(215, 179, 106, 140), 2 if item.id in selected else 1))
                painter.drawRect(view)
                if item.content_rect is not None:
                    cx, cy, cw, ch = item.content_rect
                    painter.setPen(QPen(QColor(120, 190, 170, 160), 1))
                    painter.drawRect(int(left + cx * scale_x), int(top + cy * scale_y), max(1, int(cw * scale_x)), max(1, int(ch * scale_y)))
                painter.setPen(QColor("#F4E2B8"))
                painter.drawText(view.adjusted(2, 1, 0, 0), f"{item.id:03d}")
        else:
            painter.setPen(QPen(QColor("#D7B36A"), 2))
            cell_w = width / max(1, sheet.columns)
            cell_h = height / max(1, sheet.rows)
            for frame_id in selected:
                source = project.source_of(frame_id)
                column = source % max(1, sheet.columns)
                row = source // max(1, sheet.columns)
                if row >= sheet.rows:
                    continue
                painter.drawRect(int(left + column * cell_w), int(top + row * cell_h), int(cell_w), int(cell_h))
        painter.end()

    def mousePressEvent(self, event):
        left, top, width, height, _scale = self._fit
        if width <= 0 or height <= 0:
            return
        x = event.position().x() - left
        y = event.position().y() - top
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        image = self.owner.image()
        extend = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if self.owner.show_verify and image is not None and image.width() > 0:
            ix = int(x / width * image.width())
            iy = int(y / height * image.height())
            hit = self.owner.session.project.frame_at_point(ix, iy)
            if hit is not None:
                self.owner.select_frame(hit.id, extend)
                return
        sheet = self.owner.session.project.sheet
        column = min(sheet.columns - 1, int(x / width * sheet.columns))
        row = min(sheet.rows - 1, int(y / height * sheet.rows))
        self.picked.emit(row * sheet.columns + column, extend)


class _Preview(QWidget):
    def __init__(self, owner: SpriteEditorWindow):
        super().__init__()
        self.owner = owner
        self.setMinimumSize(240, 240)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0C0D10"))
        if self.owner.show_checker:
            _checker(painter, self.width(), self.height(), 12)
        image = self.owner.image()
        frame_id = self.owner.current_frame_id()
        if image is None or image.isNull() or frame_id is None:
            painter.setPen(QColor("#9AA49C"))
            painter.drawText(12, self.height() // 2, "No frame")
            painter.end()
            return
        project = self.owner.session.project
        rect = project.rect_of(frame_id)
        scale = _pixel_scale(rect[2], rect[3], self.width() - 16, self.height() - 16)
        drawn_w = max(1, int(rect[2] * scale))
        drawn_h = max(1, int(rect[3] * scale))
        left = (self.width() - drawn_w) // 2
        top = (self.height() - drawn_h) // 2
        if self.owner.bass_offset():
            left += int(8 * scale)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(
            QRect(left, top, drawn_w, drawn_h),
            image,
            QRect(rect[0], rect[1], rect[2], rect[3]),
        )
        painter.end()


def _fit(image_w: int, image_h: int, width: int, height: int):
    if image_w <= 0 or image_h <= 0 or width <= 0 or height <= 0:
        return 0, 0, 0, 0, 1.0
    scale = min(width / image_w, height / image_h)
    drawn_w = max(1, int(image_w * scale))
    drawn_h = max(1, int(image_h * scale))
    return (width - drawn_w) // 2, (height - drawn_h) // 2, drawn_w, drawn_h, scale


def _pixel_scale(frame_w: int, frame_h: int, width: int, height: int) -> float:
    if frame_w <= 0 or frame_h <= 0:
        return 1.0
    raw = min(width / frame_w, height / frame_h)
    if raw >= 1:
        return float(max(1, int(raw)))
    return raw


class SpriteEditorWindow(QMainWindow):
    """Create and edit beat-reactive sprite animations."""

    send_requested = Signal(dict)

    def __init__(self, project: SpriteProject | None = None, parent=None):
        super().__init__(parent)
        self.session = EditorSession(project or heart_dancer_project())
        self._animation = self.session.project.base_animation().name if self.session.project.base_animation() else ""
        self._image: QImage | None = None
        self._image_key = ""
        self._thumbs: dict[tuple, QPixmap] = {}
        self._seconds = 0.0
        self._playing = False
        self._playback_speed = 1.0
        self._follow = False
        self._song_clock = None
        self._song_events: list[dict] = []
        self._test_events: list[dict] = []
        self._levels: dict = {}
        self._clock_frame = None
        self._solo = "combined"
        self._filling = False
        self._json_dirty = False
        self.show_grid = True
        self.show_verify = False
        self.show_checker = True
        self._beat_lamp_on = False
        self.setWindowTitle("Sprite Animation Editor")
        self.resize(1180, 760)
        self.setMinimumSize(880, 560)
        self._build_actions()
        self._build_body()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._load_image()
        self.refresh()

    def set_song_clock(self, clock) -> None:
        """``clock()`` returns seconds, bpm, playing, events and the dancer frame."""
        self._song_clock = clock

    def selected_frame_ids(self) -> list[int]:
        return [int(item.data(Qt.ItemDataRole.UserRole)) for item in self.frame_list.selectedItems()]

    def selected_timeline_rows(self) -> list[int]:
        return sorted(int(item.data(Qt.ItemDataRole.UserRole + 1)) for item in self.timeline.selectedItems())

    def current_animation(self) -> Animation | None:
        return self.session.project.animations.get(self._animation)

    def effective_bpm(self) -> float:
        if self._follow and self._levels is not None and getattr(self, "_follow_bpm", None):
            return float(self._follow_bpm)
        return float(self.session.project.bpm)

    def current_frame_id(self) -> int | None:
        project = self.session.project
        bpm = self.effective_bpm()
        if self._solo != "combined":
            animation = self._solo_animation(self._solo)
            return None if animation is None else frame_id_at(animation, self._seconds, bpm)
        if self._follow and self._clock_frame is not None:
            source = display_source(project, int(self._clock_frame), self._levels)
            return self._id_for_source(source)
        return shown_frame_id(project, self._seconds, self.active_events(), bpm)

    def base_frame(self) -> int | None:
        project = self.session.project
        if self._follow and self._clock_frame is not None:
            animation = project.base_animation()
            if animation and animation.frames:
                return animation.frames[int(self._clock_frame) % len(animation.frames)]
        return base_frame_id(project, self._seconds, self.effective_bpm())

    def active_events(self) -> list[dict]:
        if self._follow:
            return list(self._song_events)
        cutoff = self._seconds - 2.0
        self._test_events = [event for event in self._test_events if float(event["time"]) >= cutoff]
        return list(self._test_events)

    def bass_offset(self) -> bool:
        if float(self._levels.get("bass", 0.0) or 0.0) >= 0.35:
            return True
        for event in self.active_events():
            if event.get("kind") == "bass" and self._seconds - float(event.get("time", 0.0)) < 0.25:
                return True
        return False

    def image(self) -> QImage | None:
        self._load_image()
        return self._image

    def refresh(self) -> None:
        self._filling = True
        project = self.session.project
        self.setWindowTitle(f"Sprite Animation Editor — {project.name}")
        self._fill_frames()
        self._fill_animations()
        self._fill_timeline()
        self._fill_properties()
        self._fill_json()
        self.bpm_box.setValue(project.bpm)
        self.undo_action.setEnabled(bool(self.session.undo_stack))
        self.redo_action.setEnabled(bool(self.session.redo_stack))
        self.sheet_view.update()
        self.preview.update()
        self._filling = False
        self._update_caption()

    def undo(self) -> None:
        if self.session.undo():
            self._keep_animation()
            self.refresh()

    def redo(self) -> None:
        if self.session.redo():
            self._keep_animation()
            self.refresh()

    def reorder_frames(self, ids: list[int], row: int) -> None:
        if not ids:
            return
        removed = 0
        order = [item.id for item in self.session.project.frames]
        for frame_id in ids:
            if frame_id in order[:row]:
                removed += 1
        dest = max(0, row - removed)

        def apply(project: SpriteProject):
            project.move_frames(ids, dest)

        self.session.change(apply)
        self.refresh()

    def drop_on_timeline(self, values: list[int], row: int, source: str) -> None:
        animation = self._animation
        if not animation:
            return
        if source == "frames":
            def apply(project: SpriteProject):
                project.add_to_animation(animation, values, None if row >= len(project.animations[animation].frames) else row)
            self.session.change(apply)
        elif source == "timeline":
            def apply(project: SpriteProject):
                project.reorder_animation(animation, values, row)
            self.session.change(apply)
        self.refresh()

    def duplicate_selected(self) -> None:
        ids = self.selected_frame_ids()
        if not ids:
            return
        created: list[int] = []

        def apply(project: SpriteProject):
            created.extend(project.duplicate_frames(ids))

        self.session.change(apply)
        self.refresh()
        self._select_frames(created)

    def delete_selected(self) -> None:
        if self.timeline.hasFocus() and self.selected_timeline_rows():
            rows = self.selected_timeline_rows()
            name = self._animation

            def apply(project: SpriteProject):
                project.remove_animation_positions(name, rows)

            self.session.change(apply)
            self.refresh()
            return
        ids = self.selected_frame_ids()
        if not ids:
            return

        def apply(project: SpriteProject):
            project.delete_frames(ids)

        self.session.change(apply)
        self.refresh()

    def copy_selected(self) -> None:
        ids = self.selected_frame_ids()
        if ids:
            self.session.clipboard = list(ids)

    def paste_frames(self) -> None:
        if not self.session.clipboard:
            return
        created: list[int] = []

        def apply(project: SpriteProject):
            created.extend(project.duplicate_frames(self.session.clipboard))

        self.session.change(apply)
        self.refresh()
        self._select_frames(created)

    def rename_selected(self) -> None:
        ids = self.selected_frame_ids()
        if not ids:
            return
        current = self.session.project.frame(ids[0])
        name, ok = QInputDialog.getText(self, "Rename frame", "Name", text=current.name if current else "")
        if not ok:
            return

        def apply(project: SpriteProject):
            project.rename_frame(ids[0], name)

        self.session.change(apply)
        self.refresh()

    def test_trigger(self, kind: str) -> None:
        self._test_events.append({"kind": kind, "time": self._seconds, "confidence": 1.0})
        self.preview.update()
        self._update_caption()

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "&New", self._new_project, QKeySequence.StandardKey.New)
        self._action(file_menu, "&Open…", self._open_project, QKeySequence.StandardKey.Open)
        self._action(file_menu, "&Save", self._save_project, QKeySequence.StandardKey.Save)
        self._action(file_menu, "Save &As…", self._save_project_as, QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        self._action(file_menu, "Import &Sprite Sheet…", self._import_sheet)
        self._action(file_menu, "Import &JSON…", self._import_json)
        self._action(file_menu, "&Export JSON…", self._export_json)
        self._action(file_menu, "Open &Heart Dancer", self._open_heart_dancer)
        self._action(file_menu, "Open &Green Dancer", self._open_green_dancer)
        self._action(file_menu, "Copy JSON", self._copy_json, QKeySequence("Ctrl+Shift+C"))
        file_menu.addSeparator()
        self._action(file_menu, "Send to &Dancer", self._send, QKeySequence("Ctrl+Return"))

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = self._action(edit_menu, "&Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action(edit_menu, "&Redo", self.redo, QKeySequence.StandardKey.Redo)
        edit_menu.addSeparator()
        self._action(edit_menu, "&Duplicate", self.duplicate_selected, QKeySequence("Ctrl+D"))
        self._action(edit_menu, "&Copy", self.copy_selected, QKeySequence.StandardKey.Copy)
        self._action(edit_menu, "&Paste", self.paste_frames, QKeySequence.StandardKey.Paste)
        self._action(edit_menu, "Delete", self.delete_selected, QKeySequence.StandardKey.Delete)
        self._action(edit_menu, "Rename…", self.rename_selected, QKeySequence("F2"))

        view_menu = self.menuBar().addMenu("&View")
        grid = QAction("Sheet Grid", self, checkable=True, checked=True)
        grid.toggled.connect(self._toggle_grid)
        view_menu.addAction(grid)
        checker = QAction("Transparency Grid", self, checkable=True, checked=True)
        checker.toggled.connect(self._toggle_checker)
        view_menu.addAction(checker)
        verify = QAction("Verify Mapping", self, checkable=True, checked=False)
        verify.toggled.connect(self._toggle_verify)
        view_menu.addAction(verify)

    def _build_body(self) -> None:
        toolbar = QToolBar()
        toolbar.setObjectName("spriteEditorBar")
        self.addToolBar(toolbar)
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Preview on the BPM clock. Space")
        self.play_button.clicked.connect(self._toggle_play)
        toolbar.addWidget(self.play_button)
        stop = QPushButton("Stop")
        stop.clicked.connect(self._stop)
        toolbar.addWidget(stop)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("BPM"))
        self.bpm_box = QDoubleSpinBox()
        self.bpm_box.setRange(40, 240)
        self.bpm_box.setDecimals(2)
        self.bpm_box.setToolTip("Base dance speed. Drum hits do not change this.")
        self.bpm_box.editingFinished.connect(self._bpm_edited)
        toolbar.addWidget(self.bpm_box)
        toolbar.addWidget(QLabel("Speed"))
        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.25, 4)
        self.speed_box.setSingleStep(0.25)
        self.speed_box.setValue(1)
        self.speed_box.setToolTip("Preview playback only. It is not saved into the animation.")
        self.speed_box.valueChanged.connect(self._speed_edited)
        toolbar.addWidget(self.speed_box)
        self.follow_box = QCheckBox("Follow song")
        self.follow_box.setToolTip("Preview against the song open in SonicLab, including detected drums.")
        self.follow_box.toggled.connect(self._follow_toggled)
        toolbar.addWidget(self.follow_box)
        self.layer_box = QComboBox()
        self.layer_box.setToolTip("Preview one layer, or the combined dance.")
        self.layer_box.currentIndexChanged.connect(self._solo_changed)
        toolbar.addWidget(self.layer_box)
        self.beat_lamp = QLabel("●")
        self.beat_lamp.setStyleSheet("color: #5E6772; font-size: 16px;")
        self.beat_lamp.setToolTip("Metronome. Amber on each beat of the BPM clock.")
        toolbar.addWidget(self.beat_lamp)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.sheet_view = _SheetView(self)
        self.sheet_view.picked.connect(self._pick_sheet_cell)
        self.frame_list = _FrameList()
        self.frame_list.itemSelectionChanged.connect(self._frame_selection_changed)
        self.frame_list.dropped.connect(self.reorder_frames)
        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(self.sheet_view)
        left_split.addWidget(self.frame_list)
        left_split.setStretchFactor(1, 1)
        left_layout.addWidget(left_split)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 4)
        self.caption = QLabel("Stopped")
        self.caption.setStyleSheet("color: #9AA49C;")
        center_layout.addWidget(self.caption)
        self.preview = _Preview(self)
        center_layout.addWidget(self.preview, 1)
        hits = QHBoxLayout()
        for kind in TRIGGERS:
            button = QPushButton(kind)
            button.setToolTip(f"Fire a {kind} trigger. The base frame number stays on the BPM.")
            button.clicked.connect(lambda _checked=False, kind=kind: self.test_trigger(kind))
            hits.addWidget(button)
        center_layout.addLayout(hits)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFrameShape(QFrame.Shape.NoFrame)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(8, 8, 8, 8)
        self.animation_list = QListWidget()
        self.animation_list.currentTextChanged.connect(self._animation_changed)
        form.addRow(self.animation_list)
        row = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._add_animation)
        vary = QPushButton("Variation")
        vary.clicked.connect(self._variation)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_animation)
        row.addWidget(add)
        row.addWidget(vary)
        row.addWidget(remove)
        form.addRow(row)
        self.stride_box = QDoubleSpinBox()
        self.stride_box.setRange(1, 12)
        self.stride_box.setDecimals(0)
        self.stride_box.setValue(3)
        self.stride_box.setToolTip("Variation keeps every Nth frame. 3 is the pixel-key step of the heart dance.")
        form.addRow("Stride", self.stride_box)
        self.frame_name = self._line(self._rename_from_field)
        form.addRow("Frame", self.frame_name)
        self.frame_meta = QLabel("")
        self.frame_meta.setWordWrap(True)
        self.frame_meta.setStyleSheet("color: #C7C1B3;")
        form.addRow(self.frame_meta)
        self.category_box = self._combo(POSE_CATEGORIES, self._category_changed)
        self.category_box.setToolTip("Pose family. Change this when the catalog is wrong.")
        form.addRow("Category", self.category_box)
        self.rect_x = self._spin(0, 8192, 1, self._rect_changed)
        self.rect_y = self._spin(0, 8192, 1, self._rect_changed)
        self.rect_w = self._spin(1, 8192, 1, self._rect_changed)
        self.rect_h = self._spin(1, 8192, 1, self._rect_changed)
        for box in (self.rect_x, self.rect_y, self.rect_w, self.rect_h):
            box.setDecimals(0)
        rect_row = QHBoxLayout()
        for box in (self.rect_x, self.rect_y, self.rect_w, self.rect_h):
            rect_row.addWidget(box)
        form.addRow("X Y W H", rect_row)
        reset_rect = QPushButton("Reset to cell")
        reset_rect.setToolTip("Put the selected frame back on its grid cell.")
        reset_rect.clicked.connect(self._reset_rect)
        form.addRow(reset_rect)
        self.anim_layer = self._combo(LAYERS, self._layer_changed)
        form.addRow("Layer", self.anim_layer)
        self.base_box = QCheckBox("Base dance")
        self.base_box.setToolTip("The base dance follows BPM and downbeats. Triggers do not move it.")
        self.base_box.toggled.connect(self._base_toggled)
        form.addRow(self.base_box)
        self.loop_box = self._combo(LOOP_MODES, self._loop_changed)
        form.addRow("Loop", self.loop_box)
        self.trigger_box = self._combo(("", *TRIGGERS), self._trigger_changed)
        form.addRow("Trigger", self.trigger_box)
        self.grid_box = self._combo(GRIDS, self._grid_changed)
        form.addRow("Grid", self.grid_box)
        self.cycle_box = self._spin(0.25, 64, 0.25, self._cycle_changed)
        self.cycle_box.setToolTip("Beats in one cycle of this animation.")
        form.addRow("Beats / cycle", self.cycle_box)
        self.fps_box = self._spin(1, 60, 1, self._fps_changed)
        self.fps_box.setToolTip("Sets an even duration for every frame at the project BPM.")
        form.addRow("FPS", self.fps_box)
        self.multiplier_box = self._spin(0.05, 8, 0.05, self._multiplier_changed)
        self.multiplier_box.setToolTip("Scales this animation only. The base dance is unchanged.")
        form.addRow("Speed ×", self.multiplier_box)
        self.intensity_box = self._spin(0, 1, 0.05, self._intensity_changed)
        form.addRow("Intensity", self.intensity_box)
        self.sensitivity_box = self._spin(0, 1, 0.05, self._sensitivity_changed)
        self.sensitivity_box.setToolTip("Higher fires on quieter events. 1 accepts any hit.")
        form.addRow("Sensitivity", self.sensitivity_box)
        self.cooldown_box = self._spin(0, 2, 0.01, self._cooldown_changed)
        self.cooldown_box.setSuffix(" s")
        form.addRow("Cooldown", self.cooldown_box)
        self.duration_box = self._spin(0.01, 8, 0.01, self._duration_changed)
        self.duration_box.setToolTip("Length of the selected timeline frames, in beats.")
        form.addRow("Frame beats", self.duration_box)
        sheet_row = QHBoxLayout()
        self.columns_box = self._spin(1, 64, 1, self._grid_size_changed)
        self.rows_box = self._spin(1, 64, 1, self._grid_size_changed)
        self.columns_box.setDecimals(0)
        self.rows_box.setDecimals(0)
        sheet_row.addWidget(self.columns_box)
        sheet_row.addWidget(self.rows_box)
        form.addRow("Columns × rows", sheet_row)
        self.cell_w = self._spin(1, 2048, 1, self._cell_changed)
        self.cell_h = self._spin(1, 2048, 1, self._cell_changed)
        self.cell_w.setDecimals(0)
        self.cell_h.setDecimals(0)
        form.addRow("Cell width", self.cell_w)
        form.addRow("Cell height", self.cell_h)
        self.space_x = self._spin(0, 128, 1, self._cell_changed)
        self.space_y = self._spin(0, 128, 1, self._cell_changed)
        self.space_x.setDecimals(0)
        self.space_y.setDecimals(0)
        form.addRow("Spacing X", self.space_x)
        form.addRow("Spacing Y", self.space_y)
        right.setWidget(form_host)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.timeline_label = QLabel("Timeline")
        self.timeline_label.setStyleSheet("color: #9AA49C;")
        header.addWidget(self.timeline_label)
        header.addStretch(1)
        use = QPushButton("Add selected frames")
        use.clicked.connect(self._add_selected_to_timeline)
        header.addWidget(use)
        bottom_layout.addLayout(header)
        self.timeline = _Timeline()
        self.timeline.dropped.connect(self.drop_on_timeline)
        self.timeline.itemSelectionChanged.connect(self._timeline_selection_changed)
        bottom_layout.addWidget(self.timeline)

        self.json_edit = QPlainTextEdit()
        self.json_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.json_edit.setReadOnly(True)
        self.json_edit.setStyleSheet("font-family: 'DejaVu Sans Mono', monospace; font-size: 11px;")
        self.json_edit.textChanged.connect(self._json_edited)
        json_bar = QHBoxLayout()
        self.json_edit_box = QCheckBox("Edit JSON")
        self.json_edit_box.toggled.connect(self._json_mode)
        apply_json = QPushButton("Apply")
        apply_json.clicked.connect(self._apply_json)
        check_json = QPushButton("Validate")
        check_json.clicked.connect(self._validate_json)
        copy_json = QPushButton("Copy")
        copy_json.clicked.connect(self._copy_json)
        json_bar.addWidget(self.json_edit_box)
        json_bar.addWidget(apply_json)
        json_bar.addWidget(check_json)
        json_bar.addWidget(copy_json)
        json_bar.addStretch(1)
        json_host = QWidget()
        json_layout = QVBoxLayout(json_host)
        json_layout.setContentsMargins(0, 0, 0, 0)
        json_layout.addLayout(json_bar)
        json_layout.addWidget(self.json_edit)

        center_split = QSplitter(Qt.Orientation.Vertical)
        center_split.addWidget(center)
        center_split.addWidget(bottom)
        center_split.setStretchFactor(0, 3)
        center_split.setStretchFactor(1, 1)
        work = QSplitter(Qt.Orientation.Horizontal)
        work.addWidget(left)
        work.addWidget(center_split)
        work.addWidget(right)
        work.setStretchFactor(1, 1)
        work.setSizes([280, 620, 280])
        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(work)
        outer.addWidget(json_host)
        outer.setStretchFactor(0, 1)
        outer.setSizes([860, 320])
        self.setCentralWidget(outer)
        self.statusBar().showMessage("BPM sets the dance. Triggers add a layer on top of it.")

    def _action(self, menu, label, slot, shortcut=None) -> QAction:
        action = QAction(label, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _line(self, slot):
        from PySide6.QtWidgets import QLineEdit

        edit = QLineEdit()
        edit.editingFinished.connect(slot)
        return edit

    def _combo(self, values, slot) -> QComboBox:
        box = QComboBox()
        for value in values:
            box.addItem(value if value else "(clock)", value)
        box.currentIndexChanged.connect(lambda _index: None if self._filling else slot())
        return box

    def _spin(self, low, high, step, slot) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.editingFinished.connect(lambda: None if self._filling else slot())
        return box

    def _load_image(self) -> None:
        path = self.session.project.resolve_sheet()
        key = str(path)
        if key == self._image_key and self._image is not None:
            return
        self._image_key = key
        self._thumbs.clear()
        if not path.is_file():
            self._image = None
            return
        image = QImage(str(path))
        self._image = None if image.isNull() else image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)

    def _thumb(self, frame_id: int, size: int) -> QPixmap:
        rect = self.session.project.rect_of(frame_id)
        key = (self._image_key, rect, size)
        cached = self._thumbs.get(key)
        if cached is not None:
            return cached
        image = self.image()
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor("#12151A"))
        if image is not None and not image.isNull():
            cropped = image.copy(*rect)
            scaled = cropped.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            painter = QPainter(pixmap)
            _checker(painter, size, size, 4)
            painter.drawImage((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
            painter.end()
        self._thumbs[key] = pixmap
        return pixmap

    def _fill_frames(self) -> None:
        selected = set(self.selected_frame_ids())
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for item in self.session.project.frames:
            rect = self.session.project.rect_of(item.id)
            where = f"{item.id:03d}  {rect[0]},{rect[1]}"
            label = where if item.name in ("", f"{item.id:03d}") else f"{item.name}\n{where}"
            row = QListWidgetItem(self._thumb(item.id, 72), label)
            row.setData(Qt.ItemDataRole.UserRole, item.id)
            tip = f"frame_{item.id:03d}\n{rect[0]}, {rect[1]}, {rect[2]}×{rect[3]}"
            if item.category:
                tip += f"\n{item.category}"
                if item.confidence:
                    tip += f" · {item.confidence}"
            if item.note:
                tip += f"\n{item.note}"
            row.setToolTip(tip)
            self.frame_list.addItem(row)
            if item.id in selected:
                row.setSelected(True)
        self.frame_list.blockSignals(False)

    def _fill_animations(self) -> None:
        project = self.session.project
        names = list(project.animations)
        if self._animation not in project.animations and names:
            self._animation = names[0]
        self.animation_list.blockSignals(True)
        self.animation_list.clear()
        self.animation_list.addItems(names)
        matches = self.animation_list.findItems(self._animation, Qt.MatchFlag.MatchExactly)
        if matches:
            self.animation_list.setCurrentItem(matches[0])
        self.animation_list.blockSignals(False)
        self.layer_box.blockSignals(True)
        current = self.layer_box.currentData()
        self.layer_box.clear()
        self.layer_box.addItem("Combined", "combined")
        for layer in project.layer_ids():
            self.layer_box.addItem(layer.replace("_", " "), layer)
        index = self.layer_box.findData(current or self._solo)
        self.layer_box.setCurrentIndex(max(0, index))
        self.layer_box.blockSignals(False)

    def _fill_timeline(self) -> None:
        animation = self.current_animation()
        self.timeline.blockSignals(True)
        self.timeline.clear()
        if animation is not None:
            for position, frame_id in enumerate(animation.frames):
                item = self.session.project.frame(frame_id)
                label = item.name if item else str(frame_id)
                row = QListWidgetItem(self._thumb(frame_id, 56), label)
                row.setData(Qt.ItemDataRole.UserRole, frame_id)
                row.setData(Qt.ItemDataRole.UserRole + 1, position)
                self.timeline.addItem(row)
            self.timeline_label.setText(f"{animation.name} · {animation.layer} · {len(animation.frames)} frames")
        self.timeline.blockSignals(False)

    def _fill_properties(self) -> None:
        animation = self.current_animation()
        ids = self.selected_frame_ids()
        frame = self.session.project.frame(ids[0]) if ids else None
        self.frame_name.setText(frame.name if frame else "")
        if frame is None:
            self.frame_meta.setText("")
            self._set_combo(self.category_box, "unknown")
        else:
            rect = self.session.project.rect_of(frame.id)
            confidence = f" · {frame.confidence}" if frame.confidence else ""
            self.frame_meta.setText(f"frame_{frame.id:03d}   {rect[0]}, {rect[1]}, {rect[2]}×{rect[3]}{confidence}")
            self._set_combo(self.category_box, frame.category or "unknown")
            self.rect_x.setValue(rect[0])
            self.rect_y.setValue(rect[1])
            self.rect_w.setValue(rect[2])
            self.rect_h.setValue(rect[3])
        sheet = self.session.project.sheet
        self.columns_box.setValue(sheet.columns)
        self.rows_box.setValue(sheet.rows)
        self.cell_w.setValue(sheet.frame_width)
        self.cell_h.setValue(sheet.frame_height)
        self.space_x.setValue(sheet.spacing_x)
        self.space_y.setValue(sheet.spacing_y)
        if animation is None:
            return
        self._set_combo(self.anim_layer, animation.layer)
        self.base_box.setChecked(animation.base)
        self._set_combo(self.loop_box, animation.loop)
        self._set_combo(self.trigger_box, animation.trigger)
        self._set_combo(self.grid_box, animation.sync.grid)
        self.cycle_box.setValue(animation.sync.beats_per_cycle)
        self.fps_box.setValue(animation.fps)
        self.multiplier_box.setValue(animation.sync.speed_multiplier)
        self.intensity_box.setValue(animation.sync.intensity)
        self.sensitivity_box.setValue(animation.sync.sensitivity)
        self.cooldown_box.setValue(animation.sync.cooldown)
        rows = self.selected_timeline_rows()
        if rows and animation.durations:
            self.duration_box.setValue(animation.durations[min(rows[0], len(animation.durations) - 1)])

    def _fill_json(self) -> None:
        if self._json_dirty:
            return
        bar = self.json_edit.verticalScrollBar().value()
        self.json_edit.blockSignals(True)
        self.json_edit.setPlainText(self.session.project.to_json())
        self.json_edit.verticalScrollBar().setValue(bar)
        self.json_edit.blockSignals(False)

    def _set_combo(self, box: QComboBox, value: str) -> None:
        index = box.findData(value)
        if index < 0:
            box.addItem(value, value)
            index = box.findData(value)
        box.setCurrentIndex(max(0, index))

    def _keep_animation(self) -> None:
        if self._animation not in self.session.project.animations:
            animation = self.session.project.base_animation()
            self._animation = animation.name if animation else ""

    def select_frame(self, frame_id: int, extend: bool) -> None:
        if not extend:
            self.frame_list.clearSelection()
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(frame_id):
                item.setSelected(True)
                self.frame_list.scrollToItem(item)
                return

    def _select_frames(self, ids: list[int]) -> None:
        wanted = set(ids)
        self.frame_list.blockSignals(True)
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in wanted)
        self.frame_list.blockSignals(False)

    def _id_for_source(self, source: int) -> int:
        for item in self.session.project.frames:
            if item.source == source:
                return item.id
        return int(source)

    def _solo_animation(self, layer: str) -> Animation | None:
        found = self.session.project.animations_on_layer(layer)
        for animation in found:
            if animation.base or not animation.trigger:
                return animation
        return found[0] if found else None

    def _mutate_animation(self, fn) -> None:
        name = self._animation
        if name not in self.session.project.animations:
            return

        def apply(project: SpriteProject):
            fn(project.animations[name])

        self.session.change(apply)
        self.refresh()

    def _tick(self) -> None:
        if self._follow and self._song_clock is not None:
            try:
                state = self._song_clock() or {}
            except Exception:
                state = {}
            self._seconds = float(state.get("seconds", self._seconds))
            if state.get("bpm"):
                self._follow_bpm = float(state["bpm"])
            self._song_events = list(state.get("events") or [])
            self._levels = dict(state.get("levels") or {})
            self._clock_frame = state.get("clock_frame")
        elif self._playing:
            self._seconds += 0.016 * self._playback_speed
        beat = int(self._seconds * self.effective_bpm() / 60.0)
        on = self._playing or self._follow
        lamp = on and (self._seconds * self.effective_bpm() / 60.0) % 1.0 < 0.12
        if lamp != self._beat_lamp_on:
            self._beat_lamp_on = lamp
            self._beat_index = beat
            self.beat_lamp.setStyleSheet("color: #D7B36A; font-size: 16px;" if lamp else "color: #5E6772; font-size: 16px;")
            if on and lamp:
                kind = "Downbeat" if beat % 4 == 0 else "Beat"
                self.statusBar().showMessage(f"{kind} {beat + 1} {_COUNTS[beat % 4]}", 300)
        self.preview.update()
        self._update_caption()

    def _update_caption(self) -> None:
        base = self.base_frame()
        shown = self.current_frame_id()
        where = "Song" if self._follow else ("Playing" if self._playing else "Stopped")
        events = sorted({str(event.get("kind")) for event in self.active_events() if abs(self._seconds - float(event.get("time", -9))) < 0.15})
        extra = f" · {', '.join(events)}" if events else ""
        detail = ""
        ids = self.selected_frame_ids()
        if ids:
            item = self.session.project.frame(ids[0])
            rect = self.session.project.rect_of(ids[0])
            kind = item.category if item and item.category else "unknown"
            detail = f" · frame_{ids[0]:03d} {rect[0]},{rect[1]} {rect[2]}×{rect[3]} {kind}"
        self.caption.setText(
            f"{where} · {self.effective_bpm():.2f} BPM · base {base} · shown {shown}{extra}{detail}"
        )

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.setText("Pause" if self._playing else "Play")

    def _stop(self) -> None:
        self._playing = False
        self._seconds = 0.0
        self._test_events.clear()
        self.play_button.setText("Play")
        self.preview.update()
        self._update_caption()

    def _toggle_grid(self, on: bool) -> None:
        self.show_grid = bool(on)
        self.sheet_view.update()

    def _toggle_checker(self, on: bool) -> None:
        self.show_checker = bool(on)
        self.sheet_view.update()
        self.preview.update()

    def _toggle_verify(self, on: bool) -> None:
        self.show_verify = bool(on)
        self.sheet_view.update()

    def _bpm_edited(self) -> None:
        if self._filling:
            return
        value = float(self.bpm_box.value())

        def apply(project: SpriteProject):
            project.bpm = value

        self.session.change(apply)
        self.refresh()

    def _speed_edited(self, value: float) -> None:
        self._playback_speed = float(value)

    def _follow_toggled(self, on: bool) -> None:
        self._follow = bool(on)

    def _solo_changed(self) -> None:
        if self._filling:
            return
        self._solo = str(self.layer_box.currentData() or "combined")
        self.preview.update()

    def _frame_selection_changed(self) -> None:
        if self._filling:
            return
        self._filling = True
        self._fill_properties()
        self._filling = False
        self.sheet_view.update()

    def _timeline_selection_changed(self) -> None:
        if self._filling:
            return
        self._filling = True
        animation = self.current_animation()
        rows = self.selected_timeline_rows()
        if animation and rows and animation.durations:
            self.duration_box.setValue(animation.durations[min(rows[0], len(animation.durations) - 1)])
        self._filling = False

    def _animation_changed(self, name: str) -> None:
        if self._filling or not name:
            return
        self._animation = name
        self._filling = True
        self._fill_timeline()
        self._fill_properties()
        self._filling = False

    def _pick_sheet_cell(self, source: int, extend: bool) -> None:
        matches = [
            index
            for index in range(self.frame_list.count())
            if self.session.project.source_of(int(self.frame_list.item(index).data(Qt.ItemDataRole.UserRole))) == source
        ]
        if not matches:
            return
        if not extend:
            self.frame_list.clearSelection()
        self.frame_list.item(matches[0]).setSelected(True)
        self.frame_list.scrollToItem(self.frame_list.item(matches[0]))

    def _add_selected_to_timeline(self) -> None:
        ids = self.selected_frame_ids()
        if not ids or not self._animation:
            return
        name = self._animation

        def apply(project: SpriteProject):
            project.add_to_animation(name, ids)

        self.session.change(apply)
        self.refresh()

    def _add_animation(self) -> None:
        holder = {}

        def apply(project: SpriteProject):
            name = unique_name("animation", project.animations)
            project.animations[name] = Animation(name=name, sync=Sync())
            holder["name"] = name

        self.session.change(apply)
        self._animation = holder.get("name", self._animation)
        self.refresh()

    def _variation(self) -> None:
        if not self._animation:
            return
        source = self._animation
        stride = int(self.stride_box.value())
        holder = {}

        def apply(project: SpriteProject):
            holder["name"] = project.variation(source, f"{source}_var", stride)

        self.session.change(apply)
        self._animation = holder.get("name", self._animation)
        self.refresh()

    def _remove_animation(self) -> None:
        if len(self.session.project.animations) <= 1 or not self._animation:
            return
        name = self._animation

        def apply(project: SpriteProject):
            project.animations.pop(name, None)

        self.session.change(apply)
        self._keep_animation()
        self.refresh()

    def _rename_from_field(self) -> None:
        if self._filling:
            return
        ids = self.selected_frame_ids()
        if not ids:
            return
        name = self.frame_name.text()

        def apply(project: SpriteProject):
            project.rename_frame(ids[0], name)

        self.session.change(apply)
        self.refresh()

    def _layer_changed(self) -> None:
        layer = str(self.anim_layer.currentData() or "body")
        self._mutate_animation(lambda animation: setattr(animation, "layer", layer))

    def _base_toggled(self, on: bool) -> None:
        if self._filling:
            return

        def apply(animation: Animation):
            animation.base = bool(on) and not animation.trigger
            if animation.base:
                animation.trigger = ""

        name = self._animation

        def change(project: SpriteProject):
            for other in project.animations.values():
                if other.name != name:
                    other.base = False
            if name in project.animations:
                apply(project.animations[name])

        self.session.change(change)
        self.refresh()

    def _loop_changed(self) -> None:
        mode = str(self.loop_box.currentData() or "loop")
        self._mutate_animation(lambda animation: setattr(animation, "loop", mode))

    def _trigger_changed(self) -> None:
        trigger = str(self.trigger_box.currentData() or "")

        def apply(animation: Animation):
            animation.trigger = trigger
            animation.sync.type = trigger or "beat"
            if trigger:
                animation.base = False

        self._mutate_animation(apply)

    def _grid_changed(self) -> None:
        grid = str(self.grid_box.currentData() or "beat")
        self._mutate_animation(lambda animation: setattr(animation.sync, "grid", grid))

    def _cycle_changed(self) -> None:
        value = float(self.cycle_box.value())
        self._mutate_animation(lambda animation: setattr(animation.sync, "beats_per_cycle", value))

    def _fps_changed(self) -> None:
        fps = float(self.fps_box.value())
        name = self._animation

        def apply(project: SpriteProject):
            project.apply_fps(name, fps)

        self.session.change(apply)
        self.refresh()

    def _multiplier_changed(self) -> None:
        value = float(self.multiplier_box.value())
        self._mutate_animation(lambda animation: setattr(animation.sync, "speed_multiplier", value))

    def _intensity_changed(self) -> None:
        value = float(self.intensity_box.value())
        self._mutate_animation(lambda animation: setattr(animation.sync, "intensity", value))

    def _sensitivity_changed(self) -> None:
        value = float(self.sensitivity_box.value())
        self._mutate_animation(lambda animation: setattr(animation.sync, "sensitivity", value))

    def _cooldown_changed(self) -> None:
        value = float(self.cooldown_box.value())
        self._mutate_animation(lambda animation: setattr(animation.sync, "cooldown", value))

    def _duration_changed(self) -> None:
        rows = self.selected_timeline_rows()
        if not rows:
            return
        beats = float(self.duration_box.value())
        name = self._animation

        def apply(project: SpriteProject):
            project.set_durations(name, rows, beats)

        self.session.change(apply)
        self.refresh()

    def _grid_size_changed(self) -> None:
        if self._filling:
            return
        columns = int(self.columns_box.value())
        rows = int(self.rows_box.value())

        def apply(project: SpriteProject):
            project.sheet.columns = columns
            project.sheet.rows = rows
            project.fill_cells()

        self.session.change(apply)
        self._thumbs.clear()
        self.refresh()

    def _category_changed(self) -> None:
        if self._filling:
            return
        category = str(self.category_box.currentData() or "unknown")
        ids = self.selected_frame_ids()

        def apply(project: SpriteProject):
            for frame_id in ids:
                item = project.frame(frame_id)
                if item is None or item.category == category:
                    continue
                item.category = category
                item.confidence = "manual"

        self.session.change(apply)
        self.refresh()

    def _rect_changed(self) -> None:
        if self._filling:
            return
        ids = self.selected_frame_ids()
        if not ids:
            return
        rect = (
            int(self.rect_x.value()),
            int(self.rect_y.value()),
            int(self.rect_w.value()),
            int(self.rect_h.value()),
        )
        frame_id = ids[0]

        def apply(project: SpriteProject):
            item = project.frame(frame_id)
            if item is not None:
                item.rect = rect

        self.session.change(apply)
        self._thumbs.clear()
        self.refresh()

    def _reset_rect(self) -> None:
        ids = self.selected_frame_ids()
        if not ids:
            return
        frame_id = ids[0]

        def apply(project: SpriteProject):
            item = project.frame(frame_id)
            if item is not None:
                item.rect = project.sheet.rect(item.source)

        self.session.change(apply)
        self._thumbs.clear()
        self.refresh()

    def _cell_changed(self) -> None:
        if self._filling:
            return
        width = int(self.cell_w.value())
        height = int(self.cell_h.value())
        space_x = int(self.space_x.value())
        space_y = int(self.space_y.value())

        def apply(project: SpriteProject):
            project.sheet.frame_width = width
            project.sheet.frame_height = height
            project.sheet.spacing_x = space_x
            project.sheet.spacing_y = space_y

        self.session.change(apply)
        self._thumbs.clear()
        self.refresh()

    def _json_mode(self, on: bool) -> None:
        self.json_edit.setReadOnly(not on)
        if not on:
            self._json_dirty = False
            self._fill_json()

    def _json_edited(self) -> None:
        if self._filling or self.json_edit.isReadOnly():
            return
        self._json_dirty = True

    def _apply_json(self) -> None:
        try:
            data = json.loads(self.json_edit.toPlainText())
        except json.JSONDecodeError as exc:
            self.statusBar().showMessage(f"JSON: {exc.msg}", 6000)
            return
        errors = validate(data)
        if errors:
            self.statusBar().showMessage(errors[0], 6000)
            return
        root = self.session.project.root
        self.session.undo_stack.append(self.session.project.to_dict())
        self.session.redo_stack.clear()
        self.session.project = SpriteProject.from_dict(data, root=root)
        self._json_dirty = False
        self._keep_animation()
        self._image_key = ""
        self.refresh()
        self.statusBar().showMessage("JSON applied", 3000)

    def _validate_json(self) -> None:
        try:
            data = json.loads(self.json_edit.toPlainText())
        except json.JSONDecodeError as exc:
            self.statusBar().showMessage(f"JSON: {exc.msg}", 6000)
            return
        errors = validate(data)
        self.statusBar().showMessage("JSON is valid" if not errors else errors[0], 6000)

    def _copy_json(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.session.project.to_json())
        self.statusBar().showMessage("JSON copied", 3000)

    def _send(self) -> None:
        data = self.session.project.to_dict()
        data["_root"] = self.session.project.root
        self.send_requested.emit(data)
        self.statusBar().showMessage("Sent to the dancer", 3000)

    def _new_project(self) -> None:
        self.session = EditorSession(SpriteProject(name="character", bpm=120))
        self.session.project.animations["idle"] = Animation(
            name="idle", layer="body", frames=[0], base=True, sync=Sync("beat", "beat", 1, 1, 1, 0.5, 0)
        )
        self.session.project.normalize()
        self._animation = "idle"
        self._image = None
        self._image_key = ""
        self._seconds = 0.0
        self.refresh()

    def _open_heart_dancer(self) -> None:
        self.session = EditorSession(heart_dancer_project())
        self._keep_animation()
        self._image = None
        self._image_key = ""
        self._thumbs.clear()
        self._seconds = 0.0
        self.refresh()
        self.statusBar().showMessage("Heart dancer", 3000)

    def _open_green_dancer(self) -> None:
        from soniclab.core.sprite_catalog import green_dancer_project

        self.session = EditorSession(green_dancer_project())
        self._keep_animation()
        self._image = None
        self._image_key = ""
        self._thumbs.clear()
        self._seconds = 0.0
        self.refresh()
        self.statusBar().showMessage("Green dancer — 80 frames, 8×10, 110×128", 4000)

    def _open_project(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open animation project", "", "Sprite project (*.json)")
        if not path:
            return
        self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            project = SpriteProject.load(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Open animation project", str(exc))
            return
        self.session = EditorSession(project)
        self._keep_animation()
        self._image = None
        self._image_key = ""
        self.refresh()

    def _save_project(self) -> None:
        root = self.session.project.root
        if not root:
            self._save_project_as()
            return
        path = Path(root) / f"{self.session.project.name}.sprite.json"
        self.session.project.save(path)
        self.statusBar().showMessage(f"Saved {path.name}", 3000)

    def _save_project_as(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Save animation project", "character.sprite.json", "Sprite project (*.json)")
        if not path:
            return
        self.session.project.save(Path(path))
        self.statusBar().showMessage(f"Saved {Path(path).name}", 3000)

    def _import_sheet(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import sprite sheet", "", "PNG (*.png)")
        if not path:
            return
        self.import_sheet(Path(path))

    def import_sheet(self, path: Path) -> None:
        image = QImage(str(path))
        if image.isNull():
            self.statusBar().showMessage("That PNG could not be read", 6000)
            return
        if image.width() == 880 and image.height() == 1280:
            width, height, columns, rows = 110, 128, 8, 10
        else:
            width, height, columns, rows = guess_frame_size(image.width(), image.height())

        def apply(project: SpriteProject):
            project.sheet.file = str(path)
            project.sheet.frame_width = width
            project.sheet.frame_height = height
            project.sheet.columns = columns
            project.sheet.rows = rows
            project.sheet.spacing_x = 0
            project.sheet.spacing_y = 0
            project.sheet.image_width = image.width()
            project.sheet.image_height = image.height()
            project.root = str(Path(path).parent)
            project.fill_cells()

        self.session.change(apply)
        self._image = None
        self._image_key = ""
        self.refresh()

    def _import_json(self) -> None:
        self._open_project()

    def _export_json(self) -> None:
        self._save_project_as()

    def _send_payload(self) -> dict:
        data = self.session.project.to_dict()
        data["_root"] = self.session.project.root
        return data
