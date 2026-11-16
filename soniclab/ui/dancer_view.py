"""The heart dancer. It paints one frame of the supplied sheet.

Which frame is decided by the playback position (see ``soniclab.core.dancer``).
The timer here only asks for that pose again. It does not keep musical time.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.core.dancer import (
    ASSETS,
    BLOCK_PX,
    FRAME_PX,
    SHEET_COLUMNS,
    SHEET_PATH,
    HitDeck,
    Pose,
    pose_at,
)
from soniclab.core.sprite_project import SpriteProject, display_source

# The characters the dancer can be. Heart is the built-in sheet; the
# others are sprite projects in the assets folder, on the same clock.
CHARACTERS = (
    ("Heart", ""),
    ("Green dancer", "green_dancer/green_dancer.sprite.json"),
    ("Walker", "walker/walker.sprite.json"),
)
EDITOR_CHARACTER = "Sprite editor"

_SHEET: QImage | None = None
_SHEET_FAILED = False


def sheet_image() -> QImage | None:
    """The authored sheet, loaded once. None if the file cannot be read."""
    global _SHEET, _SHEET_FAILED
    if _SHEET is not None:
        return _SHEET
    if _SHEET_FAILED or not SHEET_PATH.is_file():
        _SHEET_FAILED = True
        return None
    image = QImage(str(SHEET_PATH))
    if image.isNull():
        _SHEET_FAILED = True
        return None
    _SHEET = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    return _SHEET


class DancerView(QWidget):
    """Enable, pixel or smooth, and the character."""

    enabled_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clock = None  # () -> (seconds, bpm, downbeats, loop, playing)
        self._audio = None  # () -> samples, or (samples, arrangement origin)
        self._enabled = True
        self._mode = "smooth"
        self._pose = Pose(0, 0.0, True)
        self._deck = HitDeck()
        self._playhead = None
        self._listen_after = None
        self._sprite_project = None
        self._project_image = None
        self.setMinimumSize(180, 200)
        self.setAutoFillBackground(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.enable_btn = QToolButton()
        self.enable_btn.setText("Dance")
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(True)
        self.enable_btn.setToolTip("Show the dancer. The pose follows the playhead.")
        self.enable_btn.toggled.connect(self._set_enabled)
        row.addWidget(self.enable_btn)
        self.mode_box = QComboBox()
        self.mode_box.addItem("Smooth", "smooth")
        self.mode_box.addItem("Pixel", "pixel")
        self.mode_box.setToolTip("Smooth plays the in-between frames. Pixel holds each sixteenth.")
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        row.addWidget(self.mode_box)
        self.heart_btn = QToolButton()
        self.heart_btn.setText("Heart")
        self.heart_btn.setToolTip("Show the heart dancer.")
        self.heart_btn.clicked.connect(self.restore_heart)
        row.addWidget(self.heart_btn)
        self.character_box = QComboBox()
        for name, path in CHARACTERS:
            self.character_box.addItem(name, path)
        self.character_box.setToolTip("Which character dances. Every one follows the same beat and downbeats.")
        self.character_box.currentIndexChanged.connect(self._character_changed)
        row.addWidget(self.character_box)
        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(40, 250)
        self.sensitivity.setValue(100)
        self.sensitivity.setFixedWidth(88)
        self.sensitivity.setToolTip(
            "How easily a drum hit adds its own accent. The dance speed stays on the tempo."
        )
        self.sensitivity.valueChanged.connect(self._set_sensitivity)
        row.addWidget(self.sensitivity)
        row.addStretch(1)
        root.addLayout(row)
        self.caption = QLabel("Stopped")
        self.caption.setStyleSheet("color: #9AA49C;")
        root.addWidget(self.caption)
        self.stage = _Stage(self)
        root.addWidget(self.stage, 1)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_clock(self, clock) -> None:
        """``clock()`` returns seconds, bpm, downbeats, loop, playing."""
        self._clock = clock

    def set_audio(self, reader) -> None:
        """Recent output, already captured off the audio callback.

        ``reader()`` may return the samples, or ``(samples, origin)`` where
        origin is the arrangement time of the first sample. The render head
        is ahead of what is heard, so hits wait out the output latency.
        """
        self._audio = reader

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def pose(self) -> Pose:
        return self._pose

    def _set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        self.enabled_changed.emit(self._enabled)
        if self._enabled and not self._timer.isActive():
            self._timer.start()
        self._tick()

    def _set_sensitivity(self, value: int) -> None:
        self._deck.sensitivity = int(value) / 100.0

    def _mode_changed(self) -> None:
        self._mode = str(self.mode_box.currentData() or "smooth")
        self._pose = Pose(-1, -1.0, False)
        self._tick()

    def _tick(self) -> None:
        if not self._enabled or self._clock is None:
            return
        seconds, bpm, downbeats, loop, playing = self._clock()
        seconds = float(seconds)
        if self._playhead is not None:
            delta = seconds - self._playhead
            # A seek or a stop. Pause leaves the playhead where it is.
            # The capture ring still holds the audio from before the jump,
            # so detection waits until that block has played out.
            if delta < -0.02 or delta > 0.35:
                self._deck.reset()
                self._listen_after = seconds + 0.05
        self._playhead = seconds
        if playing and self._audio is not None:
            if self._listen_after is None or seconds >= self._listen_after:
                self._listen_after = None
                self._take_hits(seconds)
        pose = pose_at(
            seconds,
            bpm,
            downbeats,
            loop,
            self._mode,
            hits=self._deck.hits,
            bass=self._deck.bass,
        )
        if pose == self._pose:
            return
        self._pose = pose
        beat = int(pose.beat) + 1
        step = "1 e & a".split()[int((pose.beat % 1.0) * 4)]
        where = "Playing" if playing else "Stopped"
        extra = f" · {pose.layers.label}" if pose.layers.label else ""
        self.caption.setText(f"{where} · beat {beat} {step}{extra}")
        self.stage.update()

    def _take_hits(self, seconds: float) -> None:
        raw = self._audio()
        origin = None
        audio = raw
        if isinstance(raw, tuple):
            audio, origin = raw[0], raw[1]
        if audio is None or not getattr(audio, "size", 0):
            return
        count = int(audio.shape[0])
        if origin is None:
            origin = seconds - count / OUTPUT_RATE
        self._deck.update(audio, OUTPUT_RATE, float(origin))

    def restore_heart(self) -> None:
        """Drop an editor project and paint the heart dancer again."""
        self._sprite_project = None
        self._project_image = None
        self._pose = Pose(-1, -1.0, False)
        self._show_character("Heart")
        self._tick()

    def apply_sprite_project(self, data: dict | None) -> None:
        """Use an editor project for the painted frame. The clock is unchanged."""
        if not data:
            self._sprite_project = None
            self._project_image = None
            self._show_character("Heart")
            return
        root = str(data.get("_root", "") or "")
        self._use_project(SpriteProject.from_dict(data, root=root))
        self._show_character(EDITOR_CHARACTER)

    @property
    def character(self) -> str:
        return self.character_box.currentText()

    def select_character(self, name: str) -> bool:
        """Dance as one of ``CHARACTERS``. Returns False for an unknown name."""
        index = self.character_box.findText(name)
        if index < 0:
            return False
        self.character_box.setCurrentIndex(index)
        return True

    def _character_changed(self, index: int) -> None:
        path = str(self.character_box.itemData(index) or "")
        name = self.character_box.itemText(index)
        if name == EDITOR_CHARACTER:
            return
        if not path:
            self.restore_heart()
            return
        try:
            project = SpriteProject.load(ASSETS / path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.caption.setText(f"Could not load {name}: {exc}")
            return
        self._use_project(project)

    def _use_project(self, project: SpriteProject) -> None:
        self._sprite_project = project
        self._project_image = None
        self._pose = Pose(-1, -1.0, False)
        self._tick()
        self.stage.update()

    def _show_character(self, name: str) -> None:
        """Show ``name`` in the picker without loading it again."""
        index = self.character_box.findText(name)
        if index < 0:
            self.character_box.addItem(name, "")
            index = self.character_box.count() - 1
        self.character_box.blockSignals(True)
        self.character_box.setCurrentIndex(index)
        self.character_box.blockSignals(False)

    def live_events(self) -> list[dict]:
        return [
            {"kind": hit.kind, "time": float(hit.time), "confidence": float(hit.confidence)}
            for hit in self._deck.hits
        ]

    def _project_sheet_image(self) -> QImage | None:
        project = self._sprite_project
        if project is None:
            return None
        if self._project_image is not None:
            return self._project_image
        path = project.resolve_sheet()
        if not path.is_file():
            return None
        image = QImage(str(path))
        if image.isNull():
            return None
        self._project_image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        return self._project_image

    def paint_stage(self, painter: QPainter, width: int, height: int) -> None:
        painter.fillRect(0, 0, width, height, QColor("#0C0D10"))
        if not self._enabled:
            return
        image = sheet_image()
        if image is None or width < 8 or height < 8:
            painter.setPen(QColor("#9AA49C"))
            painter.drawText(8, height // 2, "Dancer sheet missing")
            return
        pose = self._pose
        frame = pose.draw_frame
        source_w = FRAME_PX
        source_h = FRAME_PX
        project_image = self._project_sheet_image()
        if project_image is not None:
            image = project_image
            source = display_source(
                self._sprite_project,
                pose.frame,
                {
                    "kick": pose.layers.kick,
                    "snare": pose.layers.snare,
                    "hat": pose.layers.hat,
                    "perc": pose.layers.perc,
                },
            )
            source_x, source_y, source_w, source_h = self._sprite_project.rect_of_source(source)
        else:
            column = frame % SHEET_COLUMNS
            row = frame // SHEET_COLUMNS
            source_x = column * FRAME_PX
            source_y = row * FRAME_PX
        if source_x < 0 or source_y < 0:
            return
        if source_x + source_w > image.width() or source_y + source_h > image.height():
            return
        side = min(width, height)
        drawn = max(16, side)
        if source_w > 0 and source_h > source_w:
            drawn_h = drawn
            drawn_w = max(1, int(round(drawn * source_w / source_h)))
        elif source_h > 0 and source_w > source_h:
            drawn_w = drawn
            drawn_h = max(1, int(round(drawn * source_h / source_w)))
        else:
            drawn_w = drawn_h = drawn
        # Accents shift the current pose. They do not change its size.
        sway = int(round(pose.sway_blocks * drawn * BLOCK_PX / FRAME_PX))
        drop = int(round(pose.drop_blocks * drawn * BLOCK_PX / FRAME_PX))
        bob = int(round(pose.hat_blocks * drawn * BLOCK_PX / FRAME_PX))
        left = (width - drawn_w) // 2 + sway
        top = (height - drawn_h) // 2 + drop - bob
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(
            QRect(left, top, drawn_w, drawn_h),
            image,
            QRect(source_x, source_y, source_w, source_h),
        )


class _Stage(QWidget):
    def __init__(self, owner: DancerView):
        super().__init__()
        self.owner = owner
        self.setMinimumHeight(160)

    def paintEvent(self, _event):
        painter = QPainter(self)
        self.owner.paint_stage(painter, self.width(), self.height())
        painter.end()
