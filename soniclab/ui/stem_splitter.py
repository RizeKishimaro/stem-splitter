"""Stem Splitter panel: split one song into stems with Demucs.

The panel is a dock beside the mixer. It never runs Demucs itself: the
split runs in the ``StemJobQueue`` worker, which starts Demucs as a separate
process, and this widget only polls for progress on a timer. Reading the
song for its waveform and asking the Demucs environment what it has also
happen on worker threads.

Demucs always writes every stem of the chosen model. The check boxes choose
which of them are exported or added to the project.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from soniclab.core.preview import PreviewPlayer
from soniclab.core.stems import (
    DEVICE_AUTO,
    DEVICE_LABELS,
    MODELS,
    SIX_STEM_MODEL,
    STATUS_CANCELLED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    EngineInfo,
    StemJobQueue,
    default_stem_folder,
    existing_stems,
    export_stems,
    model_stems,
    probe_engine,
    track_order,
)
from soniclab.core.waveform import display_envelope, peak_envelope

AUDIO_FILTER = "Audio (*.wav *.flac *.ogg *.aiff *.aif *.mp3)"
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
SETUP_GUIDE = "docs/stem_splitter.md"
_ENVELOPE_BUCKETS = 4000


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class WaveformPreview(QWidget):
    """The source song drawn like a clip on the timeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lows = np.zeros(0, dtype=np.float32)
        self._highs = np.zeros(0, dtype=np.float32)
        self._title = ""
        self._prompt = "Drop a song here, or choose a file"
        self.setMinimumHeight(56)

    def set_wave(self, lows, highs, title: str) -> None:
        self._lows, self._highs, self._title = lows, highs, title
        self.update()

    def set_prompt(self, text: str) -> None:
        """Empty the view and show ``text`` in it."""
        self._lows = self._highs = np.zeros(0, dtype=np.float32)
        self._title = ""
        self._prompt = text
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        if self._lows.size == 0:
            painter.fillRect(self.rect(), QColor("#12151A"))
            pen = QPen(QColor("#2A3038"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.setPen(QColor("#8B939F"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._prompt)
            painter.end()
            return
        # Same colours as a clip on the timeline.
        painter.fillRect(self.rect(), QColor("#1C382C"))
        painter.setPen(QPen(QColor("#3E6A52"), 1))
        painter.drawRect(rect)
        top = 15
        height = max(4, rect.height() - top - 2)
        middle = top + height / 2.0
        painter.setPen(QColor(46, 78, 62, 160))
        painter.drawLine(1, int(middle), rect.width() - 1, int(middle))
        lows, highs = display_envelope(self._lows, self._highs, max(1, rect.width() - 2))
        painter.setPen(QColor("#8FCB9B"))
        half = height / 2.0 - 1.0
        step = (rect.width() - 2) / max(1, lows.size)
        for index in range(lows.size):
            x = 1 + int(index * step)
            high = middle - float(np.clip(highs[index], -1.0, 1.0)) * half
            low = middle - float(np.clip(lows[index], -1.0, 1.0)) * half
            painter.drawLine(x, int(high), x, int(max(low, high + 1)))
        painter.setPen(QColor("#E7F2EA"))
        metrics = QFontMetrics(painter.font())
        painter.drawText(6, 12, metrics.elidedText(self._title, Qt.TextElideMode.ElideMiddle, rect.width() - 12))
        painter.end()


class StemSplitterPanel(QWidget):
    """Choose a song, split it, audition the stems, export or add them."""

    # song name, {stem: path} of the ticked stems, stems in track order
    add_requested = Signal(str, dict, tuple)
    message = Signal(str)

    def __init__(self, jobs: StemJobQueue | None = None, player: PreviewPlayer | None = None, parent=None):
        super().__init__(parent)
        self.jobs = jobs if jobs is not None else StemJobQueue()
        self.player = player if player is not None else PreviewPlayer()
        # Set by the main window: where stems go by default, and whether
        # another split (split on import) is already running.
        self.default_folder = lambda: default_stem_folder(None)
        self.busy_elsewhere = lambda: False

        self._source: Path | None = None
        self._loading: Path | None = None
        self._folder: Path | None = None  # None: use default_folder()
        self._stems: dict[str, str] = {}
        self._job: str | None = None
        self._started = 0.0
        self._engine: EngineInfo | None = None
        self._probing = False
        self._results: queue.Queue = queue.Queue()
        self._rows: dict[str, tuple[QCheckBox, QToolButton]] = {}

        self._build()
        self.setAcceptDrops(True)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.poll)
        self._restore_settings()
        self._rebuild_stem_rows()
        self._sync()

    # ------------------------------------------------------------------ build

    def _button(self, text: str, tip: str, slot, name: str = "panelBtn") -> QToolButton:
        button = QToolButton()
        button.setObjectName(name)
        button.setText(text)
        button.setToolTip(tip)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.clicked.connect(slot)
        return button

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("sectionLabel")
        return label

    @staticmethod
    def _dim(text: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("dim")
        return label

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(18)

        # --- song
        song = QVBoxLayout()
        song.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(self._section("Song"))
        head.addStretch(1)
        self.play_source = self._button("▶", "Play the song", lambda: self.preview(None), "playStemBtn")
        self.play_source.setCheckable(True)
        head.addWidget(self.play_source)
        self.choose_button = self._button("Choose File…", "Choose the song to split", self.choose_file)
        head.addWidget(self.choose_button)
        song.addLayout(head)
        self.wave = WaveformPreview()
        song.addWidget(self.wave, 1)
        self.source_info = self._dim("No song chosen")
        song.addWidget(self.source_info)
        root.addLayout(song, 5)

        # --- settings and progress
        settings = QVBoxLayout()
        settings.setSpacing(6)
        settings.addWidget(self._section("Split"))
        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        self.model_box = QComboBox()
        for name, model in MODELS.items():
            self.model_box.addItem(model.label, name)
        self.model_box.setToolTip("Which Demucs model to use")
        self.model_box.currentIndexChanged.connect(self._model_changed)
        self.device_box = QComboBox()
        self.device_box.setToolTip(
            "Where Demucs runs. Auto uses an NVIDIA GPU when there is one, otherwise the CPU."
        )
        self.device_box.currentIndexChanged.connect(self._save_settings)
        self._set_devices((DEVICE_AUTO, "cpu"))
        self.folder_label = self._dim()
        self.folder_label.setMinimumWidth(120)
        change = self._button("Change…", "Choose where the stems are saved", self.choose_folder)
        self.reset_folder = self._button("Default", "Save stems in the default folder again", self.use_default_folder)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(change)
        folder_row.addWidget(self.reset_folder)
        form.addWidget(self._dim("Model"), 0, 0)
        form.addWidget(self.model_box, 0, 1)
        form.addWidget(self._dim("Device"), 1, 0)
        form.addWidget(self.device_box, 1, 1)
        form.addWidget(self._dim("Save to"), 2, 0)
        form.addLayout(folder_row, 2, 1)
        form.setColumnStretch(1, 1)
        settings.addLayout(form)
        self.engine_label = self._dim()
        self.engine_label.setWordWrap(True)
        settings.addWidget(self.engine_label)
        settings.addStretch(1)
        run = QHBoxLayout()
        run.setSpacing(8)
        self.split_button = self._button("Split", "Split the song into stems", self.start, "primaryBtn")
        self.cancel_button = self._button("Cancel", "Stop splitting", self.cancel)
        self.progress = QProgressBar()
        self.progress.setObjectName("stemProgress")
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        run.addWidget(self.split_button)
        run.addWidget(self.cancel_button)
        run.addWidget(self.progress, 1)
        settings.addLayout(run)
        self.status_label = self._dim()
        self.status_label.setWordWrap(True)
        settings.addWidget(self.status_label)
        root.addLayout(settings, 5)

        # --- stems
        stems = QVBoxLayout()
        stems.setSpacing(6)
        stems.addWidget(self._section("Stems"))
        self.stem_grid = QGridLayout()
        self.stem_grid.setHorizontalSpacing(14)
        self.stem_grid.setVerticalSpacing(4)
        stems.addLayout(self.stem_grid)
        self.model_note = self._dim()
        self.model_note.setWordWrap(True)
        stems.addWidget(self.model_note)
        stems.addStretch(1)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.export_selected_button = self._button(
            "Export Selected…", "Copy the ticked stems to a folder", self.export_selected
        )
        self.export_all_button = self._button("Export All…", "Copy every stem to a folder", self.export_all)
        self.add_button = self._button(
            "Add to Project", "Add the ticked stems to the project as tracks", self.add_selected, "primaryBtn"
        )
        actions.addWidget(self.export_selected_button)
        actions.addWidget(self.export_all_button)
        actions.addStretch(1)
        actions.addWidget(self.add_button)
        stems.addLayout(actions)
        root.addLayout(stems, 5)

    def _rebuild_stem_rows(self) -> None:
        while self.stem_grid.count():
            item = self.stem_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows.clear()
        names = model_stems(self.model)
        rows = (len(names) + 1) // 2
        for index, stem in enumerate(names):
            box = QCheckBox(stem.capitalize())
            box.setChecked(True)
            box.toggled.connect(self._sync)
            play = self._button("▶", f"Play the {stem} stem", lambda _=False, name=stem: self.preview(name), "playStemBtn")
            play.setCheckable(True)
            column = (index // rows) * 2
            self.stem_grid.addWidget(play, index % rows, column)
            self.stem_grid.addWidget(box, index % rows, column + 1)
            self._rows[stem] = (box, play)
        self.stem_grid.setColumnStretch(1, 1)
        self.stem_grid.setColumnStretch(3, 1)
        self.model_note.setText(MODELS[self.model].note if self.model in MODELS else "")

    # --------------------------------------------------------------- settings

    @property
    def model(self) -> str:
        return self.model_box.currentData() or SIX_STEM_MODEL

    @property
    def device(self) -> str:
        return self.device_box.currentData() or DEVICE_AUTO

    @property
    def folder(self) -> Path:
        return self._folder if self._folder is not None else Path(self.default_folder())

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def stems(self) -> dict[str, str]:
        """Finished stems of the current song and model: {stem: path}."""
        return dict(self._stems)

    @property
    def splitting(self) -> bool:
        return self._job is not None

    def selected_stems(self) -> dict[str, str]:
        return {stem: path for stem, path in self._stems.items() if stem in self._rows and self._rows[stem][0].isChecked()}

    def _set_devices(self, devices) -> None:
        wanted = self.device_box.currentData() or DEVICE_AUTO
        self.device_box.blockSignals(True)
        self.device_box.clear()
        for device in devices:
            self.device_box.addItem(DEVICE_LABELS.get(device, device), device)
        index = self.device_box.findData(wanted)
        self.device_box.setCurrentIndex(max(0, index))
        self.device_box.blockSignals(False)

    def _persist(self) -> bool:
        # Tests run offscreen and must not touch the user's settings.
        return os.environ.get("QT_QPA_PLATFORM") != "offscreen"

    def _restore_settings(self) -> None:
        model = SIX_STEM_MODEL
        if self._persist():
            settings = QSettings("SonicLab", "SonicLab")
            model = str(settings.value("stemSplitter/model", SIX_STEM_MODEL))
            folder = settings.value("stemSplitter/folder", "")
            self._folder = Path(str(folder)) if folder else None
            self._wanted_device = str(settings.value("stemSplitter/device", DEVICE_AUTO))
        else:
            self._wanted_device = DEVICE_AUTO
        index = self.model_box.findData(model)
        self.model_box.blockSignals(True)
        self.model_box.setCurrentIndex(max(0, index))
        self.model_box.blockSignals(False)

    def _save_settings(self, *_args) -> None:
        if not self._persist():
            return
        settings = QSettings("SonicLab", "SonicLab")
        settings.setValue("stemSplitter/model", self.model)
        settings.setValue("stemSplitter/device", self.device)
        settings.setValue("stemSplitter/folder", str(self._folder) if self._folder is not None else "")

    def _model_changed(self, *_args) -> None:
        self.stop_preview()
        self._rebuild_stem_rows()
        self._stems = {}
        self._save_settings()
        self._find_saved_stems()
        self._sync()

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Save stems to", str(self.folder))
        if chosen:
            self.set_folder(chosen)

    def set_folder(self, folder) -> None:
        self._folder = Path(folder)
        self._save_settings()
        self._stems = {}
        self._find_saved_stems()
        self._sync()

    def use_default_folder(self) -> None:
        self._folder = None
        self._save_settings()
        self._stems = {}
        self._find_saved_stems()
        self._sync()

    # ----------------------------------------------------------------- source

    def choose_file(self) -> None:
        start = str(self._source.parent) if self._source is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, "Choose a song to split", start, AUDIO_FILTER)
        if path:
            self.set_source(path)

    def set_source(self, path) -> None:
        """Load ``path`` as the song to split. Reading it runs on a thread."""
        if self.splitting:
            self._status("Cancel the running split before choosing another song.")
            return
        source = Path(path).expanduser().resolve()
        self.stop_preview()
        self._source = source
        self._loading = source
        self._stems = {}
        self.wave.set_prompt(f"Reading {source.name}…")
        self.source_info.setText(str(source))
        self._status("")
        folder, model = self.folder, self.model
        threading.Thread(target=self._load, args=(source, folder, model), name="soniclab-stem-source", daemon=True).start()
        self._timer.start()
        self._sync()

    def _load(self, source: Path, folder: Path, model: str) -> None:
        try:
            info = sf.info(str(source))
            audio, _rate = sf.read(str(source), dtype="float32", always_2d=True)
            lows, highs = peak_envelope(audio, _ENVELOPE_BUCKETS)
            saved = existing_stems(source, folder, model)
            self._results.put(("loaded", source, lows, highs, info.duration, info.samplerate, info.channels, saved, model))
        except Exception as exc:
            self._results.put(("load_failed", source, str(exc)))

    def _find_saved_stems(self) -> None:
        """Stems split earlier for this song and model, without Demucs."""
        if self._source is None or self.splitting:
            return
        try:
            saved = existing_stems(self._source, self.folder, self.model)
        except OSError:
            saved = None
        if saved is not None:
            self._stems = saved
            self._status("Saved stems found for this song.")
        else:
            self._status("")

    # ------------------------------------------------------------------ engine

    def showEvent(self, event):
        super().showEvent(event)
        self.check_engine()

    def check_engine(self, force: bool = False) -> None:
        """Ask the Demucs environment what it has, once, on a thread."""
        if self._probing or (self._engine is not None and not force):
            return
        self._probing = True
        python = self.jobs.python

        def probe():
            self._results.put(("engine", probe_engine(python)))

        threading.Thread(target=probe, name="soniclab-stem-probe", daemon=True).start()
        self._timer.start()
        self._sync()

    def _engine_text(self) -> str:
        if self._probing:
            return "Checking the separation engine…"
        info = self._engine
        if info is None:
            return ""
        if not info.available:
            return f"{info.message}. Setup: {SETUP_GUIDE}"
        text = f"Demucs {info.demucs_version} · PyTorch {info.torch_version}"
        downloaded = info.downloaded.get(self.model)
        if downloaded is True:
            text += " · model downloaded"
        elif downloaded is False:
            text += " · the model downloads on the first split"
        return text

    # ------------------------------------------------------------------- split

    def start(self) -> None:
        if self._source is None or self.splitting:
            return
        if self.busy_elsewhere():
            self._status("Another song is being split. Wait for it to finish.")
            return
        self.stop_preview()
        self._stems = {}
        self._started = time.monotonic()
        self._job = self.jobs.submit(self._source, self.folder, model=self.model, device=self.device)
        self.progress.setValue(0)
        self._status("Starting Demucs…")
        self._timer.start()
        self._sync()

    def cancel(self) -> None:
        if not self.splitting:
            return
        self.jobs.cancel()
        self._status("Cancelling…")

    def poll(self) -> None:
        """Take results from the worker threads. Runs on the UI timer."""
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            self._handle_result(item)
        for _ in range(32):
            item = self.jobs.poll()
            if item is None:
                break
            kind, job = item[0], item[1]
            if job != self._job:
                continue
            if kind == "progress":
                fraction = max(0.0, min(1.0, float(item[2])))
                self.progress.setValue(int(round(fraction * 1000)))
                self._status(f"Splitting… {int(fraction * 100)}%   {format_time(time.monotonic() - self._started)}")
            elif kind == "finished":
                self._finished(item[2])
        if self.player.playing is False and any(play.isChecked() for _box, play in self._rows.values()):
            self._sync()
        if self.player.playing is False and self.play_source.isChecked():
            self._sync()
        if not self.splitting and self._loading is None and not self._probing and not self.player.playing:
            self._timer.stop()

    def _handle_result(self, item) -> None:
        kind = item[0]
        if kind == "engine":
            self._probing = False
            self._engine = item[1]
            self._set_devices(self._engine.devices)
            index = self.device_box.findData(self._wanted_device)
            if index >= 0:
                self.device_box.setCurrentIndex(index)
        elif kind == "loaded" and item[1] == self._loading:
            _kind, source, lows, highs, duration, rate, channels, saved, model = item
            self._loading = None
            self.wave.set_wave(lows, highs, source.name)
            layout = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels} ch"
            self.source_info.setText(f"{format_time(duration)}   {rate / 1000:.1f} kHz   {layout}")
            self.source_info.setToolTip(str(source))
            if saved is not None and model == self.model:
                self._stems = saved
                self._status("Saved stems found for this song.")
        elif kind == "load_failed" and item[1] == self._loading:
            self._loading = None
            self._source = None
            self.wave.set_prompt("Drop a song here, or choose a file")
            self.source_info.setText(f"Could not read {item[1].name}: {item[2]}")
        self._sync()

    def _finished(self, result) -> None:
        self._job = None
        elapsed = format_time(time.monotonic() - self._started)
        if result.status == STATUS_OK:
            self._stems = dict(result.stems)
            self.progress.setValue(1000)
            if result.cached:
                text = "Saved stems found for this song."
            else:
                text = f"Split into {len(result.stems)} stems in {elapsed}."
            if result.message:
                text += f" {result.message}."
            self._status(text)
            self.message.emit(f"Split “{self._name()}” into {len(result.stems)} stems")
            if self._engine is not None and self._engine.downloaded.get(result.model) is False:
                self.check_engine(force=True)  # the weights are downloaded now
        elif result.status == STATUS_CANCELLED:
            self.progress.setValue(0)
            self._status("Split cancelled. Nothing was saved.")
        elif result.status == STATUS_UNAVAILABLE:
            self.progress.setValue(0)
            self._status(f"{result.message or 'Demucs is not installed'}. Setup: {SETUP_GUIDE}")
        else:
            self.progress.setValue(0)
            self._status(f"Could not split: {result.message or 'Demucs failed'}")
        self._sync()

    # ----------------------------------------------------------------- preview

    def preview(self, stem: str | None) -> None:
        """Play a stem, or the song for None. A second click stops it."""
        path = self._stems.get(stem) if stem is not None else (str(self._source) if self._source else None)
        if path is None:
            self._sync()
            return
        if self.player.path == path:
            self.player.stop()
        else:
            error = self.player.play(path)
            if error:
                self._status(error)
            else:
                self._timer.start()
        self._sync()

    def stop_preview(self) -> None:
        if self.player.playing:
            self.player.stop()
            self._sync()

    # ------------------------------------------------------------------ export

    def _name(self) -> str:
        return self._source.stem if self._source is not None else "song"

    def export_selected(self) -> None:
        self._export(self.selected_stems())

    def export_all(self) -> None:
        self._export(self._stems)

    def _export(self, stems: dict) -> None:
        if not stems:
            return
        chosen = QFileDialog.getExistingDirectory(self, "Export stems to", str(self._source.parent))
        if chosen:
            self.export_to(chosen, stems)

    def export_to(self, folder, stems: dict | None = None) -> list[Path]:
        """Copy ``stems`` (default: the ticked ones) into ``folder``."""
        stems = self.selected_stems() if stems is None else stems
        try:
            written = export_stems(stems, folder, self._name())
        except OSError as exc:
            self._status(f"Could not export: {exc}")
            return []
        self._status(f"Exported {len(written)} stem{'s' if len(written) != 1 else ''} to {folder}")
        return written

    def add_selected(self) -> None:
        stems = self.selected_stems()
        if not stems:
            return
        order = tuple(name for name in track_order(model_stems(self.model)) if name in stems)
        self.stop_preview()
        self.add_requested.emit(self._name(), stems, order)

    # -------------------------------------------------------------------- sync

    def _status(self, text: str) -> None:
        self.status_label.setText(text)

    def _sync(self, *_args) -> None:
        """Enable, tick and label everything from the current state."""
        has_source = self._source is not None and self._loading is None
        ready = bool(self._stems)
        self.split_button.setEnabled(has_source and not self.splitting)
        self.cancel_button.setEnabled(self.splitting)
        self.choose_button.setEnabled(not self.splitting)
        self.model_box.setEnabled(not self.splitting)
        self.device_box.setEnabled(not self.splitting)
        self.play_source.setEnabled(has_source)
        playing = self.player.path
        self.play_source.setChecked(has_source and playing == str(self._source))
        self.play_source.setText("■" if self.play_source.isChecked() else "▶")
        for stem, (box, play) in self._rows.items():
            path = self._stems.get(stem)
            box.setEnabled(path is not None)
            play.setEnabled(path is not None)
            play.setChecked(path is not None and playing == path)
            play.setText("■" if play.isChecked() else "▶")
        selected = bool(self.selected_stems())
        self.export_selected_button.setEnabled(selected and not self.splitting)
        self.export_all_button.setEnabled(ready and not self.splitting)
        self.add_button.setEnabled(selected and not self.splitting)
        folder = str(self.folder)
        metrics = QFontMetrics(self.folder_label.font())
        self.folder_label.setText(metrics.elidedText(folder, Qt.TextElideMode.ElideMiddle, max(120, self.folder_label.width())))
        self.folder_label.setToolTip(folder)
        self.reset_folder.setEnabled(self._folder is not None and not self.splitting)
        self.engine_label.setText(self._engine_text())
        if not self.splitting and not ready:
            self.progress.setValue(0)

    # ------------------------------------------------------------- drag & drop

    @staticmethod
    def _dropped_song(event) -> str | None:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if url.isLocalFile() and os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS:
                return path
        return None

    def dragEnterEvent(self, event):
        if self._dropped_song(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_song(event)
        if path is not None:
            event.acceptProposedAction()
            self.set_source(path)

    def shutdown(self) -> None:
        """Stop the preview and any running split. Called when the app closes."""
        self._timer.stop()
        self.player.stop()
        if self.splitting:
            self.jobs.cancel()
