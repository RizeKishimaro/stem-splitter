import os
import tempfile
import time
import uuid

import numpy as np
import soundfile as sf
from PySide6.QtCore import QEvent, QObject, QSettings, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QFont, QKeySequence
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from soniclab.core.alignment import classify_onsets, matched_errors, nudged_alignment
from soniclab.core.tempo import stretch as stretch_audio
from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.audio_loader import AudioLoader, Loaded, decode as decode_audio
from soniclab.core.downbeat_loop import downbeat_loop_region, loop_shortcut
from soniclab.core.bar_export import (
    FORMATS,
    FORMATS_BY_KEY,
    bar_frames,
    default_name,
    write_audio,
)
from soniclab.core.bar_map import (
    bar_index,
    bar_span,
    bars_from_downbeats,
    format_clock,
    insert_line,
    nearest_line,
    parse_clock,
    position as bar_position,
    remove_line,
    set_span,
)
from soniclab.core.drums import pick_drum_track
from soniclab.core.stems import (
    STATUS_CANCELLED as STEMS_CANCELLED,
    STATUS_OK as STEMS_OK,
    STEMS,
    StemJobQueue,
    default_stem_folder,
    existing_stems,
    is_stem_name,
)
from soniclab.core.downbeat_transfer import (
    DownbeatFileError,
    FILE_SUFFIX,
    default_folder,
    export_document,
    import_events,
    read_document,
    track_events,
    transfer_events,
    write_document,
)
from soniclab.core.onset_jobs import OnsetJobQueue
from soniclab.core.beat_analysis import (
    STATUS_CANCELLED,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_UNAVAILABLE,
)
from soniclab.core.beat_jobs import BeatJobQueue
from soniclab.core.history import Edit, History
from soniclab.core.project import Project
from soniclab.core.recorder import Recorder, input_devices
from soniclab.ui.browser import MediaBrowser
from soniclab.ui.headers import HeaderColumn
from soniclab.ui.icons import (
    loop_icon,
    metronome_icon,
    pause_icon,
    play_icon,
    record_icon,
    return_icon,
    stop_icon,
)
from soniclab.ui.inspector import Inspector
from soniclab.ui.mixer import Mixer
from soniclab.ui.stem_splitter import StemSplitterPanel
from soniclab.ui.theme import DARK_STYLE, build_palette
from soniclab.ui.visualizer import Visualizer
from soniclab.ui import ui_trace
from soniclab.ui.timeline import (
    FOLLOW_CENTER,
    FOLLOW_MODES,
    FOLLOW_OFF,
    FOLLOW_PAGE,
    HEADER_WIDTH,
    PIXELS_PER_SECOND,
    RULER_HEIGHT,
    TRACK_HEIGHT,
    ClipItem,
    TimelineView,
    parse_time_signature,
    stem_group,
    stem_loop,
    track_canvas,
)

# The playhead redraws on its own timer, about 60 times a second. Meters,
# labels and background jobs stay on the slower transport timer.
PLAYHEAD_FRAME_MS = 16
FOLLOW_LABELS = {FOLLOW_OFF: "Off", FOLLOW_PAGE: "Follow", FOLLOW_CENTER: "Center"}
# A stem of a split song plays on its split's shared clock. A stem in the
# split's synced loop is one voice keyed (track index, STEM_VOICE); each
# other stem block is its own voice, keyed (track index, stem_block(bar)),
# so one stem can loop several of its bars at once. Negative = a stem.
STEM_VOICE = -1
STEM_BLOCK = -2


def stem_block(bar: int) -> int:
    """The voice key part of a stem's bar, and back (it is its own inverse)."""
    return STEM_BLOCK - int(bar)

# The shortest synced loop a marker drag may leave, in seconds.
MIN_STEM_LOOP = 0.05


def loop_group(group: str) -> str:
    """The engine clock of a split's synced loop. Its stems loop together;
    the split's other stems keep the split's own clock."""
    return f"{group}/loop"
FOLLOW_TIPS = {
    FOLLOW_OFF: "The canvas stays put; the playhead moves across it",
    FOLLOW_PAGE: "Scroll smoothly when the playhead reaches the right edge",
    FOLLOW_CENTER: "Keep the playhead in the middle; the canvas moves behind it",
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


ARROW_KEYS = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)
SEEK_STEP_SECONDS = 5.0
VOLUME_STEP = 0.10
# Same ceiling as the master fader.
MASTER_GAIN_MAX = 1.5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SonicLab")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 640)
        # Drag events reach the window when a child does not take drops.
        self.setAcceptDrops(True)
        self.setFont(QFont("DejaVu Sans", 10))
        app = QApplication.instance()
        if app is not None:
            app.setOrganizationName("SonicLab")
            app.setApplicationName("SonicLab")
            app.setStyle("Fusion")
            app.setPalette(build_palette())
            app.setStyleSheet(DARK_STYLE)
            app.installEventFilter(self)
        self.engine = AudioEngine()
        self.project = Project()
        self.history = History(self._sync_history)
        self.recorder = Recorder()
        self.beat_jobs = BeatJobQueue()
        self.onset_jobs = OnsetJobQueue()
        self.stem_jobs = StemJobQueue()
        self.loader = AudioLoader()
        self._stem_names: dict[str, str] = {}
        self._stem_sources: dict[str, str] = {}
        self.current_path = None
        self._selected = -1
        # None: both canvases are heard. 0 or 1: only that canvas (its ▶ button).
        self._canvas_focus: int | None = None
        self._sized_rows = -1  # second-canvas rows when it was last sized
        # Where each canvas's playhead rests. A canvas's playhead moves only
        # while that canvas is heard, or (stopped) when it was the one clicked.
        self._canvas_positions = [0.0, 0.0]
        self._active_canvas = 0
        self._hover_canvas = 0  # the canvas the mouse was over last; Play plays it
        self._track_focus: int | None = None  # a bar's ▶: hear only this track
        # Stem blocks playing on their own, apart from the transport:
        # (track, bar) -> the track they belong to, to notice edits.
        self._preview_tracks: dict = {}
        self._lane_ids = 1
        self._canvases_rebuilding = False
        self._gesture_before = None
        self._history_frozen = False
        self._record_started_at = 0.0
        self._armed_at_record = []
        self.input_device = None
        self.input_channel = 0
        self.input_stereo = False
        self.input_channels = 1
        self._last_bpm = 120.0
        self._last_signature = "4/4"
        self._capture_shown = None
        self._playing_shown = None
        self._scroll_lock = False
        self._loop_bounds = None
        self.follow_mode = FOLLOW_OFF
        self._last_frame = 0.0
        self._frames = 0
        self._build_arrangement()
        self._build_transport()
        self._build_docks()
        self._build_menus()
        self._connect()
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._transport_tick)
        self.timer.start()
        self.frame_timer = QTimer(self)
        self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_timer.setInterval(PLAYHEAD_FRAME_MS)
        self.frame_timer.timeout.connect(self._animate_playheads)
        self._refresh_timeline()
        self._set_session_title()
        self._sync_transport_lamp()
        self._sync_capture()
        self._restore_view()
        ui_trace.install(self)
        # Canvas 2 is part of every session: open on each launch, whatever
        # the last session ended with. The action shows the one instance.
        self.second_canvas_action.setChecked(True)
        self._canvases_settled = False
        self.statusBar().showMessage("Import audio to begin")
        self._build_stem_progress()
        rate = QLabel("44.1 kHz    Stereo    Snap beat")
        rate.setObjectName("dim")
        self.statusBar().addPermanentWidget(rate)

    def _build_stem_progress(self):
        """Banner above the timeline shown while Demucs splits a song."""
        self.stem_banner = QFrame()
        self.stem_banner.setObjectName("stemBanner")
        row = QHBoxLayout(self.stem_banner)
        row.setContentsMargins(12, 6, 8, 6)
        row.setSpacing(12)
        self.stem_label = QLabel()
        self.stem_label.setObjectName("stemLabel")
        # Fixed widths: a new percentage or song name never re-lays the bar.
        self.stem_label.setFixedWidth(360)
        self.stem_bar = QProgressBar()
        self.stem_bar.setObjectName("stemProgress")
        self.stem_bar.setRange(0, 1000)
        self.stem_bar.setTextVisible(False)
        self.stem_bar.setFixedHeight(8)
        self.stem_bar.setMinimumWidth(220)
        self.stem_cancel = self._text_button(
            "Cancel", "Stop splitting stems", self._cancel_stems
        )
        row.addWidget(self.stem_label)
        row.addWidget(self.stem_bar, 1)
        row.addWidget(self.stem_cancel)
        self.stem_bar.setFixedWidth(220)
        # Its place in the status bar is kept while hidden, so showing it
        # never slides the status bar's other contents.
        policy = self.stem_banner.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.stem_banner.setSizePolicy(policy)
        self.stem_banner.setVisible(False)
        # In the status bar, not above the canvases: showing and hiding it
        # must not push the canvases down and back up.
        row.setContentsMargins(8, 0, 4, 0)
        self.stem_banner.setMaximumHeight(self.statusBar().sizeHint().height())
        self.statusBar().addPermanentWidget(self.stem_banner)

    def _canvas_pair(self, canvas: int):
        headers = HeaderColumn(title="Tracks" if canvas == 0 else "Canvas 2")
        timeline = TimelineView(canvas=canvas)
        timeline.setMinimumWidth(320)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.addWidget(headers)
        splitter.addWidget(timeline)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([228, 1100])
        return headers, timeline, splitter

    def _build_arrangement(self):
        # Two canvases, one above the other. Both are part of the same song:
        # every track plays, whichever canvas it is drawn on. The second
        # canvas is hidden until it is used or turned on (View menu).
        self.headers, self.timeline, self.splitter = self._canvas_pair(0)
        self.headers2, self.timeline2, self.splitter2 = self._canvas_pair(1)
        self.timeline2.setMinimumHeight(140)
        self.timeline2.empty_text = (
            "Second canvas. Drag a track here, or Ctrl-drag (⌘-drag) a bar to put it here. "
            "Everything on it plays with the song."
        )
        self.timeline.peer = self.timeline2
        self.timeline2.peer = self.timeline
        self.canvas_split = QSplitter(Qt.Orientation.Vertical)
        self.canvas_split.setChildrenCollapsible(False)
        self.canvas_split.setHandleWidth(3)
        self.canvas_split.addWidget(self.splitter)
        self.canvas_split.addWidget(self.splitter2)
        self.canvas_split.setStretchFactor(0, 3)
        self.canvas_split.setStretchFactor(1, 2)
        self.splitter2.setVisible(False)
        # Stacked, the canvases share one time axis: they scroll together so
        # bars line up. Side by side, each scrolls on its own, so two parts of
        # the song can be seen at once. Zoom is always shared.
        self.canvas_side_by_side = False
        self.splitter.splitterMoved.connect(lambda *_: self._match_header_widths(self.splitter, self.splitter2))
        self.splitter2.splitterMoved.connect(lambda *_: self._match_header_widths(self.splitter2, self.splitter))
        self.timeline.horizontalScrollBar().valueChanged.connect(
            lambda value: self._follow_scroll(self.timeline2, value)
        )
        self.timeline2.horizontalScrollBar().valueChanged.connect(
            lambda value: self._follow_scroll(self.timeline, value)
        )
        arrangement = QWidget()
        self.arrangement_layout = QVBoxLayout(arrangement)
        self.arrangement_layout.setContentsMargins(0, 0, 0, 0)
        self.arrangement_layout.setSpacing(0)
        self.arrangement_layout.addWidget(self.canvas_split, 1)
        self.setCentralWidget(arrangement)

    def _build_transport(self):
        bar = QToolBar("Transport")
        bar.setObjectName("transportBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

        word = QLabel("SonicLab")
        word.setObjectName("wordmark")
        bar.addWidget(word)

        self.play_btn = self._tool_button(play_icon(), "Play the canvas under the mouse (Space)", self._play_hovered, "playBtn")
        self.pause_btn = self._tool_button(pause_icon(), "Pause", self._pause)
        self.stop_btn = self._tool_button(stop_icon(), "Stop", self._stop)
        self.return_btn = self._tool_button(
            return_icon(), "Return to start", self._return_to_start
        )
        for button in (self.play_btn, self.pause_btn, self.stop_btn, self.return_btn):
            bar.addWidget(button)

        self.record_btn = self._tool_button(
            record_icon("#A65A5A"), "Record", self._toggle_record, "recordBtn"
        )
        bar.addWidget(self.record_btn)
        self.loop_btn = self._tool_button(
            loop_icon(), "Loop the downbeat region", self._toggle_loop, "loopBtn"
        )
        self.loop_btn.setCheckable(True)
        bar.addWidget(self.loop_btn)
        bar.addWidget(
            self._future_button(metronome_icon(), "The metronome is not available yet")
        )

        self.bpm = QDoubleSpinBox()
        self.bpm.setRange(30, 300)
        self.bpm.setDecimals(2)
        self.bpm.setValue(120)
        self.bpm.setSuffix(" BPM")
        self.bpm.setFixedWidth(108)
        self.bpm.setToolTip("Tempo")
        self.signature = QComboBox()
        self.signature.addItems(["4/4", "3/4", "6/8", "5/4", "7/8"])
        self.signature.setFixedWidth(68)
        self.signature.setToolTip("Time signature")
        self.snap_btn = QToolButton()
        self.snap_btn.setText("Snap")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setChecked(True)
        self.snap_btn.setToolTip("Snap clips to the beat")
        self.snap_btn.toggled.connect(self._set_snap)
        self.follow_btn = QToolButton()
        self.follow_btn.setObjectName("followBtn")
        self.follow_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.follow_btn.clicked.connect(self._cycle_follow_mode)
        bar.addWidget(self.bpm)
        bar.addWidget(self.signature)
        bar.addWidget(self.snap_btn)
        bar.addWidget(self.follow_btn)

        readout = QFrame()
        readout.setObjectName("lcd")
        readout_layout = QHBoxLayout(readout)
        readout_layout.setContentsMargins(10, 2, 10, 2)
        readout_layout.setSpacing(12)
        self.position_label = QLabel("00:00.000")
        self.position_label.setObjectName("position")
        self.position_label.setMinimumWidth(92)
        self.musical_label = QLabel("001.1")
        self.musical_label.setObjectName("musical")
        self.musical_label.setMinimumWidth(44)
        readout_layout.addWidget(self.position_label)
        readout_layout.addWidget(self.musical_label)
        bar.addWidget(readout)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        zoom_out = self._text_button(
            "−", "Zoom out", lambda: self._each_timeline(lambda view: view.zoom_horizontal(1 / 1.1))
        )
        zoom_in = self._text_button(
            "+", "Zoom in", lambda: self._each_timeline(lambda view: view.zoom_horizontal(1.1))
        )
        lanes_out = self._text_button(
            "▴", "Shorter tracks", lambda: self._each_timeline(lambda view: view.zoom_vertical(1 / 1.08))
        )
        lanes_in = self._text_button(
            "▾", "Taller tracks", lambda: self._each_timeline(lambda view: view.zoom_vertical(1.08))
        )
        for button in (zoom_out, zoom_in, lanes_out, lanes_in):
            bar.addWidget(button)

        self.import_btn = self._text_button(
            "Import", "Import audio", self._import_audio
        )
        self.add_track_btn = self._text_button(
            "Add Track", "Add an empty track", self._add_empty_track
        )
        bar.addWidget(self.import_btn)
        bar.addWidget(self.add_track_btn)

    def _tool_button(self, icon, tip, slot, name="iconBtn") -> QToolButton:
        button = QToolButton()
        button.setObjectName(name)
        button.setIcon(icon)
        button.setToolTip(tip)
        button.setStatusTip(tip)
        button.setAutoRaise(True)
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(28, 26)
        button.clicked.connect(slot)
        return button

    def _future_button(self, icon, message) -> QToolButton:
        return self._tool_button(
            icon, message, lambda: self._unavailable(message), "futureBtn"
        )

    def _text_button(self, text, tip, slot) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.clicked.connect(slot)
        return button

    def _build_docks(self):
        self.browser = MediaBrowser()
        self.inspector = Inspector()
        self.mixer = Mixer()
        features = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.browser_dock = self._dock("Browser", self.browser, features)
        self.inspector_dock = self._dock("Inspector", self.inspector, features)
        self.mixer_dock = self._dock("Mixer", self.mixer, features)
        self.visualizer = Visualizer()
        self.visualizer_dock = self._dock("Visualizer", self.visualizer, features)
        self.visualizer_dock.setObjectName("visualizerDock")
        self.stem_splitter = StemSplitterPanel()
        self.stem_splitter.default_folder = lambda: default_stem_folder(
            self.current_path
        )
        self.stem_splitter.busy_elsewhere = lambda: self.stem_jobs.pending() > 0
        self.stem_splitter.add_requested.connect(self._add_split_stems)
        self.stem_splitter.message.connect(
            lambda text: self.statusBar().showMessage(text, 8000)
        )
        self.stem_splitter_dock = self._dock(
            "Stem Splitter", self.stem_splitter, features
        )
        self.stem_splitter_dock.setObjectName("stemSplitterDock")
        self.browser_dock.setMinimumWidth(200)
        self.inspector_dock.setMinimumWidth(240)
        self.mixer_dock.setMinimumHeight(200)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.browser_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.mixer_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.visualizer_dock)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.stem_splitter_dock
        )
        self.tabifyDockWidget(self.mixer_dock, self.visualizer_dock)
        self.tabifyDockWidget(self.visualizer_dock, self.stem_splitter_dock)
        self.mixer_dock.raise_()
        self.visualizer_dock.visibilityChanged.connect(self._sync_capture)
        self.resizeDocks(
            [self.browser_dock, self.inspector_dock],
            [240, 260],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.mixer_dock, self.visualizer_dock], [236, 220], Qt.Orientation.Vertical
        )

    def _dock(self, title, widget, features) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title.lower() + "Dock")
        dock.setWidget(widget)
        dock.setFeatures(features)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        return dock

    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        self._action(
            file_menu, "&New Session", self._new_project, QKeySequence.StandardKey.New
        )
        self._action(
            file_menu,
            "&Open Project…",
            self._open_project,
            QKeySequence.StandardKey.Open,
        )
        recent = file_menu.addMenu("Open Recent")
        recent.aboutToShow.connect(lambda: self._fill_recent_menu(recent))
        file_menu.addSeparator()
        self._action(
            file_menu,
            "&Save Project",
            self._save_project,
            QKeySequence.StandardKey.Save,
        )
        self._action(
            file_menu,
            "Save Project &As…",
            self._save_project_as,
            QKeySequence.StandardKey.SaveAs,
        )
        file_menu.addSeparator()
        self._action(
            file_menu, "&Import Audio…", self._import_audio, QKeySequence("Ctrl+I")
        )
        self.split_on_import_action = QAction("Split Songs into Stems on Import", self)
        self.split_on_import_action.setCheckable(True)
        self.split_on_import_action.setToolTip(
            "Run Demucs on imported songs and add drums, bass, other and vocals"
        )
        stored = QSettings("SonicLab", "SonicLab").value("import/splitStems", True)
        self.split_on_import_action.setChecked(
            str(stored).lower() not in ("false", "0")
        )
        self.split_on_import_action.toggled.connect(
            lambda checked: QSettings("SonicLab", "SonicLab").setValue(
                "import/splitStems", bool(checked)
            )
        )
        file_menu.addAction(self.split_on_import_action)
        self._action(file_menu, "Stem Splitter…", self._show_stem_splitter)
        file_menu.addSeparator()
        self.export_bars_action = self._action(
            file_menu,
            "Export Bars…",
            lambda: self._export_bars(),
            QKeySequence("Ctrl+Shift+E"),
        )
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, QKeySequence.StandardKey.Quit)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = self._action(
            edit_menu, "&Undo", self._undo, QKeySequence.StandardKey.Undo
        )
        self.redo_action = self._action(
            edit_menu, "&Redo", self._redo, QKeySequence.StandardKey.Redo
        )
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        edit_menu.addSeparator()
        self.rename_action = self._action(
            edit_menu, "Rename Track…", self._rename_track
        )
        self.delete_action = self._action(edit_menu, "Delete Track", self._delete_track)

        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self.browser_dock,
            self.inspector_dock,
            self.mixer_dock,
            self.visualizer_dock,
            self.stem_splitter_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        self.beats_action = QAction("Beat Markers", self)
        self.beats_action.setCheckable(True)
        self.beats_action.setChecked(True)
        self.beats_action.setToolTip("Show beats detected in the imported audio")
        self.beats_action.toggled.connect(lambda on: self._each_timeline(lambda view: view.set_show_beats(on)))
        view_menu.addAction(self.beats_action)
        self.detected_action = QAction("Downbeats", self)
        self.detected_action.setCheckable(True)
        self.detected_action.setChecked(True)
        self.detected_action.setToolTip("Show downbeats detected in the audio")
        self.detected_action.toggled.connect(lambda on: self._each_timeline(lambda view: view.set_show_detected(on)))
        view_menu.addAction(self.detected_action)
        self.onsets_action = QAction("Onsets", self)
        self.onsets_action.setCheckable(True)
        self.onsets_action.setChecked(True)
        self.onsets_action.setToolTip("Show onset timing against the beat reference")
        self.onsets_action.toggled.connect(lambda on: self._each_timeline(lambda view: view.set_show_onsets(on)))
        view_menu.addAction(self.onsets_action)
        view_menu.addSeparator()
        self.second_canvas_action = QAction("Second Canvas", self)
        self.second_canvas_action.setCheckable(True)
        self.second_canvas_action.setShortcut(QKeySequence("Ctrl+Shift+2"))
        self.second_canvas_action.setToolTip("A second arrangement beside the first; it plays with the song")
        self.second_canvas_action.toggled.connect(self._toggle_second_canvas)
        view_menu.addAction(self.second_canvas_action)
        layout_menu = view_menu.addMenu("Canvas Layout")
        group = QActionGroup(self)
        self.canvas_stacked_action = QAction("Stacked (Second Canvas Below)", self, checkable=True)
        self.canvas_side_action = QAction("Side by Side (Second Canvas Right)", self, checkable=True)
        for action, side in ((self.canvas_stacked_action, False), (self.canvas_side_action, True)):
            group.addAction(action)
            layout_menu.addAction(action)
            action.triggered.connect(lambda _checked=False, side=side: self._set_canvas_layout(side))
        self.canvas_stacked_action.setChecked(True)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            stored = QSettings("SonicLab", "SonicLab").value("view/canvasSideBySide", False)
            if str(stored).lower() in ("true", "1"):
                self._set_canvas_layout(True)
        follow_menu = view_menu.addMenu("Playhead Follow")
        follow_group = QActionGroup(self)
        self.follow_actions: dict[str, QAction] = {}
        for mode in FOLLOW_MODES:
            action = QAction(FOLLOW_LABELS[mode], self, checkable=True)
            action.setToolTip(FOLLOW_TIPS[mode])
            action.triggered.connect(lambda _checked=False, mode=mode: self._set_follow_mode(mode))
            follow_group.addAction(action)
            follow_menu.addAction(action)
            self.follow_actions[mode] = action
        follow_menu.addSeparator()
        self._action(follow_menu, "Next Follow Mode", self._cycle_follow_mode, QKeySequence("Ctrl+Shift+F"))
        self.follow_btn.setMenu(follow_menu)
        stored = str(QSettings("SonicLab", "SonicLab").value("view/followPlayhead", FOLLOW_OFF))
        self._set_follow_mode(stored if stored in FOLLOW_MODES else FOLLOW_OFF, announce=False)
        self._action(
            view_menu,
            "Zoom In",
            lambda: self._each_timeline(lambda view: view.zoom_horizontal(1.1)),
            QKeySequence.StandardKey.ZoomIn,
        )
        self._action(
            view_menu,
            "Zoom Out",
            lambda: self._each_timeline(lambda view: view.zoom_horizontal(1 / 1.1)),
            QKeySequence.StandardKey.ZoomOut,
        )
        self._action(
            view_menu, "Reset Zoom", lambda: self._each_timeline(lambda view: view.reset_zoom()), QKeySequence("Ctrl+0")
        )
        view_menu.addSeparator()
        self._action(
            view_menu, "Taller Tracks", lambda: self._each_timeline(lambda view: view.zoom_vertical(1.08))
        )
        self._action(
            view_menu, "Shorter Tracks", lambda: self._each_timeline(lambda view: view.zoom_vertical(1 / 1.08))
        )
        self._action(view_menu, "Reset Track Height", lambda: self._each_timeline(lambda view: view.reset_track_heights()))

        track_menu = self.menuBar().addMenu("&Track")
        self._action(
            track_menu, "Add Track…", self._add_empty_track, QKeySequence("Ctrl+T")
        )
        track_menu.addAction(self.rename_action)
        track_menu.addAction(self.delete_action)
        track_menu.addSeparator()
        self.mute_action = QAction("Mute", self)
        self.mute_action.setCheckable(True)
        self.mute_action.triggered.connect(self._mute_toggled)
        self.solo_action = QAction("Solo", self)
        self.solo_action.setCheckable(True)
        self.solo_action.triggered.connect(self._solo_toggled)
        track_menu.addAction(self.mute_action)
        track_menu.addAction(self.solo_action)
        track_menu.addSeparator()
        self.move_up_action = self._action(
            track_menu, "Move Up", lambda: self._move_track(-1)
        )
        self.move_down_action = self._action(
            track_menu, "Move Down", lambda: self._move_track(1)
        )
        self.canvas_move_action = self._action(
            track_menu, "Move to Other Canvas", self._move_selected_to_other_canvas, QKeySequence("Ctrl+Shift+M")
        )

        clip_menu = self.menuBar().addMenu("&Clip")
        clip_menu.addAction(self.rename_action)
        self.move_start_action = self._action(
            clip_menu, "Move to Start", self._move_selection_to_start
        )
        self.split_action = self._action(
            clip_menu, "Split at Playhead", self._split_selected, QKeySequence("Ctrl+E")
        )
        self.duplicate_action = self._action(
            clip_menu, "Duplicate", self._duplicate_selected, QKeySequence("Ctrl+D")
        )
        clip_menu.addSeparator()
        self.analyze_action = self._action(
            clip_menu, "Analyze Downbeats", self._analyze_beats
        )
        self.onset_action = self._action(
            clip_menu, "Analyze Onsets", self._analyze_onsets
        )
        self.reference_action = self._action(
            clip_menu, "Set Beat Reference", self._set_beat_reference
        )
        self.nudge_action = self._action(
            clip_menu, "Nudge to Reference", self._nudge_to_reference
        )
        clip_menu.addSeparator()
        downbeat_menu = clip_menu.addMenu("Downbeat Markers")
        self.apply_downbeats_action = self._action(
            downbeat_menu,
            "Apply to Other Tracks…",
            lambda: self._apply_downbeats_to_tracks(),
        )
        # Lambdas: triggered(checked) would otherwise fill the optional argument.
        self.export_downbeats_action = self._action(
            downbeat_menu, "Export to File…", lambda: self._export_downbeats()
        )
        self.import_downbeats_action = self._action(
            downbeat_menu, "Import from File…", lambda: self._import_downbeats()
        )
        self.clear_downbeats_action = self._action(
            downbeat_menu, "Clear Applied Downbeats", self._clear_applied_downbeats
        )
        downbeat_menu.addSeparator()
        self.drum_track_action = self._action(
            downbeat_menu, "Drum Track (Others Follow It)", self._toggle_drum_track
        )
        self.drum_track_action.setCheckable(True)
        self.downbeat_menu = downbeat_menu
        clip_menu.aboutToShow.connect(self._sync_clip_actions)

        audio_menu = self.menuBar().addMenu("&Audio")
        self._action(audio_menu, "Play", self._play_hovered)
        self.play_both_action = self._action(audio_menu, "Play Both Canvases", lambda: self._play())
        self._action(audio_menu, "Pause", self._pause)
        self._action(audio_menu, "Stop", self._stop)
        self._action(audio_menu, "Return to Start", self._return_to_start)
        audio_menu.addSeparator()
        self._action(audio_menu, "Record", self._toggle_record)
        self._action(audio_menu, "Input Device…", self._choose_input)
        audio_menu.addSeparator()
        settings = QSettings("SonicLab", "SonicLab")
        self.loop_actions = []
        for intervals in (1, 2, 3, 4):
            label = (
                "Loop 1 Downbeat" if intervals == 1 else f"Loop {intervals} Downbeats"
            )
            action = self._action(
                audio_menu,
                label,
                lambda checked=False, count=intervals: self._loop_downbeats(count),
                QKeySequence(loop_shortcut(intervals, settings)),
            )
            self.loop_actions.append(action)
        self._action(audio_menu, "Clear Loop", self._clear_loop)
        audio_menu.addAction(self._unavailable_action("Metronome"))

        bars_menu = self.menuBar().addMenu("&Bars")
        self._action(
            bars_menu,
            "Edit Bar at Playhead…",
            lambda: self._edit_bar(),
            QKeySequence("Ctrl+B"),
        )
        self._action(
            bars_menu,
            "Add Bar Line at Playhead",
            self._add_bar_line,
            QKeySequence("Ctrl+Shift+B"),
        )
        self._action(bars_menu, "Remove Nearest Bar Line", self._remove_bar_line)
        bars_menu.addSeparator()
        self._action(
            bars_menu, "Mark Bars at Drum Downbeats", self._mark_bars_from_drums
        )
        self._action(bars_menu, "Even Bars at the BPM Tempo", self._even_bars_at_tempo)
        self._action(bars_menu, "Reset Bars", self._reset_bars)
        bars_menu.addSeparator()
        bars_menu.addAction(self.export_bars_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self._unavailable_action("Effects"))
        tools_menu.addAction(self._unavailable_action("Audio Analysis"))

        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "About SonicLab", self._about)

        self.project_name = QLineEdit()
        self.project_name.setObjectName("projectName")
        self.project_name.setText(self.project.name)
        self.project_name.editingFinished.connect(self._rename_project)
        self.menuBar().setCornerWidget(self.project_name, Qt.Corner.TopRightCorner)
        self._sync_clip_actions()

    def _action(self, menu, label, slot, shortcut=None) -> QAction:
        action = QAction(label, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _unavailable_action(self, label) -> QAction:
        action = QAction(f"{label}", self)
        action.setEnabled(False)
        action.setStatusTip(f"{label} is not available yet")
        action.setToolTip(f"{label} is not available yet")
        return action

    def _connect(self):
        self.bpm.valueChanged.connect(self._bpm_changed)
        self._tempo_timer = QTimer(self)
        self._tempo_timer.setSingleShot(True)
        self._tempo_timer.setInterval(700)
        self._tempo_timer.timeout.connect(self._apply_typed_tempo)
        self.bpm.editingFinished.connect(self._commit_edit)
        self.signature.currentTextChanged.connect(self._signature_changed)
        for headers, timeline in self._canvases():
            timeline.on_trim = self._trim_clip
            timeline.on_release = self._clip_released
            timeline.on_block_move = self._move_block
            timeline.seek_requested.connect(
                lambda seconds, canvas=timeline.canvas: self._seek_on_canvas(canvas, seconds)
            )
            timeline.track_selected.connect(lambda index, view=timeline: self._on_view_selected(view, index))
            timeline.layout_changed.connect(self._canvas_layout_changed)
            timeline.bars_changed.connect(
                lambda lines: self._set_bars(lines, "Bar line moved")
            )
            timeline.bar_edit_requested.connect(self._edit_bar)
            timeline.bars_selected.connect(self._select_bars)
            timeline.bar_play_requested.connect(lambda bar, track: self._play_bar(bar, track))
            timeline.downbeat_clicked.connect(self._show_bar_menu)
            timeline.rename_requested.connect(self._rename_track_at)
            timeline.canvas_move_requested.connect(self._move_track_to_canvas)
            timeline.blocks_move_requested.connect(
                lambda blocks, grabbed, shift, canvas, row, view=timeline: self._move_blocks(
                    view, blocks, grabbed, shift, canvas, row
                )
            )
            timeline.canvas_drop_requested.connect(
                lambda indices, grabbed, shift, row, view=timeline: self._drop_on_canvas(
                    view, indices, grabbed, shift, row
                )
            )
            timeline.stem_loop_dragged.connect(self._drag_stem_loop)
            timeline.row_move_requested.connect(
                lambda index, row, view=timeline: self._move_track_to_row(index, view, row)
            )
            timeline.viewport().setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            timeline.viewport().customContextMenuRequested.connect(
                lambda pos, view=timeline: self._clip_context_menu(pos, view)
            )
            timeline.verticalScrollBar().valueChanged.connect(
                lambda value, pair=(headers, timeline): self._sync_from_timeline(value, pair)
            )
            headers.scroll.verticalScrollBar().valueChanged.connect(
                lambda value, pair=(headers, timeline): self._sync_from_headers(value, pair)
            )
            headers.selected.connect(self._select_track)
            headers.gain_changed.connect(self._set_gain)
            headers.pan_changed.connect(self._set_pan)
            headers.mute_changed.connect(self._set_muted)
            headers.solo_changed.connect(self._set_solo)
            headers.arm_changed.connect(self._set_armed)
            headers.edit_finished.connect(self._commit_edit)
            headers.resize_committed.connect(self._on_track_resized)
            headers.unavailable.connect(self._unavailable)
            headers.rename_requested.connect(self._rename_track_at)
            headers.play_requested.connect(lambda canvas=timeline.canvas: self._play_canvas(canvas))
        self.mixer.selected.connect(self._select_track)
        self.mixer.gain_changed.connect(self._set_gain)
        self.mixer.pan_changed.connect(self._set_pan)
        self.mixer.mute_changed.connect(self._set_muted)
        self.mixer.solo_changed.connect(self._set_solo)
        self.mixer.edit_finished.connect(self._commit_edit)
        self.browser.track_activated.connect(self._select_track)
        self.browser.recent_activated.connect(self._open_recent)
        self.inspector.name_edited.connect(self._rename_track_to)
        self.inspector.start_edited.connect(self._set_start)
        self.inspector.gain_edited.connect(
            lambda gain: self._set_gain(self._selected, gain)
        )
        self.inspector.pan_edited.connect(
            lambda pan: self._set_pan(self._selected, pan)
        )
        self.inspector.mute_edited.connect(
            lambda muted: self._set_muted(self._selected, muted)
        )
        self.inspector.solo_edited.connect(
            lambda solo: self._set_solo(self._selected, solo)
        )
        self.inspector.clip_gain_edited.connect(self._set_clip_gain)
        self.inspector.fade_in_edited.connect(
            lambda value: self._set_fade("fade_in", value)
        )
        self.inspector.fade_out_edited.connect(
            lambda value: self._set_fade("fade_out", value)
        )
        self.inspector.edit_finished.connect(self._commit_edit)
        self.visualizer.set_reader(self._read_visualizer)

    def eventFilter(self, watched, event):
        if not isinstance(watched, QObject):
            return False
        try:
            event_type = event.type()
        except (AttributeError, RuntimeError, TypeError):
            return False
        if event_type in (
            QEvent.Type.DragEnter,
            QEvent.Type.DragMove,
            QEvent.Type.Drop,
        ):
            return self._handle_drop_event(watched, event_type, event)
        if event_type == QEvent.Type.Enter and isinstance(watched, QWidget):
            if watched is self.splitter2 or self.splitter2.isAncestorOf(watched):
                self._hover_canvas = 1
            elif watched is self.splitter or self.splitter.isAncestorOf(watched):
                self._hover_canvas = 0
        if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            # Esc cancels a stem block drag on either canvas; the block stays.
            if any(view.cancel_block_drag() for view in self._timelines()):
                self.statusBar().showMessage("Move cancelled", 2000)
                return True
            return False
        if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if QApplication.activeModalWidget() is not None:
                return False
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QAbstractSpinBox, QComboBox)):
                return False
            if not event.isAutoRepeat():
                self._toggle_transport()
            return True
        if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Delete:
            if QApplication.activeModalWidget() is not None:
                return False
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QAbstractSpinBox, QComboBox)):
                return False
            self._delete_track()
            return True
        if (
            event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() in ARROW_KEYS
        ):
            return self._handle_arrow_key(event_type, event)
        return False

    def _handle_arrow_key(self, event_type, event) -> bool:
        """Left/Right seek 5 s, Up/Down change the master volume by 10%.

        Text fields, spin boxes, lists and sliders keep their own arrow keys,
        and so do arrows with a modifier held.
        """
        if (
            QApplication.activeModalWidget() is not None
            or QApplication.activePopupWidget() is not None
        ):
            return False
        if event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier:
            return False
        focused = QApplication.focusWidget()
        if isinstance(
            focused,
            (
                QLineEdit,
                QAbstractSpinBox,
                QComboBox,
                QAbstractSlider,
                QAbstractItemView,
            ),
        ):
            return False
        key = event.key()
        if event_type == QEvent.Type.KeyRelease:
            # One undo step per volume gesture, however long the key is held.
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down) and not event.isAutoRepeat():
                self._commit_edit()
            return True
        if key == Qt.Key.Key_Left:
            self._seek_by(-SEEK_STEP_SECONDS)
        elif key == Qt.Key.Key_Right:
            self._seek_by(SEEK_STEP_SECONDS)
        elif key == Qt.Key.Key_Up:
            self._step_master_volume(VOLUME_STEP)
        else:
            self._step_master_volume(-VOLUME_STEP)
        return True

    def _seek_by(self, seconds: float) -> None:
        end = max(
            [
                float(t.get("start", 0.0) or 0.0) + float(t.get("duration", 0.0) or 0.0)
                for t in self.project.tracks
            ]
            + [0.0]
        )
        target = max(0.0, float(self.engine.position) + float(seconds))
        if end > 0.0:
            target = min(target, end)
        self.engine.seek(target)
        self._refresh_transport()
        self.statusBar().showMessage(
            f"{'⏪' if seconds < 0 else '⏩'} {format_clock(target)}", 1500
        )

    def _step_master_volume(self, step: float) -> None:
        # Round to whole percent so repeated steps land on 10%, 20%, …
        gain = round(self.engine.output_gain + step, 2)
        gain = max(0.0, min(MASTER_GAIN_MAX, gain))
        self._set_gain(-1, gain)
        self.statusBar().showMessage(f"Master volume {gain * 100:.0f}%", 1500)

    def _toggle_transport(self):
        if self.engine.playing and self._track_focus is None:
            self._pause()
        else:
            self._play_hovered()

    def _play_hovered(self) -> None:
        """Play the canvas the mouse is (or was last) over, from its own
        playhead. With one canvas this is plain Play."""
        if not self.splitter2.isVisible():
            self._clear_track_focus()
            self._play(keep_focus=True)
            return
        if self.engine.playing and self._canvas_focus == self._hover_canvas and self._track_focus is None:
            return
        self._play_canvas(self._hover_canvas)

    def _unavailable(self, message: str):
        self.statusBar().showMessage(message, 4000)

    def _import_audio(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import audio",
            "",
            "Audio files (*.wav *.flac *.ogg *.aiff *.aif *.mp3);;All files (*)",
        )
        self._import_paths(paths)

    def _import_paths(self, paths, split: bool = True, note: str | None = None) -> None:
        """Add audio files as tracks, pick the drum track, queue analysis.

        A song that will be split is not added itself. Its stems are added
        now when an earlier split exists, otherwise when Demucs finishes.
        Files are decoded on worker threads; the tracks arrive together, in
        one edit and one canvas refresh, when the last one is ready. ``note``
        replaces the status message shown when they do.
        """
        items = []  # ("file", path, name, source) or ("stems", name, stems)
        splitting = []
        reused = []
        not_split = False
        for path in list(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            source = os.path.abspath(path)
            wants_split = (
                split
                and self.split_on_import_action.isChecked()
                and not is_stem_name(name)
            )
            if wants_split:
                stems = existing_stems(source, default_stem_folder(self.current_path))
                if stems is not None:
                    items.append(("stems", name, stems))
                    reused.append(name)
                    continue
                if self.stem_jobs.available:
                    job = self.stem_jobs.submit(source, default_stem_folder(self.current_path))
                    self._stem_names[job] = name
                    self._stem_sources[job] = source
                    splitting.append(name)
                    continue
                not_split = True
            items.append(("file", path, name, source))
        if splitting:
            self._show_stem_progress(0.0)
        files = []
        for item in items:
            if item[0] == "file":
                files.append(item[1])
            else:
                files.extend(os.path.abspath(item[2][stem]) for stem in STEMS)
        if not files:
            return
        self.statusBar().showMessage("Loading audio…")
        self.loader.submit(
            files, lambda results: self._finish_import(items, results, reused, not_split, bool(splitting), note)
        )

    def _finish_import(self, items, results, reused, not_split, splitting, note=None) -> None:
        """Add decoded files as tracks: one edit, one refresh."""
        ui_trace.mark(f"import {len(items)} item(s)")
        added = False
        imported = []
        self._remember_edit()
        for item in items:
            try:
                if item[0] == "stems":
                    _kind, name, stems = item
                    self._add_stem_tracks(name, stems, results=results)
                    imported.extend(stems[stem] for stem in STEMS)
                    added = True
                    continue
                _kind, path, name, source = item
                loaded = results.get(path)
                if isinstance(loaded, BaseException):
                    raise loaded
                self.project.tracks.append(
                    {
                        "name": name,
                        "path": source,
                        "start": 0.0,
                        "duration": len(loaded.audio) / OUTPUT_RATE,
                        **loaded.track_fields(),
                    }
                )
                added = True
                imported.append(source)
            except Exception as exc:
                QMessageBox.warning(self, "Could not import audio", f"{item[1]}\n\n{exc}")
        drum = self._detect_drum_track() if added else None
        self._commit_edit()
        if added:
            self._reload_engine()
        self._refresh_timeline()
        if not (added and self.project.tracks):
            self.statusBar().clearMessage()
            return
        self._select_track(len(self.project.tracks) - 1)
        self._queue_beat_sources(imported)
        lead = self.project.tracks[drum].get("name") if drum is not None else None
        if reused:
            message = f"Loaded saved stems for “{reused[0]}”" + (
                f" and {len(reused) - 1} more" if len(reused) > 1 else ""
            )
        elif not_split:
            message = "Audio imported. Demucs was not found, so the song was not split"
        else:
            message = "Audio imported"
        message += f". Drum track: {lead}" if lead else ". No drum track found"
        if note:
            self.statusBar().showMessage(note, 10000)
        elif not splitting:
            self.statusBar().showMessage(message, 8000)

    def finish_loading(self) -> None:
        """Wait for audio still being decoded and add it now."""
        self.loader.wait()

    def _add_stem_tracks(self, name: str, stems: dict, order=STEMS, results=None) -> list[dict]:
        """Append one track per stem in ``order``. Marks drums if none lead.

        ``results`` holds audio already decoded by the loader, by path;
        without it the files are decoded here.
        """
        added = []
        group = uuid.uuid4().hex  # these stems share one playback clock
        for stem in order:
            path = os.path.abspath(stems[stem])
            loaded = results.get(path) if results is not None else None
            if isinstance(loaded, BaseException):
                raise loaded
            if results is not None and loaded is None:
                raise FileNotFoundError(path)
            fields = loaded.track_fields() if isinstance(loaded, Loaded) else {"_audio": decode_audio(path)}
            added.append(
                {
                    "name": f"{name} {stem}",
                    "path": path,
                    "start": 0.0,
                    "duration": len(fields["_audio"]) / OUTPUT_RATE,
                    "stem_group": group,
                    **fields,
                }
            )
        self.project.tracks.extend(added)
        if self._drum_index() is None and added and order[0] == "drums":
            added[0]["drum_reference"] = True
        return added

    def _show_stem_splitter(self) -> None:
        self.stem_splitter_dock.show()
        self.stem_splitter_dock.raise_()

    def _add_split_stems(self, name: str, stems: dict, order: tuple) -> None:
        """Add stems from the Stem Splitter panel as tracks, all starting at 0.
        They are decoded off the GUI thread and added together."""
        files = [os.path.abspath(stems[stem]) for stem in order]
        self.statusBar().showMessage(f"Loading the stems of “{name}”…")
        self.loader.submit(files, lambda results: self._finish_split_stems(name, stems, order, results))

    def _finish_split_stems(self, name: str, stems: dict, order: tuple, results) -> None:
        self._remember_edit()
        try:
            added = self._add_stem_tracks(name, stems, order, results=results)
        except Exception as exc:
            self._gesture_before = None
            QMessageBox.warning(self, "Could not load stems", str(exc))
            return
        if self._drum_index() is None:
            self._detect_drum_track()
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(len(self.project.tracks) - len(added))
        self._queue_beat_sources(track["path"] for track in added)
        drum = self._drum_index()
        lead = self.project.tracks[drum].get("name") if drum is not None else "none"
        self.statusBar().showMessage(
            f"Added {len(added)} stem{'s' if len(added) != 1 else ''} of “{name}”. Drum track: {lead}",
            10000,
        )

    def _show_stem_progress(self, fraction: float | None) -> None:
        """Show the splitting banner, or hide it when ``fraction`` is None."""
        waiting = self.stem_jobs.pending()
        visible = fraction is not None and waiting > 0
        self.stem_banner.setVisible(visible)
        if not visible:
            return
        self.stem_bar.setValue(int(round(max(0.0, min(1.0, fraction)) * 1000)))
        name = self._splitting_name()
        queued = f"   ·   {waiting - 1} more waiting" if waiting > 1 else ""
        text = f"Splitting “{name}”…  {int(fraction * 100):3d}%{queued}"
        metrics = self.stem_label.fontMetrics()
        self.stem_label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, self.stem_label.width()))
        self.stem_label.setToolTip(f"Splitting “{name}” into stems: drums, bass, other and vocals{queued}")

    def _splitting_name(self) -> str:
        return next(iter(self._stem_names.values()), "song")

    def _cancel_stems(self) -> None:
        self.stem_jobs.cancel()
        self.stem_label.setText("Cancelling…")

    def _poll_stems(self) -> None:
        for _ in range(16):
            message = self.stem_jobs.poll()
            if message is None:
                return
            kind, job = message[0], message[1]
            if kind == "progress" and job in self._stem_names:
                # The first name in the dict is the job Demucs is running.
                if next(iter(self._stem_names)) == job:
                    self._show_stem_progress(float(message[2]))
            elif kind == "finished":
                name = self._stem_names.pop(job, "song")
                source = self._stem_sources.pop(job, None)
                self._stems_finished(name, source, message[2])
                self._show_stem_progress(0.0 if self._stem_names else None)

    def _stems_finished(self, name: str, source: str | None, result) -> None:
        if result.status != STEMS_OK:
            cancelled = result.status == STEMS_CANCELLED
            text = (
                f"Stem split of “{name}” cancelled"
                if cancelled
                else (
                    f"Could not split “{name}”: {result.message or 'Demucs failed'}. Imported the full song"
                )
            )
            self.statusBar().showMessage(text, 10000)
            if source and not cancelled:
                # Never leave the user with nothing: fall back to the song.
                self._import_paths([source], split=False, note=text)
            return
        files = [os.path.abspath(result.stems[stem]) for stem in STEMS]
        self.loader.submit(files, lambda results: self._finish_stems(name, result, results))

    def _finish_stems(self, name: str, result, results) -> None:
        self._remember_edit()
        try:
            added = self._add_stem_tracks(name, result.stems, results=results)
        except Exception as exc:
            self._gesture_before = None
            QMessageBox.warning(self, "Could not load stems", str(exc))
            return
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(len(self.project.tracks) - len(added))
        self._queue_beat_sources(track["path"] for track in added)
        drum = self._drum_index()
        lead = self.project.tracks[drum].get("name") if drum is not None else "none"
        self.statusBar().showMessage(
            f"Split “{name}” into {len(added)} stems. Drum track: {lead}", 10000
        )

    def _dropped_audio(self, event) -> list[str]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return [
            url.toLocalFile()
            for url in mime.urls()
            if url.isLocalFile()
            and os.path.splitext(url.toLocalFile())[1].lower() in AUDIO_EXTENSIONS
        ]

    def _handle_drop_event(self, watched, event_type, event) -> bool:
        """Accept audio files dropped anywhere in this window."""
        if not isinstance(watched, QWidget) or watched.window() is not self:
            return False
        if QApplication.activeModalWidget() is not None:
            return False
        paths = self._dropped_audio(event)
        if not paths:
            return False
        event.acceptProposedAction()
        if watched is self.stem_splitter or self.stem_splitter.isAncestorOf(watched):
            # A song dropped on the Stem Splitter is split, not imported.
            if event_type == QEvent.Type.Drop:
                QTimer.singleShot(0, lambda: self.stem_splitter.set_source(paths[0]))
            return True
        if event_type == QEvent.Type.Drop:
            # Let the file manager finish its drag before dialogs can appear.
            QTimer.singleShot(0, lambda: self._import_paths(paths))
        return True

    def _canvases(self):
        """(header column, timeline view) of each canvas, main first."""
        return ((self.headers, self.timeline), (self.headers2, self.timeline2))

    def _timelines(self):
        return (self.timeline, self.timeline2)

    def _view_for(self, index: int) -> TimelineView:
        """The timeline view that draws track ``index``."""
        tracks = self.project.tracks
        if 0 <= index < len(tracks) and track_canvas(tracks[index]) == 1:
            return self.timeline2
        return self.timeline

    def _header_heights(self, tracks, timeline=None) -> list[int]:
        timeline = timeline or self.timeline
        rows, primaries = timeline.lane_rows(tracks)
        heights = []
        for index in range(len(tracks)):
            if rows[index] >= 0 and primaries[rows[index]] == index:
                heights.append(timeline.row_height(index))
            else:
                heights.append(0)
        return heights

    def _lane_primary(self, index: int) -> int:
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return index
        rows, primaries = self._view_for(index).lane_rows(tracks)
        if not primaries or rows[index] < 0:
            return index
        return primaries[rows[index]]

    def _refresh_timeline(self):
        bpm = float(self.bpm.value())
        time_signature = self.signature.currentText()
        numerator, denominator = parse_time_signature(time_signature)
        self.project.bpm = bpm
        self.project.time_signature = [numerator, denominator]
        bars = self._bar_lines()
        if any(track_canvas(track) == 1 for track in self.project.tracks):
            self.splitter2.setVisible(True)
            self.second_canvas_action.blockSignals(True)
            self.second_canvas_action.setChecked(True)
            self.second_canvas_action.blockSignals(False)
        self._canvases_rebuilding = True
        try:
            for timeline in self._timelines():
                timeline.rebuild(
                    self.project.tracks,
                    self._clip_moved,
                    bpm,
                    time_signature,
                    bars,
                )
        finally:
            self._canvases_rebuilding = False
        self._rebuild_side_panels()
        self._sync_bpm_field()

    def _canvas_layout_changed(self) -> None:
        # While _refresh_timeline rebuilds both canvases it lays the panels
        # out once at the end, not once per canvas.
        if not self._canvases_rebuilding:
            self._rebuild_side_panels()

    def _rebuild_side_panels(self):
        tracks = self.project.tracks
        for headers, timeline in self._canvases():
            headers.rebuild(tracks, self._header_heights(tracks, timeline), self._selected)
            headers.set_scene_height(timeline.scene_obj.sceneRect().height())
            headers.sync(timeline.verticalScrollBar().value())
        self.mixer.rebuild(tracks, self._selected)
        self.visualizer.set_sources([track.get("name", "Track") for track in tracks])
        self._refresh_browser()
        self._show_selection()
        self._apply_detected_beats()
        self._apply_onsets()

    def _refresh_browser(self):
        self.browser.set_session(
            self.project.name or "Untitled",
            self.current_path,
            self.project.tracks,
            self._recent_paths(),
        )

    def _add_empty_track(self):
        name, ok = QInputDialog.getText(self, "New track", "Track name:")
        if not ok or not name.strip():
            return
        self._remember_edit()
        audio = np.zeros((1, 2), dtype=np.float32)
        self.project.tracks.append(
            {
                "name": name.strip(),
                "path": None,
                "start": 0.0,
                "duration": 0.0,
                "_audio": audio,
                "empty": True,
            }
        )
        index = self.engine.add_clip(audio, 0.0)
        self._push_mix(index, self.project.tracks[-1])
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(len(self.project.tracks) - 1)

    def _clip_moved(self, index, start):
        if index < len(self.project.tracks):
            if not self._history_frozen:
                self._remember_edit()
            self.project.tracks[index]["start"] = start
            self.engine.set_clip_start(index, start)
            if index == self._selected:
                self.inspector.set_start(start)

    def _push_mix(self, index, track):
        self.engine.set_clip_gain(index, float(track.get("gain", 1.0)))
        self.engine.set_clip_pan(index, float(track.get("pan", 0.0)))
        self._push_audible(index, track)
        duration = float(track.get("duration", 0.0) or 0.0)
        self.engine.set_clip_region(
            index,
            float(track.get("start", 0.0)),
            float(track.get("offset", 0.0) or 0.0),
            duration if duration > 0 else 1.0 / OUTPUT_RATE,
            float(track.get("fade_in", 0.0) or 0.0),
            float(track.get("fade_out", 0.0) or 0.0),
            float(track.get("clip_gain", 1.0) or 1.0),
        )

    def _push_audible(self, index, track) -> None:
        """Mute and solo for the engine. While one canvas plays on its own,
        the other canvas's tracks are silent and their solos ignored; the
        tracks' own mute and solo settings are not changed."""
        outside = self._canvas_focus is not None and track_canvas(track) != self._canvas_focus
        focus = self._track_focus
        if focus is not None and 0 <= focus < len(self.project.tracks):
            # A bar's ▶ plays its own track (stem) only.
            outside = outside or self.project.tracks[focus] is not track
        self.engine.set_clip_muted(index, bool(track.get("muted", False)) or outside)
        self.engine.set_clip_solo(index, bool(track.get("solo", False)) and not outside)

    def _set_canvas_focus(self, canvas: int | None) -> None:
        """Hear one canvas only (0 or 1), or both (None)."""
        self._canvas_focus = canvas
        for index, track in enumerate(self.project.tracks):
            self._push_audible(index, track)
        self._sync_canvas_play_buttons()

    def _seek_on_canvas(self, canvas: int, seconds: float) -> None:
        """A click on one canvas's ruler moves that canvas's playhead."""
        self._canvas_positions[canvas] = max(0.0, float(seconds))
        self._active_canvas = canvas
        self.engine.seek(seconds)

    def _play_canvas(self, canvas: int) -> None:
        """The ▶ in a canvas's corner: play that canvas alone, or stop it."""
        if self.engine.playing and self._canvas_focus == canvas and self._track_focus is None:
            self._pause()
            return
        self._clear_track_focus()  # every stem on the canvas
        if canvas == 1 and not self.splitter2.isVisible():
            return
        if self.engine.playing:
            # Switching canvases while playing: keep the current position.
            self._canvas_positions[canvas] = float(self.engine.position)
        else:
            self.engine.seek(self._canvas_positions[canvas])  # start where its playhead rests
        self._active_canvas = canvas
        self._set_canvas_focus(canvas)
        if not self.engine.playing:
            self._play(keep_focus=True)
        name = "the main canvas" if canvas == 0 else "canvas 2"
        self.statusBar().showMessage(f"Playing {name} only. ▶ in the toolbar plays both", 4000)

    def _sync_canvas_play_buttons(self) -> None:
        playing = self.engine.playing
        self.headers.set_playing(playing and self._canvas_focus == 0)
        self.headers2.set_playing(playing and self._canvas_focus == 1)

    def _reload_engine(self):
        # One step for the audio thread: playback never hears the clips
        # half reloaded, e.g. while importing during playback.
        with self.engine.batch():
            self.engine.clear_clips()
            for index, track in enumerate(self.project.tracks):
                audio = track.get("_audio")
                if audio is None:
                    audio = np.zeros((1, 2), dtype=np.float32)
                    track["_audio"] = audio
                self.engine.add_clip(audio, float(track.get("start", 0.0)))
                self._push_mix(index, track)
            if self._preview_tracks:
                self._follow_preview_tracks()
                self._push_stem_mutes()
            self._push_stem_loops()
        if self._preview_tracks:
            QTimer.singleShot(0, self._prune_previews)  # after the canvases redraw

    def _set_session_title(self):
        name = (self.project.name or "Untitled").strip() or "Untitled"
        self.project.name = name
        self.project_name.blockSignals(True)
        self.project_name.setText(name)
        self.project_name.blockSignals(False)
        self.setWindowTitle(f"SonicLab — {name}")

    def _rename_project(self):
        self._remember_edit()
        self.project.name = self.project_name.text().strip() or "Untitled"
        self._set_session_title()
        self._commit_edit()
        self._refresh_browser()

    def _musical_position(self, seconds):
        numerator, _denominator = parse_time_signature(self.signature.currentText())
        bar, beat_in_bar = bar_position(
            self.timeline.bar_lines,
            float(seconds),
            numerator,
            self.timeline.bar_fallback,
        )
        return f"{bar:03d}.{beat_in_bar}"

    def _sync_transport_lamp(self):
        playing = self.engine.playing
        if playing == self._playing_shown:
            return
        self._sync_canvas_play_buttons()  # playback can also end by itself
        self._playing_shown = playing
        self.play_btn.setProperty("playing", playing)
        self.play_btn.setIcon(play_icon("#F4E7CC" if playing else "#E6E8EC"))
        self.play_btn.style().unpolish(self.play_btn)
        self.play_btn.style().polish(self.play_btn)
        self.play_btn.update()

    def _shown_position(self, playing: bool) -> float:
        # Playing: what the speakers play now, between audio blocks too.
        return self.engine.heard_position() if playing else self.engine.position

    def _update_playheads(self) -> tuple[bool, float]:
        """Move each canvas's playhead; the canvases themselves stay put."""
        playing = self.engine.playing
        seconds = self._shown_position(playing)
        for timeline in self._timelines():
            canvas = timeline.canvas
            heard = self._canvas_focus is None or self._canvas_focus == canvas
            if playing:
                moving = heard
            elif self._canvas_focus is not None:
                moving = self._canvas_focus == canvas
            else:
                moving = canvas == self._active_canvas
            if moving:
                self._canvas_positions[canvas] = seconds
            timeline.set_playhead(self._canvas_positions[canvas])
        return playing, seconds

    def _animate_playheads(self) -> None:
        """One playhead frame. Reads the engine's snapshot; never waits on audio."""
        now = time.monotonic()
        dt = min(0.1, now - self._last_frame) if self._last_frame else PLAYHEAD_FRAME_MS / 1000.0
        self._last_frame = now
        shown = self._update_playheads()
        playing = shown[0]
        previews = self._sync_previews() if (self._preview_tracks or self.engine.previews()) else {}
        if not playing and not previews:
            self.frame_timer.stop()
            self._last_frame = 0.0
        self._frames += 1
        if not playing or self._frames % 2 == 0:
            # Labels, meters and job polls at half the frame rate.
            self._refresh_transport(shown)
        if not playing or self.follow_mode == FOLLOW_OFF:
            return
        followers = [
            timeline
            for timeline in self._timelines()
            if timeline.isVisible() and (self._canvas_focus is None or self._canvas_focus == timeline.canvas)
        ]
        if not self.canvas_side_by_side:
            # Stacked canvases share one scroll; two followers would fight.
            followers = followers[:1]
        for timeline in followers:
            timeline.follow_playhead(self.follow_mode, dt)

    def _transport_tick(self) -> None:
        # While playing, the frame timer moves the playhead and calls this.
        if not self.frame_timer.isActive():
            self._refresh_transport()

    def _refresh_transport(self, shown: tuple[bool, float] | None = None):
        playing, seconds = self._update_playheads() if shown is None else shown
        minutes = int(seconds // 60)
        secs = seconds % 60
        self.position_label.setText(f"{minutes:02d}:{secs:06.3f}")
        self.musical_label.setText(self._musical_position(seconds))
        if (playing or self._preview_tracks) and not self.frame_timer.isActive():
            self.frame_timer.start()
        self._sync_transport_lamp()
        self._sync_capture()
        master, tracks = self.engine.take_peaks()
        self.mixer.set_levels(tracks, master)
        self._poll_recording()
        self._poll_beats()
        self._poll_onsets()
        self._poll_stems()
        self.loader.poll()
        if self.engine.playing:
            self.stem_splitter.stop_preview()  # one sound source at a time

    def _set_follow_mode(self, mode: str, announce: bool = True) -> None:
        """Off, Follow or Center: how the canvas scrolls during playback."""
        if mode not in FOLLOW_MODES:
            return
        self.follow_mode = mode
        self.follow_actions[mode].setChecked(True)
        self.follow_btn.setText(f"Follow: {FOLLOW_LABELS[mode]}")
        self.follow_btn.setToolTip(
            f"Playhead follow: {FOLLOW_LABELS[mode]}. {FOLLOW_TIPS[mode]}.\n"
            "Click for the next mode (Ctrl+Shift+F), or pick one from the arrow"
        )
        if announce:
            QSettings("SonicLab", "SonicLab").setValue("view/followPlayhead", mode)
            self.statusBar().showMessage(f"Playhead follow: {FOLLOW_LABELS[mode]}. {FOLLOW_TIPS[mode]}", 3000)

    def _cycle_follow_mode(self) -> None:
        modes = list(FOLLOW_MODES)
        self._set_follow_mode(modes[(modes.index(self.follow_mode) + 1) % len(modes)])

    def _clear_track_focus(self) -> None:
        if self._track_focus is not None:
            self._track_focus = None
            for index, track in enumerate(self.project.tracks):
                self._push_audible(index, track)

    def _play(self, keep_focus: bool = False):
        if not keep_focus:
            self._clear_track_focus()
            if self._canvas_focus is not None:
                self._set_canvas_focus(None)  # the transport plays both canvases
        try:
            self.engine.play()
        except Exception as exc:
            QMessageBox.critical(
                self, "Audio output error", f"Could not start audio output.\n\n{exc}"
            )
        if self.engine.playing:
            self.frame_timer.start()
        self._sync_transport_lamp()

    def _pause(self):
        self.engine.pause()
        self._sync_transport_lamp()
        self._sync_canvas_play_buttons()

    def _stop(self):
        # Stop is the panic button: the song and every stem block.
        self.engine.preview_stop_all()
        self._sync_previews()
        self.engine.stop()
        self._sync_transport_lamp()
        self._sync_canvas_play_buttons()

    def _return_to_start(self):
        self.engine.seek(0.0)

    def _loop_downbeats(self, intervals: int) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to loop", 3000)
            return
        track = self.project.tracks[index]
        events = self._track_events(track)
        region = downbeat_loop_region(
            events,
            self.engine.position,
            intervals,
            float(track.get("start", 0.0) or 0.0),
            float(track.get("offset", 0.0) or 0.0),
        )
        if region is None:
            if any(int(position) == 1 for _time, position in events):
                noun = "interval" if int(intervals) == 1 else "intervals"
                self.statusBar().showMessage(
                    f"Not enough downbeats to loop {int(intervals)} {noun}",
                    4000,
                )
            else:
                self.statusBar().showMessage(
                    "This clip has no detected downbeats", 4000
                )
            return
        self._engage_loop(region[0], region[1], intervals)

    def _engage_loop(
        self, start: float, end: float, intervals: int | None = None
    ) -> None:
        if not self.engine.set_loop(start, end):
            self.statusBar().showMessage("Could not set the loop region", 4000)
            return
        bounds = self.engine.loop
        if bounds is not None:
            position = self.engine.position
            if position < bounds[0] or position >= bounds[1]:
                self.engine.seek(bounds[0])
        self._loop_bounds = (float(start), float(end))
        for timeline in self._timelines():
            timeline.set_loop_region(start, end)
        self._set_loop_button(True)
        if intervals is None:
            self.statusBar().showMessage("Loop on", 3000)
            return
        noun = "interval" if int(intervals) == 1 else "intervals"
        self.statusBar().showMessage(f"Looping {int(intervals)} downbeat {noun}", 4000)

    def _clear_loop(self) -> None:
        self.engine.clear_loop()
        self._loop_bounds = None
        for timeline in self._timelines():
            timeline.set_loop_region(None, None)
        self._set_loop_button(False)
        self.statusBar().showMessage("Loop off", 3000)

    def _toggle_loop(self) -> None:
        if self.engine.loop is not None:
            self.engine.clear_loop()
            for timeline in self._timelines():
                timeline.set_loop_region(None, None)
            self._set_loop_button(False)
            self.statusBar().showMessage("Loop off", 3000)
            return
        if self._loop_bounds is None:
            self._set_loop_button(False)
            self.statusBar().showMessage("Use Ctrl+Alt+1–4 to loop downbeats", 4000)
            return
        start, end = self._loop_bounds
        self._engage_loop(start, end)

    def _set_loop_button(self, enabled: bool) -> None:
        self.loop_btn.blockSignals(True)
        self.loop_btn.setChecked(bool(enabled))
        self.loop_btn.blockSignals(False)
        self.loop_btn.setProperty("looping", bool(enabled))
        self.loop_btn.setIcon(loop_icon("#D7B36A" if enabled else "#8B939F"))
        self.loop_btn.style().unpolish(self.loop_btn)
        self.loop_btn.style().polish(self.loop_btn)
        self.loop_btn.update()

    # -- tempo ------------------------------------------------------------

    @staticmethod
    def _stretch_of(track) -> float:
        """How much a track's audio is stretched in time from its file."""
        try:
            factor = float(track.get("stretch", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
        return factor if factor > 0 else 1.0

    def _song_tempo(self) -> float | None:
        """The song's tempo from its bar lines (placed or the drum
        downbeats), or None when the bars are a plain tempo grid."""
        lines = self._bar_lines()
        if len(lines) < 2:
            return None
        lengths = np.diff(np.asarray(lines, dtype=float))
        lengths = lengths[lengths > 0.05]
        if lengths.size == 0:
            return None
        beats_per_bar, beat_unit = parse_time_signature(self.signature.currentText())
        bar = float(np.median(lengths))
        return 60.0 * beats_per_bar * (4.0 / beat_unit) / bar

    def _sync_bpm_field(self) -> None:
        """Show the song's tempo when its bars come from the music."""
        tempo = self._song_tempo()
        if tempo is None:
            self.bpm.setToolTip("Tempo of the bar grid")
            return
        self.bpm.setToolTip(
            "Tempo of the song, from its downbeats. Type a new tempo to speed up or slow "
            "down every track without changing its pitch."
        )
        tempo = round(max(self.bpm.minimum(), min(self.bpm.maximum(), tempo)), 2)
        if abs(self.bpm.value() - tempo) >= 0.005 and not self._tempo_timer.isActive():
            self.bpm.blockSignals(True)
            self.bpm.setValue(tempo)
            self.bpm.blockSignals(False)
            self._last_bpm = tempo

    def _bpm_changed(self, value) -> None:
        if self._song_tempo() is None:
            self._update_grid(value)  # no music bars: the field is the grid tempo
            return
        # Bars follow the music: a new tempo stretches the song. Wait until
        # typing or clicking the arrows stops.
        self._tempo_timer.start()

    def _apply_typed_tempo(self) -> None:
        self.change_tempo(float(self.bpm.value()))

    def change_tempo(self, new_bpm: float) -> bool:
        """Speed up or slow down the whole song to ``new_bpm`` without
        changing pitch. Every track is time-stretched; positions, bars and
        beats scale with it. One undo step."""
        old_bpm = self._song_tempo()
        if old_bpm is None or new_bpm <= 0:
            return False
        ratio = old_bpm / float(new_bpm)  # time scale of this change
        if abs(ratio - 1.0) < 1e-4:
            self._sync_bpm_field()
            return False
        tracks = self.project.tracks
        sources = [track.get("_source_audio", track.get("_audio")) for track in tracks]
        factors = [self._stretch_of(track) * ratio for track in tracks]
        work = [(source, factor) for source, factor in zip(sources, factors)]

        def run(report):
            out = []
            for number, (source, factor) in enumerate(work):
                if source is None or len(source) <= 1:
                    out.append(source)
                    continue
                out.append(stretch_audio(source, factor, lambda part, n=number: report((n + part) / len(work))))
            return out

        was_playing = self.engine.playing
        if was_playing:
            self.engine.pause()
        position = float(self.engine.position)
        stretched = self._with_progress(f"Changing the tempo to {new_bpm:.2f} BPM…", run)
        self._remember_edit()
        for track, source, audio, factor in zip(tracks, sources, stretched, factors):
            if source is not None:
                track["_source_audio"] = source
                track["_audio"] = audio
            if abs(factor - 1.0) < 1e-6:
                track.pop("stretch", None)
            else:
                track["stretch"] = round(factor, 9)
            for key in ("start", "offset", "duration", "fade_in", "fade_out", "_alignment"):
                if key in track and track[key] is not None:
                    track[key] = float(track[key]) * ratio
            if track.get("downbeats"):
                track["downbeats"] = [[float(time) * ratio, position_] for time, position_ in track["downbeats"]]
        if self.project.bars:
            self.project.bars = [round(float(time) * ratio, 6) for time in self.project.bars]
        if self._loop_bounds is not None:
            start, end = self._loop_bounds
            self._loop_bounds = (start * ratio, end * ratio)
            if self.engine.loop is not None:
                self._engage_loop(*self._loop_bounds)
        self._canvas_positions = [value * ratio for value in self._canvas_positions]
        self.bpm.blockSignals(True)
        self.bpm.setValue(float(new_bpm))
        self.bpm.blockSignals(False)
        self._last_bpm = float(new_bpm)
        self._reload_engine()
        self.engine.seek(position * ratio)
        self._commit_edit()
        self._refresh_timeline()
        if was_playing:
            self._play(keep_focus=True)
        self.statusBar().showMessage(
            f"Tempo {old_bpm:.2f} → {new_bpm:.2f} BPM. Every track was stretched; pitch is unchanged", 6000
        )
        return True

    def _even_bars_at_tempo(self) -> None:
        """Replace the bar lines with an even grid at the BPM field's tempo,
        starting at the first bar line. The audio is not changed."""
        lines = self._bar_lines()
        first = float(lines[0]) if lines else 0.0
        beats_per_bar, beat_unit = parse_time_signature(self.signature.currentText())
        bar = 60.0 * beats_per_bar * (4.0 / beat_unit) / max(1.0, float(self.bpm.value()))
        end = max(
            [float(t.get("start", 0.0) or 0.0) + float(t.get("duration", 0.0) or 0.0) for t in self.project.tracks]
            + [first + bar]
        )
        count = int(np.ceil((end - first) / bar)) + 1
        self._set_bars([first + bar * index for index in range(count)], f"Even bars at {self.bpm.value():.2f} BPM")

    def _with_progress(self, label: str, work):
        """Run ``work(report)`` on a worker thread while a progress dialog
        keeps the window responsive. ``report(fraction)`` may be called from
        the worker. Returns ``work``'s result or raises its exception."""
        import threading

        from PySide6.QtWidgets import QProgressDialog

        state = {"fraction": 0.0, "done": False, "result": None, "error": None}

        def report(fraction):
            state["fraction"] = max(state["fraction"], min(1.0, float(fraction)))

        def target():
            try:
                state["result"] = work(report)
            except Exception as exc:  # shown to the caller
                state["error"] = exc
            finally:
                state["done"] = True

        dialog = QProgressDialog(label, None, 0, 1000, self)
        dialog.setWindowTitle("SonicLab")
        dialog.setMinimumDuration(400)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        worker = threading.Thread(target=target, name="soniclab-tempo", daemon=True)
        worker.start()
        while not state["done"]:
            dialog.setValue(int(state["fraction"] * 1000))
            QApplication.processEvents()
            worker.join(0.02)
        dialog.setValue(1000)
        dialog.close()
        if state["error"] is not None:
            raise state["error"]
        return state["result"]

    def _update_grid(self, value):
        first = self._gesture_before is None and not self._history_frozen
        self._remember_edit()
        if first and self._gesture_before is not None:
            self._gesture_before["bpm"] = self._last_bpm
        self._last_bpm = float(value)
        self._refresh_timeline()
        snap = "snap beat" if self.timeline.snap_to_grid else "snap off"
        self.statusBar().showMessage(f"Tempo {float(value):.2f} BPM · {snap}", 3000)

    def _select_track(self, index: int):
        if index < 0 or index >= len(self.project.tracks):
            self._on_track_selected(-1)
            return
        view = self._view_for(index)
        self._clear_selection(view.peer)
        view.select_index(index)

    @staticmethod
    def _clear_selection(view) -> None:
        """Deselect everything in ``view`` without reporting "no selection"."""
        if view is None:
            return
        view.scene_obj.blockSignals(True)
        view.scene_obj.clearSelection()
        view.scene_obj.blockSignals(False)
        view.viewport().update()

    def _on_view_selected(self, view, index: int) -> None:
        if index < 0 and view.peer is not None and view.peer.selected_indices():
            return  # the selection moved to the other canvas
        if index >= 0:
            self._clear_selection(view.peer)
        self._on_track_selected(index)

    def _on_track_selected(self, index: int):
        self._selected = index
        self._show_selection()
        self._sync_clip_actions()

    def _show_selection(self):
        index = self._selected
        if 0 <= index < len(self.project.tracks):
            self.inspector.show_track(index, self.project.tracks[index])
        else:
            self.inspector.show_track(-1, None)
        for headers in (self.headers, self.headers2):
            headers.set_selected(self._lane_primary(index))
        self.mixer.set_selected(index)

    def _sync_clip_actions(self):
        count = len(self.project.tracks)
        has = 0 <= self._selected < count
        self.rename_action.setEnabled(has)
        self.delete_action.setEnabled(has)
        self.move_start_action.setEnabled(has)
        self.split_action.setEnabled(has)
        self.duplicate_action.setEnabled(has)
        self.analyze_action.setEnabled(has)
        self.onset_action.setEnabled(has)
        self.reference_action.setEnabled(has)
        self.nudge_action.setEnabled(has)
        self.downbeat_menu.setEnabled(has)
        has_events = has and bool(
            self._track_events(self.project.tracks[self._selected])
        )
        self.apply_downbeats_action.setEnabled(has_events and count > 1)
        self.export_downbeats_action.setEnabled(has_events)
        self.import_downbeats_action.setEnabled(has)
        self.clear_downbeats_action.setEnabled(
            has and bool(self.project.tracks[self._selected].get("downbeats"))
        )
        self.drum_track_action.setEnabled(has)
        self.drum_track_action.setChecked(
            has and bool(self.project.tracks[self._selected].get("drum_reference"))
        )
        self.move_up_action.setEnabled(has and self._selected > 0)
        self.move_down_action.setEnabled(has and self._selected < count - 1)
        self.mute_action.setEnabled(has)
        self.solo_action.setEnabled(has)
        if not has:
            return
        track = self.project.tracks[self._selected]
        self.mute_action.blockSignals(True)
        self.solo_action.blockSignals(True)
        self.mute_action.setChecked(bool(track.get("muted", False)))
        self.solo_action.setChecked(bool(track.get("solo", False)))
        self.mute_action.blockSignals(False)
        self.solo_action.blockSignals(False)

    def _clip_at_index(self, index):
        for item in self._view_for(index).scene_obj.items():
            if isinstance(item, ClipItem) and item.clip_index == index:
                return item
        return None

    def _set_gain(self, index, gain):
        gain = max(0.0, min(1.5, float(gain)))
        if index < 0:
            self._remember_edit()
            self.engine.set_output_gain(gain)
            self.mixer.set_master_gain(gain)
            return
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        self.project.tracks[index]["gain"] = gain
        self.engine.set_clip_gain(index, gain)
        for headers in (self.headers, self.headers2):
            headers.set_gain(index, gain)
        self.mixer.set_gain(index, gain)
        if index == self._selected:
            self.inspector.set_gain(gain)

    def _set_pan(self, index, pan):
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        pan = max(-1.0, min(1.0, float(pan)))
        self.project.tracks[index]["pan"] = pan
        self.engine.set_clip_pan(index, pan)
        for headers in (self.headers, self.headers2):
            headers.set_pan(index, pan)
        self.mixer.set_pan(index, pan)
        if index == self._selected:
            self.inspector.set_pan(pan)

    def _set_muted(self, index, muted):
        muted = bool(muted)
        if index < 0:
            self._remember_edit()
            self.engine.set_master_muted(muted)
            self.mixer.set_master_muted(muted)
            self._commit_edit()
            return
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        self.project.tracks[index]["muted"] = muted
        self._push_audible(index, self.project.tracks[index])
        self._push_stem_mutes()
        for headers in (self.headers, self.headers2):
            headers.set_muted(index, muted)
        self.mixer.set_muted(index, muted)
        if index == self._selected:
            self.inspector.set_muted(muted)
            self._sync_clip_actions()
        self._commit_edit()

    def _set_solo(self, index, solo):
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        solo = bool(solo)
        self.project.tracks[index]["solo"] = solo
        self._push_audible(index, self.project.tracks[index])
        for headers in (self.headers, self.headers2):
            headers.set_solo(index, solo)
        self.mixer.set_solo(index, solo)
        if index == self._selected:
            self.inspector.set_solo(solo)
            self._sync_clip_actions()
        self._commit_edit()

    def _mute_toggled(self, checked):
        self._set_muted(self._selected, checked)

    def _solo_toggled(self, checked):
        self._set_solo(self._selected, checked)

    def _set_start(self, seconds):
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        seconds = max(0.0, float(seconds))
        self._remember_edit()
        clip = self._clip_at_index(index)
        if clip is not None:
            clip.setPos(
                HEADER_WIDTH + seconds * self.timeline.pixels_per_second,
                clip.pos().y(),
            )
        else:
            self.project.tracks[index]["start"] = seconds
            self.engine.set_clip_start(index, seconds)
        self._commit_edit()
        if index == self._selected:
            self.inspector.set_start(self.project.tracks[index]["start"])

    def _move_selection_to_start(self):
        self._set_start(0.0)

    def _rename_track(self):
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        current = self.project.tracks[index].get("name", "")
        name, ok = QInputDialog.getText(
            self, "Rename track", "Track name:", text=current
        )
        if ok:
            self._rename_track_to(name)

    def _rename_track_to(self, name: str):
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        cleaned = name.strip()
        if not cleaned:
            return
        self._remember_edit()
        self.project.tracks[index]["name"] = cleaned
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(index)

    def _delete_track(self):
        indices = self._selection()
        if not indices:
            return
        if len(indices) == 1:
            name = self.project.tracks[indices[0]].get("name", "track")
            prompt = f"Delete “{name}”?"
        else:
            prompt = f"Delete {len(indices)} tracks?"
        answer = QMessageBox.question(
            self,
            "Delete track",
            f"{prompt} You can undo this.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._remember_edit()
        for index in sorted(indices, reverse=True):
            del self.project.tracks[index]
        self._selected = -1
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()

    def _on_track_resized(self, index, height):
        QTimer.singleShot(0, lambda: self._apply_row_height(index, height))

    def _apply_row_height(self, index, height):
        for timeline in self._timelines():
            timeline.set_row_height(index, height)
        self._refresh_timeline()
        if 0 <= index < len(self.project.tracks):
            self._select_track(index)

    def _sync_from_timeline(self, value, pair=None):
        if self._scroll_lock:
            return
        headers, _timeline = pair or (self.headers, self.timeline)
        self._scroll_lock = True
        headers.sync(value)
        self._scroll_lock = False

    def _sync_from_headers(self, value, pair=None):
        if self._scroll_lock:
            return
        _headers, timeline = pair or (self.headers, self.timeline)
        self._scroll_lock = True
        timeline.verticalScrollBar().setValue(value)
        self._scroll_lock = False

    def _each_timeline(self, action) -> None:
        for timeline in self._timelines():
            action(timeline)

    def _selected_indices(self) -> list[int]:
        """Selected tracks on both canvases."""
        return sorted(set(self.timeline.selected_indices()) | set(self.timeline2.selected_indices()))

    def _toggle_second_canvas(self, visible: bool) -> None:
        if not visible and any(track_canvas(track) == 1 for track in self.project.tracks):
            # Never hide tracks that are playing in the song.
            self.second_canvas_action.blockSignals(True)
            self.second_canvas_action.setChecked(True)
            self.second_canvas_action.blockSignals(False)
            self.statusBar().showMessage("Move the second canvas's tracks back before hiding it", 4000)
            return
        self.splitter2.setVisible(bool(visible))
        if visible:
            self._size_canvases()
            self._refresh_timeline()

    def _set_canvas_layout(self, side_by_side: bool) -> None:
        """Second canvas below the main one, or to its right."""
        side_by_side = bool(side_by_side)
        self.canvas_side_by_side = side_by_side
        self.canvas_split.setOrientation(
            Qt.Orientation.Horizontal if side_by_side else Qt.Orientation.Vertical
        )
        (self.canvas_side_action if side_by_side else self.canvas_stacked_action).setChecked(True)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QSettings("SonicLab", "SonicLab").setValue("view/canvasSideBySide", side_by_side)
        if not side_by_side:
            # Stacked again: line the two time axes back up.
            self._follow_scroll(self.timeline2, self.timeline.horizontalScrollBar().value())
        if self.splitter2.isVisible():
            self._size_canvases()
            if side_by_side:
                # Scrolling on its own now: start where its first clip is.
                first = next(
                    (i for i, track in enumerate(self.project.tracks) if track_canvas(track) == 1), None
                )
                if first is not None:
                    self._reveal_track(first)
        self.statusBar().showMessage(
            "Canvases side by side" if side_by_side else "Canvases stacked", 2500
        )

    def _size_canvases(self) -> None:
        """Side by side: half each, with narrow track headers. Stacked: the
        second canvas is as tall as its rows, so it has no empty space."""
        narrow = 150 if self.canvas_side_by_side else 208
        for headers, splitter in ((self.headers, self.splitter), (self.headers2, self.splitter2)):
            headers.setMinimumWidth(narrow)
            total = max(2, splitter.width())
            first = narrow if self.canvas_side_by_side else max(narrow, splitter.sizes()[0])
            splitter.setSizes([first, max(1, total - first)])
        if self.canvas_side_by_side:
            total = max(2, self.canvas_split.width())
            self.canvas_split.setSizes([total // 2, total - total // 2])
        else:
            total = max(2, self.canvas_split.height())
            wanted = self._second_canvas_height()
            self.canvas_split.setSizes([max(1, total - wanted), wanted])
            self.splitter2.setSizes(self.splitter.sizes())
        self._sized_rows = self._second_canvas_rows()

    def _second_canvas_rows(self) -> int:
        return len(self.timeline2.lane_rows(self.project.tracks)[1])

    def _second_canvas_height(self) -> int:
        """Ruler, rows, one empty row and the scroll bar of the second canvas;
        never more than half the space."""
        view = self.timeline2
        _rows, primaries = view.lane_rows(self.project.tracks)
        rows = view._display_top(len(primaries), primaries) if primaries else RULER_HEIGHT
        # One empty row below the last: room to start a selection box and
        # to drop clips onto a new row.
        rows += TRACK_HEIGHT
        scroll = view.horizontalScrollBar().sizeHint().height()
        return int(min(max(view.minimumHeight(), rows + scroll + 8), self.canvas_split.height() // 2))

    def _reveal_track(self, index: int) -> None:
        """Scroll the canvas that holds track ``index`` until its clip shows."""
        def reveal():
            view = self._view_for(index)
            for item in view.scene_obj.items():
                if isinstance(item, ClipItem) and isValid(item) and item.clip_index == index:
                    clip = item.sceneBoundingRect()
                    shown = view.mapToScene(view.viewport().rect()).boundingRect()
                    if shown.intersects(clip):
                        return  # already on screen: leave the view alone
                    # Bring its start into view (a long clip can be wider than the view).
                    width = min(clip.width(), shown.width() / 2)
                    view.ensureVisible(clip.x(), clip.y(), max(1.0, width), clip.height(), 40, 10)
                    return
        QTimer.singleShot(0, reveal)

    def _follow_scroll(self, other, value: int) -> None:
        """Keep the stacked canvases on the same stretch of time."""
        if self.canvas_side_by_side or other.horizontalScrollBar().value() == value:
            return
        if self.engine.playing and self._canvas_focus is not None and other.canvas != self._canvas_focus:
            return  # one canvas plays alone: the other one stays where it is
        other.horizontalScrollBar().setValue(value)

    def _match_header_widths(self, source, other) -> None:
        if not self.canvas_side_by_side:
            other.setSizes(source.sizes())

    def _move_track_to_canvas(self, index: int, canvas: int, start: float | None = None) -> None:
        """Put a whole track (and any track sharing its row) on ``canvas``."""
        ui_trace.mark(f"move track {index} to canvas {canvas}")
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return
        lane = tracks[index].get("lane")
        group = [i for i, track in enumerate(tracks) if i == index or (lane is not None and track.get("lane") == lane)]
        shift = 0.0 if start is None else max(0.0, float(start)) - float(tracks[index].get("start", 0.0) or 0.0)
        self._remember_edit()
        for i in group:
            track = tracks[i]
            if canvas:
                track["canvas"] = int(canvas)
            else:
                track.pop("canvas", None)
            track["start"] = max(0.0, float(track.get("start", 0.0) or 0.0) + shift)
            self.engine.set_clip_start(i, track["start"])
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(index)
        where = "the second canvas" if canvas else "the main canvas"
        # Neither canvas scrolls: a transfer never moves the view under the user.
        self.statusBar().showMessage(f"Moved “{tracks[index].get('name', 'Track')}” to {where}", 3000)

    def _move_track_to_row(self, index: int, view, display_row: int) -> None:
        """Put one clip on another row of its canvas: it joins the clips on
        that row, or gets a row of its own past the last one. Its time stays
        where the drag left it. One undo step with the drag."""
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return
        rows, primaries = view.lane_rows(tracks)
        track = tracks[index]
        self._remember_edit()
        if 0 <= display_row < len(primaries):
            new_index = self._join_row(track, tracks[primaries[display_row]])
            where = f"row {display_row + 1}"
        else:
            track.pop("lane", None)
            tracks.pop(index)
            tracks.append(track)
            new_index = len(tracks) - 1
            where = "a new row"
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(new_index)
        self.statusBar().showMessage(f"Moved “{track.get('name', 'Track')}” to {where}", 3000)

    def _drop_on_canvas(self, source, indices, grabbed: int, shift: float, target_row: int) -> None:
        """Clips dragged from ``source`` onto the other canvas. They move in
        time by ``shift``; the grabbed clip lands on ``target_row`` there and
        the others keep their rows relative to it. A row that exists there is
        joined; past the last row (or above the first) new rows are made.
        Only the dragged clips move, never the rest of their rows."""
        target = source.peer
        tracks = self.project.tracks
        if target is None or not (0 <= grabbed < len(tracks)):
            return
        indices = [i for i in indices if 0 <= i < len(tracks)]
        source_rows, _ = source.lane_rows(tracks)
        _, target_primaries = target.lane_rows(tracks)
        hosts = [tracks[i] for i in target_primaries]  # by object: indices change below
        moving = [tracks[i] for i in indices]
        rows = {id(track): source_rows[i] - source_rows[grabbed] + target_row for i, track in zip(indices, moving)}
        canvas = target.canvas
        if canvas == 1 and not self.splitter2.isVisible():
            self._toggle_second_canvas(True)
        self._remember_edit()
        new_lanes: dict[int, int] = {}
        for track in moving:
            track["start"] = max(0.0, float(track.get("start", 0.0) or 0.0) + float(shift))
            row = rows[id(track)]
            if 0 <= row < len(hosts) and hosts[row] is not track:
                self._join_row(track, hosts[row])
                continue
            # A new row: clips that land on the same new row share it.
            if row not in new_lanes:
                new_lanes[row] = self._lane_ids
                self._lane_ids += 1
            track["lane"] = new_lanes[row]
            if canvas:
                track["canvas"] = canvas
            else:
                track.pop("canvas", None)
            tracks.pop(next(i for i, other in enumerate(tracks) if other is track))
            tracks.append(track)
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        grabbed_track = moving[indices.index(grabbed)] if grabbed in indices else None
        if grabbed_track is not None:
            index = next(i for i, other in enumerate(tracks) if other is grabbed_track)
            self._select_track(index)  # dropped where the user can see it: no scroll
        where = "the second canvas" if canvas else "the main canvas"
        count = len(moving)
        self.statusBar().showMessage(
            f"Moved {count} clip{'s' if count != 1 else ''} to {where}", 3000
        )

    def _move_blocks(self, source, blocks, grabbed: int, shift: float, canvas: int, target_row: int) -> None:
        ui_trace.mark(f"move {len(blocks)} block(s) to canvas {canvas}")
        """Picked bars dragged together. Each bar is cut out of its track as
        a piece of its own (a piece that already is one bar just moves),
        moved by ``shift`` seconds onto ``canvas``. The grabbed bar lands on
        ``target_row`` there; the others keep their rows relative to it,
        joining rows that exist and making new ones past the last. One undo."""
        tracks = self.project.tracks
        blocks = [b for b in blocks if 0 <= int(b[0]) < len(tracks)]
        if not blocks or not (0 <= grabbed < len(blocks)):
            return
        target = source if source.canvas == canvas else source.peer
        if target is None:
            return
        if canvas == 1 and not self.splitter2.isVisible():
            self._toggle_second_canvas(True)
        source_rows, _ = source.lane_rows(tracks)
        _, target_primaries = target.lane_rows(tracks)
        hosts = [tracks[i] for i in target_primaries]
        base = source_rows[int(blocks[grabbed][0])]
        grouped: dict[int, list] = {}
        for index, local_start, local_duration, bar in blocks:
            grouped.setdefault(int(index), []).append((float(local_start), float(local_duration), int(bar)))
        self._remember_edit()
        pieces = []  # (piece, relative row)
        replaced = {}  # id(track) -> remaining segments
        for index, items in grouped.items():
            track = tracks[index]
            relative = source_rows[index] - base
            start = float(track.get("start", 0.0) or 0.0)
            offset = float(track.get("offset", 0.0) or 0.0)
            duration = float(track.get("duration", 0.0) or 0.0)
            items.sort()
            if len(items) == 1 and items[0][0] <= 0.05 and items[0][1] >= duration - 0.05:
                pieces.append((track, relative, True))  # the whole clip is this bar
                continue
            for local_start, local_duration, bar in items:
                piece = dict(track)
                piece["start"] = start + local_start
                piece["offset"] = offset + local_start
                piece["duration"] = local_duration
                piece["armed"] = False
                piece["bar_block"] = True
                piece["name"] = f"{track.get('name', 'Audio')} · bar {bar + 1}"
                piece.pop("lane", None)
                pieces.append((piece, relative, False))
            # What is left of the track around the bars taken out.
            segments = []
            cursor = 0.0
            for local_start, local_duration, _bar in items:
                if local_start - cursor >= 0.05:
                    segments.append((cursor, local_start - cursor))
                cursor = max(cursor, local_start + local_duration)
            if duration - cursor >= 0.05:
                segments.append((cursor, duration - cursor))
            replaced[id(track)] = (track, segments, start, offset)
        # Rebuild the track list with the remaining segments in place.
        rebuilt = []
        for track in tracks:
            entry = replaced.get(id(track))
            if entry is None:
                rebuilt.append(track)
                continue
            track, segments, start, offset = entry
            if not segments:
                continue  # every bar of it was taken
            lane = track.get("lane")
            if len(segments) > 1 and lane is None:
                lane = self._lane_ids
                self._lane_ids += 1
            for number, (local_start, length) in enumerate(segments):
                part = track if number == 0 else dict(track)
                part["start"] = start + local_start
                part["offset"] = offset + local_start
                part["duration"] = length
                if lane is not None:
                    part["lane"] = lane
                if number:
                    part["armed"] = False
                rebuilt.append(part)
        tracks[:] = rebuilt
        new_lanes: dict[int, int] = {}
        moved = []
        for piece, relative, whole in pieces:
            piece["start"] = max(0.0, float(piece["start"]) + float(shift))
            if not whole:
                tracks.append(piece)
            row = target_row + relative
            if 0 <= row < len(hosts) and hosts[row] is not piece and any(h is hosts[row] for h in tracks):
                self._join_row(piece, hosts[row])
            else:
                if row not in new_lanes:
                    new_lanes[row] = self._lane_ids
                    self._lane_ids += 1
                piece["lane"] = new_lanes[row]
                if canvas:
                    piece["canvas"] = canvas
                else:
                    piece.pop("canvas", None)
                tracks.pop(next(i for i, other in enumerate(tracks) if other is piece))
                tracks.append(piece)
            moved.append(piece)
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        indices = {next(i for i, other in enumerate(tracks) if other is piece) for piece in moved}
        for view in self._timelines():
            view.set_picks({})
        target.pick_whole_clips(indices)
        first = min(indices)
        self._select_track(first)  # dropped where the user can see it: no scroll
        count = len(moved)
        where = "canvas 2" if canvas else "the main canvas"
        self.statusBar().showMessage(f"Moved {count} bar{'s' if count != 1 else ''} on {where}", 3000)

    def _join_row(self, track: dict, host: dict) -> int:
        """Put ``track`` on ``host``'s row (they share a lane), just after the
        row's last clip so the row keeps its place. Returns its new index."""
        tracks = self.project.tracks
        lane = host.get("lane")
        if lane is None:
            lane = self._lane_ids
            self._lane_ids += 1
            host["lane"] = lane
        track["lane"] = lane
        if track_canvas(host):
            track["canvas"] = track_canvas(host)
        else:
            track.pop("canvas", None)
        # By identity: track dicts hold numpy arrays, so == would not work.
        tracks.pop(next(i for i, other in enumerate(tracks) if other is track))
        members = [i for i, other in enumerate(tracks) if other.get("lane") == lane]
        new_index = (members[-1] + 1) if members else len(tracks)
        tracks.insert(new_index, track)
        return new_index

    def _move_selected_to_other_canvas(self) -> None:
        """The selection to the other canvas, at the same time, on new rows
        past its last one (keeping their rows relative to each other):
        picked blocks if any are picked (cut out of their tracks), else
        every selected clip, else the selected track."""
        tracks = self.project.tracks
        for view in self._timelines():
            blocks = view.picked_blocks()
            if not blocks:
                continue
            rows, _ = view.lane_rows(tracks)
            grabbed = min(range(len(blocks)), key=lambda n: (rows[blocks[n][0]], blocks[n][1]))
            target = 0 if view.canvas == 1 else 1
            if target == 1 and not self.splitter2.isVisible():
                self._toggle_second_canvas(True)
            other = view.peer
            _, primaries = other.lane_rows(tracks)
            self._move_blocks(view, [item[:4] for item in blocks], grabbed, 0.0, target, len(primaries))
            return
        for view in self._timelines():
            indices = [i for i in view.selected_indices() if 0 <= i < len(tracks)]
            if len(indices) < 2:
                continue
            if view.canvas == 0 and not self.splitter2.isVisible():
                self._toggle_second_canvas(True)
            rows, _ = view.lane_rows(tracks)
            grabbed = min(indices, key=lambda i: rows[i])
            _, primaries = view.peer.lane_rows(tracks)
            self._drop_on_canvas(view, indices, grabbed, 0.0, len(primaries))
            return
        index = self._selected
        if not (0 <= index < len(tracks)):
            return
        target = 0 if track_canvas(tracks[index]) == 1 else 1
        if target == 1:
            self._toggle_second_canvas(True)
        self._move_track_to_canvas(index, target)

    def _clip_context_menu(self, pos, view=None):
        view = view or self.timeline
        bar = view.bar_under(pos)
        clip = view.clip_at(pos)
        menu = QMenu(self)
        block = view.block_at(pos)
        states = self.engine.previews()
        if block is not None:
            key = self._voice_key(*block)
            state = states.get(key)
            playing = state is not None and state[0]
            if key[1] < 0:
                play = f"{'Pause' if playing else 'Play'} This Stem in Sync"
            else:
                play = f"{'Pause' if playing else 'Play'} This Block on Its Own"
            menu.addAction(play, lambda block=block: self._toggle_block(*block))
            if state is not None:
                stop = "Stop This Stem" if key[1] < 0 else "Stop This Block"
                menu.addAction(stop, lambda key=key: self._stop_block(key))
        if any(view.picks() for view in self._timelines()):
            menu.addAction("Play Picked Blocks Together", self._play_picked_blocks)
            menu.addAction("Loop Picked Stems in Sync", self._loop_picked_stems)
            menu.addAction("Move Picked Blocks to Other Canvas", self._move_selected_to_other_canvas)
        if block is not None and stem_loop(self.project.tracks[block[0]]) is not None:
            menu.addAction("Remove This Stem from the Synced Loop", lambda index=block[0]: self._remove_from_stem_loop(index))
            menu.addAction("Clear the Synced Loop", lambda index=block[0]: self._clear_stem_loop(index))
        if states:
            menu.addAction("Stop All Blocks", self._stop_blocks)
        if menu.actions():
            menu.addSeparator()
        if bar >= 0:
            self._add_bar_actions(menu, bar)
        if clip is None:
            if menu.actions():
                menu.exec(view.viewport().mapToGlobal(pos))
            return
        if menu.actions():
            menu.addSeparator()
        self._select_track(clip.clip_index)
        menu.addAction(self.rename_action)
        menu.addAction(self.move_start_action)
        menu.addAction(self.mute_action)
        menu.addAction(self.solo_action)
        menu.addSeparator()
        menu.addAction(self.split_action)
        menu.addAction(self.duplicate_action)
        menu.addAction(self.analyze_action)
        menu.addAction(self.onset_action)
        menu.addAction(self.reference_action)
        menu.addAction(self.nudge_action)
        self._sync_clip_actions()
        menu.addMenu(self.downbeat_menu)
        menu.addAction(self.canvas_move_action)
        menu.addAction(self.move_up_action)
        menu.addAction(self.move_down_action)
        menu.addSeparator()
        menu.addAction(self.delete_action)
        menu.exec(view.viewport().mapToGlobal(pos))

    def _new_project(self):
        if self.project.tracks:
            answer = QMessageBox.question(
                self,
                "New session",
                "Close the current session and start a new one?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._stop()
        self._remember_edit()
        self.engine.clear_clips()
        self.project = Project()
        self.current_path = None
        self._selected = -1
        for timeline in self._timelines():
            timeline.pixels_per_second = float(PIXELS_PER_SECOND)
            timeline.track_height = TRACK_HEIGHT
            timeline._row_heights = []
        self.bpm.blockSignals(True)
        self.signature.blockSignals(True)
        self.bpm.setValue(120)
        self.signature.setCurrentText("4/4")
        self.bpm.blockSignals(False)
        self.signature.blockSignals(False)
        self._last_bpm = 120.0
        self._last_signature = "4/4"
        self._commit_edit()
        self._refresh_timeline()
        self._set_session_title()
        self.statusBar().showMessage("New session", 3000)

    def _save_project(self):
        path = self.current_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save project",
                "Untitled.soniclab",
                "SonicLab project (*.soniclab)",
            )
        if not path:
            return
        serial_tracks = [
            {key: value for key, value in track.items() if not key.startswith("_")}
            for track in self.project.tracks
        ]
        numerator, denominator = parse_time_signature(self.signature.currentText())
        saved_project = Project()
        saved_project.name = self.project.name
        saved_project.bpm = float(self.bpm.value())
        saved_project.time_signature = [numerator, denominator]
        saved_project.bars = list(self.project.bars)
        saved_project.tracks = serial_tracks
        self.project.bpm = saved_project.bpm
        self.project.time_signature = saved_project.time_signature
        try:
            saved_project.save(path)
            self.current_path = path
            self._remember_path(path)
            self._set_session_title()
            self._refresh_browser()
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _save_project_as(self):
        previous = self.current_path
        self.current_path = None
        try:
            self._save_project()
        finally:
            if self.current_path is None:
                self.current_path = previous

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "SonicLab project (*.soniclab *.json)"
        )
        if path:
            self._load_project(path)

    def _open_recent(self, path: str):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing project", f"Could not find\n{path}")
            return
        self._load_project(path)

    def _load_project(self, path: str):
        self._remember_edit()
        try:
            self.engine.stop()
            self.engine.preview_stop_all()  # its voices point at the old clips
            self._preview_tracks.clear()
            self.engine.clear_clips()
            self.project = Project.load(path)
            for track in self.project.tracks:
                source = track.get("path")
                if source and os.path.exists(source):
                    audio, sr = sf.read(source, dtype="float32", always_2d=True)
                    audio = AudioEngine.resample(audio, sr)
                else:
                    audio = np.zeros((1, 2), dtype=np.float32)
                factor = self._stretch_of(track)
                if factor != 1.0 and len(audio) > 1:
                    # Saved at another tempo than the file's: stretch it again.
                    track["_source_audio"] = audio
                    audio = self._with_progress(
                        f"Setting the tempo of “{track.get('name', 'Track')}”…",
                        lambda report, audio=audio, factor=factor: stretch_audio(audio, factor, report),
                    )
                track["_audio"] = audio
                if len(audio) > 1:
                    file_duration = len(audio) / OUTPUT_RATE
                    offset = max(0.0, float(track.get("offset", 0.0) or 0.0))
                    offset = min(offset, max(0.0, file_duration - 0.01))
                    stored = track.get("duration")
                    duration = (
                        max(0.01, file_duration - offset)
                        if stored is None
                        else min(float(stored), max(0.01, file_duration - offset))
                    )
                    track["offset"] = offset
                    track["duration"] = duration
                else:
                    track["duration"] = float(track.get("duration", 30) or 0.0)
                index = self.engine.add_clip(audio, track.get("start", 0))
                self._push_mix(index, track)
            self.current_path = path
            self._selected = 0 if self.project.tracks else -1
            numerator, denominator = parse_time_signature(self.project.time_signature)
            signature = f"{numerator}/{denominator}"
            self.bpm.blockSignals(True)
            self.signature.blockSignals(True)
            self.bpm.setValue(float(self.project.bpm))
            if self.signature.findText(signature) < 0:
                self.signature.addItem(signature)
            self.signature.setCurrentText(signature)
            self.bpm.blockSignals(False)
            self.signature.blockSignals(False)
            self._remember_path(path)
            self._refresh_timeline()
            if self._selected >= 0:
                self._select_track(self._selected)
            self._set_session_title()
            self._last_bpm = float(self.bpm.value())
            self._last_signature = self.signature.currentText()
            self._commit_edit()
            self._queue_beat_sources(track.get("path") for track in self.project.tracks)
            self.statusBar().showMessage(f"Opened {os.path.basename(path)}", 4000)
        except Exception as exc:
            self._gesture_before = None
            QMessageBox.critical(self, "Project error", str(exc))

    def _recent_paths(self):
        value = QSettings("SonicLab", "SonicLab").value("recentProjects", [])
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _remember_path(self, path: str):
        path = os.path.abspath(path)
        recent = [path] + [item for item in self._recent_paths() if item != path]
        QSettings("SonicLab", "SonicLab").setValue("recentProjects", recent[:10])

    def _fill_recent_menu(self, menu):
        menu.clear()
        paths = self._recent_paths()
        if not paths:
            empty = QAction("No recent projects", self)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        for path in paths:
            label = os.path.basename(path)
            if not os.path.exists(path):
                label += "  (missing)"
            action = QAction(label, self)
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, item=path: self._open_recent(item)
            )
            menu.addAction(action)

    def _selection(self) -> list[int]:
        indices = [
            index
            for index in self._selected_indices()
            if 0 <= index < len(self.project.tracks)
        ]
        if indices:
            return indices
        if 0 <= self._selected < len(self.project.tracks):
            return [self._selected]
        return []

    def _snapshot(self) -> dict:
        return {
            "tracks": [dict(track) for track in self.project.tracks],
            "master_gain": self.engine.output_gain,
            "master_muted": self.engine.master_muted,
            "bpm": float(self.bpm.value()),
            "signature": self.signature.currentText(),
            "name": self.project.name,
            "bars": list(self.project.bars),
        }

    def _snapshot_changed(self, before: dict, after: dict) -> bool:
        if (
            before["master_gain"] != after["master_gain"]
            or before["master_muted"] != after["master_muted"]
            or before["bpm"] != after["bpm"]
            or before["signature"] != after["signature"]
            or before["name"] != after["name"]
            or before.get("bars") != after.get("bars")
            or len(before["tracks"]) != len(after["tracks"])
        ):
            return True
        keys = (
            "name",
            "path",
            "start",
            "duration",
            "offset",
            "gain",
            "pan",
            "muted",
            "solo",
            "fade_in",
            "fade_out",
            "clip_gain",
            "armed",
            "empty",
            "downbeats",
            "drum_reference",
            "lane",
            "canvas",
            "bar_block",
            "stretch",
            "stem_group",
            "stem_loop",
        )
        for left, right in zip(before["tracks"], after["tracks"]):
            for key in keys:
                if left.get(key) != right.get(key):
                    return True
        return False

    def _remember_edit(self) -> None:
        if self._history_frozen or self._gesture_before is not None:
            return
        self._gesture_before = self._snapshot()

    def _commit_edit(self) -> None:
        if self._history_frozen or self._gesture_before is None:
            return
        before = self._gesture_before
        self._gesture_before = None
        after = self._snapshot()
        if not self._snapshot_changed(before, after):
            return
        self.history.push(
            Edit(
                lambda: self._restore_snapshot(before),
                lambda: self._restore_snapshot(after),
            )
        )

    def _restore_snapshot(self, snap: dict) -> None:
        self._history_frozen = True
        self._gesture_before = None
        try:
            self.project.tracks = [dict(item) for item in snap["tracks"]]
            self.project.name = snap.get("name") or "Untitled"
            self.project.bars = list(snap.get("bars") or [])
            self.engine.set_output_gain(snap.get("master_gain", 1.0))
            self.engine.set_master_muted(snap.get("master_muted", False))
            self.mixer.set_master_gain(snap.get("master_gain", 1.0))
            self.mixer.set_master_muted(snap.get("master_muted", False))
            self.bpm.blockSignals(True)
            self.signature.blockSignals(True)
            self.bpm.setValue(float(snap.get("bpm", 120.0)))
            signature = snap.get("signature") or "4/4"
            if self.signature.findText(signature) < 0:
                self.signature.addItem(signature)
            self.signature.setCurrentText(signature)
            self.bpm.blockSignals(False)
            self.signature.blockSignals(False)
            self._last_bpm = float(self.bpm.value())
            self._last_signature = self.signature.currentText()
            self._reload_engine()
            if self._selected >= len(self.project.tracks):
                self._selected = len(self.project.tracks) - 1
            self._refresh_timeline()
            self._set_session_title()
            if self._selected >= 0:
                self._select_track(self._selected)
            else:
                self._on_track_selected(-1)
        finally:
            self._history_frozen = False

    def _undo(self) -> None:
        self.history.undo()

    def _redo(self) -> None:
        self.history.redo()

    def _sync_history(self) -> None:
        if not hasattr(self, "undo_action"):
            return
        self.undo_action.setEnabled(self.history.can_undo)
        self.redo_action.setEnabled(self.history.can_redo)

    def _signature_changed(self, text: str) -> None:
        first = self._gesture_before is None and not self._history_frozen
        self._remember_edit()
        if first and self._gesture_before is not None:
            self._gesture_before["signature"] = self._last_signature
        self._last_signature = text
        self._refresh_timeline()
        self._commit_edit()

    def _set_snap(self, checked: bool) -> None:
        for timeline in self._timelines():
            timeline.set_snap(bool(checked))
        self.statusBar().showMessage("Snap beat" if checked else "Snap off", 2000)

    def _clip_released(self, index: int) -> None:
        before = self._gesture_before
        if before is not None and 0 <= index < len(self.project.tracks):
            old_tracks = before["tracks"]
            if index < len(old_tracks):
                delta = float(self.project.tracks[index].get("start", 0.0)) - float(
                    old_tracks[index].get("start", 0.0)
                )
                others = [
                    item for item in self._selected_indices() if item != index
                ]
                if abs(delta) > 1e-6 and others:
                    for other in others:
                        if other >= len(self.project.tracks) or other >= len(
                            old_tracks
                        ):
                            continue
                        start = max(
                            0.0, float(old_tracks[other].get("start", 0.0)) + delta
                        )
                        self.project.tracks[other]["start"] = start
                        self.engine.set_clip_start(other, start)
                    self._refresh_timeline()
        self._commit_edit()

    def _trim_clip(self, index: int, start: float, duration: float) -> None:
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        track = self.project.tracks[index]
        old_start = float(track.get("start", 0.0))
        offset = max(
            0.0, float(track.get("offset", 0.0) or 0.0) + float(start) - old_start
        )
        audio = track.get("_audio")
        file_seconds = (
            len(audio) / OUTPUT_RATE
            if audio is not None and len(audio)
            else offset + float(duration)
        )
        offset = min(offset, max(0.0, file_seconds))
        duration = max(0.05, min(float(duration), max(0.05, file_seconds - offset)))
        track["start"] = max(0.0, float(start))
        track["offset"] = offset
        track["duration"] = duration
        self._push_mix(index, track)
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(index)

    def _move_block(
        self,
        index: int,
        local_start: float,
        local_duration: float,
        new_start: float,
        share_lane: bool,
        insert_at: int,
        bar: int,
        canvas: int | None = None,
        join: bool = False,
    ) -> None:
        """Move one bar of a track. The other bars stay where they are.

        ``canvas`` is set when the bar was dropped on the other canvas.
        ``join``: it was dropped on another track's row, and joins that row
        (``insert_at`` is the row's first track).
        """
        ui_trace.mark(f"move bar {bar + 1} of track {index}" + ("" if canvas is None else f" to canvas {canvas}"))
        host = (
            self.project.tracks[insert_at]
            if join and 0 <= insert_at < len(self.project.tracks) and insert_at != index
            else None
        )
        if not (0 <= index < len(self.project.tracks)):
            return
        track = self.project.tracks[index]
        start = float(track.get("start", 0.0))
        duration = float(track.get("duration", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        local_start = max(0.0, min(float(local_start), duration))
        local_end = max(local_start, min(local_start + float(local_duration), duration))
        if local_end - local_start < 0.05 or duration <= 0:
            return
        self._remember_edit()
        if local_end - local_start >= duration - 0.05:
            track["start"] = max(0.0, float(new_start))
            if canvas is not None:
                if canvas:
                    track["canvas"] = int(canvas)
                else:
                    track.pop("canvas", None)
                track.pop("lane", None)
            self.engine.set_clip_start(index, track["start"])
            self._commit_edit()
            self._refresh_timeline()
            self._select_track(index)
            return
        piece = dict(track)
        piece["start"] = max(0.0, float(new_start))
        piece["offset"] = offset + local_start
        piece["duration"] = local_end - local_start
        piece["armed"] = False
        # One block from now on, wherever it lands; see TimelineView._place_bar_blocks.
        piece["bar_block"] = True
        if canvas is not None:
            if canvas:
                piece["canvas"] = int(canvas)
            else:
                piece.pop("canvas", None)
        piece["name"] = f"{track.get('name', 'Audio')} · bar {int(bar) + 1}"
        head = local_start
        tail_length = duration - local_end
        tail = None
        if head >= 0.05 and tail_length >= 0.05:
            track["duration"] = head
            tail = dict(track)
            tail["start"] = start + local_end
            tail["offset"] = offset + local_end
            tail["duration"] = tail_length
            tail["armed"] = False
        elif head < 0.05:
            track["start"] = start + local_end
            track["offset"] = offset + local_end
            track["duration"] = tail_length
        else:
            track["duration"] = head
        lane = track.get("lane")
        if share_lane or tail is not None:
            if lane is None:
                lane = self._lane_ids
                self._lane_ids += 1
                track["lane"] = lane
            if tail is not None:
                tail["lane"] = lane
            if share_lane:
                piece["lane"] = lane
            else:
                piece.pop("lane", None)
        else:
            piece.pop("lane", None)
        if share_lane:
            self.project.tracks.insert(index + 1, piece)
            piece_index = index + 1
            if tail is not None:
                self.project.tracks.insert(index + 2, tail)
        else:
            if tail is not None:
                self.project.tracks.insert(index + 1, tail)
                if insert_at > index:
                    insert_at += 1
            insert_at = max(0, min(int(insert_at), len(self.project.tracks)))
            if insert_at == index:
                insert_at = index + 1
            self.project.tracks.insert(insert_at, piece)
            piece_index = insert_at
        if host is not None:
            piece_index = self._join_row(piece, host)
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(piece_index)
        self.statusBar().showMessage(f"Moved bar {int(bar) + 1}", 2000)

    def _split_selected(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        track = self.project.tracks[index]
        start = float(track.get("start", 0.0))
        duration = float(track.get("duration", 0.0) or 0.0)
        local = float(self.engine.position) - start
        if local <= 0.05 or local >= duration - 0.05:
            self.statusBar().showMessage(
                "Move the playhead inside the clip to split", 3000
            )
            return
        self._remember_edit()
        tail = dict(track)
        tail["start"] = start + local
        tail["offset"] = float(track.get("offset", 0.0) or 0.0) + local
        tail["duration"] = duration - local
        tail["armed"] = False
        track["duration"] = local
        self.project.tracks.insert(index + 1, tail)
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(index)

    def _duplicate_selected(self) -> None:
        indices = self._selection()
        if not indices:
            return
        self._remember_edit()
        shift = 0
        for index in indices:
            source = index + shift
            original = self.project.tracks[source]
            clone = dict(original)
            clone["name"] = f"{original.get('name', 'Audio')} copy"
            clone["start"] = float(original.get("start", 0.0)) + float(
                original.get("duration", 0.0) or 0.0
            )
            clone["armed"] = False
            self.project.tracks.insert(source + 1, clone)
            shift += 1
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()

    def _move_track(self, direction: int) -> None:
        index = self._selected
        target = index + direction
        if not (
            0 <= index < len(self.project.tracks)
            and 0 <= target < len(self.project.tracks)
        ):
            return
        self._remember_edit()
        tracks = self.project.tracks
        tracks[index], tracks[target] = tracks[target], tracks[index]
        self._selected = target
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(target)

    def _set_clip_gain(self, value: float) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        self.project.tracks[index]["clip_gain"] = max(0.0, min(2.0, float(value)))
        self._push_mix(index, self.project.tracks[index])

    def _set_fade(self, key: str, value: float) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        self._remember_edit()
        self.project.tracks[index][key] = max(0.0, float(value))
        self._push_mix(index, self.project.tracks[index])
        self._commit_edit()
        self._refresh_timeline()
        self._select_track(index)

    def _set_armed(self, index: int, armed: bool) -> None:
        if not (0 <= index < len(self.project.tracks)):
            return
        self.project.tracks[index]["armed"] = bool(armed)

    def _read_visualizer(self, source: int, count: int):
        if not self.engine.playing:
            return None
        if int(source) < 0:
            return self.engine.read_master(count)
        return self.engine.read_track(int(source), count)

    def _sync_capture(self, *_args) -> None:
        visible = self.visualizer.isVisible()
        if visible == self._capture_shown:
            return
        self._capture_shown = visible
        self.engine.set_capture(visible)
        self.visualizer.set_active(visible)

    def _restore_view(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        settings = QSettings("SonicLab", "SonicLab")
        state = settings.value("windowState")
        if state is not None:
            self.restoreState(state)
        value = settings.value("visualizer")
        if hasattr(value, "keys"):
            self.visualizer.apply_state(dict(value))

    def _toggle_record(self) -> None:
        if self.recorder.recording:
            self._stop_recording()
            return
        armed = [
            index
            for index, track in enumerate(self.project.tracks)
            if track.get("armed")
        ]
        if not armed:
            self.statusBar().showMessage("Arm a track before recording", 4000)
            return
        if self.input_device is None:
            devices, error = input_devices()
            if error or not devices:
                QMessageBox.warning(
                    self,
                    "No input",
                    error or "No input device is available.",
                )
                return
            self.input_device, _name, self.input_channels = devices[0]
            self.input_channel = 0
            self.input_stereo = self.input_channels >= 2
        stream_channels = 2 if self.input_stereo else 1
        if not self.input_stereo:
            stream_channels = max(1, self.input_channel + 1)
        else:
            stream_channels = max(stream_channels, self.input_channel + 2)
        try:
            self.recorder.start(
                self.input_device,
                stream_channels,
                self.input_channel,
                self.input_stereo,
            )
        except Exception as exc:
            self.recorder.stop_stream()
            QMessageBox.critical(self, "Recording error", str(exc))
            self._set_record_lamp(False)
            return
        self._record_started_at = float(self.engine.position)
        self._armed_at_record = armed
        self._set_record_lamp(True)
        self.statusBar().showMessage("Recording")

    def _stop_recording(self) -> None:
        self._set_record_lamp(False)
        try:
            directory = self._media_dir()
            self.recorder.finish_async(self._unique_take(directory))
        except Exception as exc:
            self.recorder.stop_stream()
            QMessageBox.critical(self, "Recording error", str(exc))
            return
        self.statusBar().showMessage("Saving recording…", 2000)

    def _poll_recording(self) -> None:
        result = self.recorder.poll()
        if not result:
            return
        kind, payload, audio = result
        if kind == "error":
            QMessageBox.critical(self, "Recording error", str(payload))
            return
        if kind == "empty" or audio is None or len(audio) == 0:
            self.statusBar().showMessage("Recording was empty", 3000)
            return
        self._place_recording(payload, audio)

    def _queue_beat_sources(self, paths) -> None:
        """Queue analysis, drum track first so the others line up sooner."""
        paths = [path for path in paths if path]
        drum = self._drum_index()
        if drum is not None:
            first = self.project.tracks[drum].get("path")
            if first in paths:
                paths.remove(first)
                paths.insert(0, first)
        for path in paths:
            self.beat_jobs.submit(path)

    def _analyze_beats(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to analyze", 3000)
            return
        path = self.project.tracks[index].get("path")
        state = self.beat_jobs.submit(path)
        if state == "missing":
            self.statusBar().showMessage(
                "This clip has no source file to analyze", 4000
            )
            return
        if state == "duplicate":
            self.statusBar().showMessage("Beat analysis is already queued", 3000)
            return
        if state == "ready":
            self._apply_detected_beats()
            self.statusBar().showMessage(
                self._beat_status(self.beat_jobs.result_for(path)), 5000
            )
            return
        self.statusBar().showMessage("Analyzing downbeats…", 3000)

    def _poll_beats(self) -> None:
        for _ in range(8):
            message = self.beat_jobs.poll()
            if message is None:
                return
            kind = message[0]
            if kind == "progress":
                self.statusBar().showMessage(f"Analyzing downbeats: {message[2]}", 2000)
            elif kind == "finished":
                self._apply_detected_beats()
                self._apply_onsets()
                self._follow_drum_bars()
                self.statusBar().showMessage(self._beat_status(message[2]), 5000)

    def _analyze_onsets(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to analyze", 3000)
            return
        path = self.project.tracks[index].get("path")
        state = self.onset_jobs.submit(path)
        if state == "missing":
            self.statusBar().showMessage(
                "This clip has no source file to analyze", 4000
            )
            return
        if state == "duplicate":
            self.statusBar().showMessage("Onset analysis is already queued", 3000)
            return
        if state == "ready":
            self._apply_onsets()
            count = len(self.onset_jobs.onsets_for(path))
            self.statusBar().showMessage(self._onset_status(count), 4000)
            return
        self.statusBar().showMessage("Analyzing onsets…", 3000)

    def _poll_onsets(self) -> None:
        for _ in range(8):
            message = self.onset_jobs.poll()
            if message is None:
                return
            if message[0] == "finished":
                self._apply_onsets()
                result = message[2]
                if getattr(result, "status", "") in (
                    "error",
                    "unavailable",
                    "cancelled",
                ):
                    self.statusBar().showMessage(
                        result.message or "Onset analysis failed", 4000
                    )
                else:
                    self.statusBar().showMessage(
                        self._onset_status(len(result.onsets)), 4000
                    )

    def _set_beat_reference(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage(
                "Select a clip to use as the beat reference", 3000
            )
            return
        for track_index, track in enumerate(self.project.tracks):
            if track_index == index:
                track["_beat_reference"] = True
            else:
                track.pop("_beat_reference", None)
        self._apply_onsets()
        name = self.project.tracks[index].get("name") or "Clip"
        self.statusBar().showMessage(f"Beat reference: {name}", 4000)

    def _nudge_to_reference(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to nudge", 3000)
            return
        if self._reference_index() is None:
            self.statusBar().showMessage("Set a beat reference first", 4000)
            return
        track = self.project.tracks[index]
        before = float(track.get("_alignment", 0.0) or 0.0)
        updated = nudged_alignment(before, matched_errors(self._onset_marks(index)))
        if updated is None:
            self.statusBar().showMessage("No confident onset matches to nudge", 4000)
            return

        def apply(value, track_index=index):
            if not (0 <= track_index < len(self.project.tracks)):
                return
            self.project.tracks[track_index]["_alignment"] = float(value)
            self._apply_onsets()

        apply(updated)
        self.history.push(Edit(lambda: apply(before), lambda: apply(updated)))
        delta = updated - before
        self.statusBar().showMessage(f"Nudged by {delta * 1000:+.0f} ms", 4000)

    def _reference_index(self) -> int | None:
        for index, track in enumerate(self.project.tracks):
            if track.get("_beat_reference"):
                return index
        return self._drum_index()

    def _onset_marks(self, index: int):
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return ()
        track = tracks[index]
        reference = self._reference_index()
        if reference is None:
            reference_events = ()
            reference_track = track
        else:
            reference_track = tracks[reference]
            reference_events = self._track_events(reference_track)
        return classify_onsets(
            tuple(
                float(time) * self._stretch_of(track) if isinstance(time, (int, float)) else time
                for time in (self.onset_jobs.onsets_for(track.get("path")) or ())
            ),
            reference_events,
            float(track.get("start", 0.0) or 0.0),
            float(track.get("offset", 0.0) or 0.0),
            float(track.get("_alignment", 0.0) or 0.0),
            float(track.get("duration", 0.0) or 0.0),
            float(reference_track.get("start", 0.0) or 0.0),
            float(reference_track.get("offset", 0.0) or 0.0),
            float(reference_track.get("_alignment", 0.0) or 0.0),
        )

    def _apply_onsets(self) -> None:
        visible = self.timeline.show_onsets
        for item in [*self.timeline.scene_obj.items(), *self.timeline2.scene_obj.items()]:
            if not isinstance(item, ClipItem) or not isValid(item):
                continue
            index = item.clip_index
            marks = self._onset_marks(index)
            item.set_onsets(
                ((mark.local_seconds, mark.kind) for mark in marks),
                visible,
            )

    @staticmethod
    def _onset_status(count: int) -> str:
        if count <= 0:
            return "No onsets detected"
        noun = "onset" if count == 1 else "onsets"
        return f"Detected {count} {noun}"

    def _apply_detected_beats(self) -> None:
        visible = self.timeline.show_detected
        tracks = self.project.tracks
        for item in [*self.timeline.scene_obj.items(), *self.timeline2.scene_obj.items()]:
            if not isinstance(item, ClipItem) or not isValid(item):
                continue
            index = item.clip_index
            if not (0 <= index < len(tracks)):
                continue
            track = tracks[index]
            item.set_detected_beats(
                self._track_events(track),
                float(track.get("offset", 0.0) or 0.0),
                visible,
            )

    def _own_events(self, track: dict) -> tuple[tuple[float, int], ...]:
        """Applied downbeats when the track has them, else Madmom's result."""
        if track.get("downbeats"):
            return track_events(track)
        factor = self._stretch_of(track)
        detected = self.beat_jobs.events_for(track.get("path"))
        if factor != 1.0:
            # Analysed in the file's own time; the clip plays stretched.
            detected = tuple((float(time) * factor, position) for time, position in detected or ())
        return track_events(track, detected)

    def _track_events(self, track: dict) -> tuple[tuple[float, int], ...]:
        """What a track shows and loops on.

        Applied downbeats win. Otherwise every track follows the drum track,
        kept at the same arrangement time. Without drum beats a track falls
        back to its own analysis.
        """
        if track.get("downbeats"):
            return track_events(track)
        drum = self._drum_index()
        if drum is not None and self.project.tracks[drum] is not track:
            source = self.project.tracks[drum]
            events = self._own_events(source)
            if events:
                return tuple(
                    (time, position)
                    for time, position in transfer_events(
                        events,
                        float(source.get("start", 0.0) or 0.0),
                        float(source.get("offset", 0.0) or 0.0),
                        float(track.get("start", 0.0) or 0.0),
                        float(track.get("offset", 0.0) or 0.0),
                    )
                )
        return self._own_events(track)

    def _drum_index(self) -> int | None:
        for index, track in enumerate(self.project.tracks):
            if track.get("drum_reference"):
                return index
        return None

    def _detect_drum_track(self) -> int | None:
        """Mark the drum track if none is marked yet. Returns its index."""
        index = pick_drum_track(self.project.tracks, OUTPUT_RATE)
        if index is not None:
            self.project.tracks[index]["drum_reference"] = True
        return index

    def _toggle_drum_track(self) -> None:
        index = self._selected
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return
        self._remember_edit()
        if tracks[index].get("drum_reference"):
            tracks[index].pop("drum_reference", None)
            message = "No drum track. Tracks show their own downbeats"
        else:
            for track in tracks:
                track.pop("drum_reference", None)
            tracks[index]["drum_reference"] = True
            name = tracks[index].get("name") or "Clip"
            message = f"Drum track: {name}. Other tracks follow its downbeats"
            self._queue_beat_sources([tracks[index].get("path")])
        self._commit_edit()
        self._apply_detected_beats()
        self._apply_onsets()
        self._follow_drum_bars()
        self.statusBar().showMessage(message, 5000)

    def _drum_bar_lines(self) -> list[float]:
        """Bar lines at the drum track's downbeats, in arrangement time."""
        drum = self._drum_index()
        if drum is None:
            return []
        track = self.project.tracks[drum]
        lines = bars_from_downbeats(
            self._own_events(track),
            float(track.get("start", 0.0) or 0.0),
            float(track.get("offset", 0.0) or 0.0),
        )
        return lines if len(lines) >= 2 else []

    def _bar_lines(self) -> list[float]:
        """Bar lines the user placed, else the drum downbeats.

        Empty means the timeline draws evenly spaced bars from the tempo.
        """
        if self.project.bars:
            return list(self.project.bars)
        return self._drum_bar_lines()

    def _follow_drum_bars(self) -> None:
        # Bars that follow the drums move when the drum downbeats change.
        if not self.project.bars and self.timeline.bar_map != self._drum_bar_lines():
            self._refresh_timeline()

    def _set_bars(self, lines, message: str) -> None:
        self._remember_edit()
        self.project.bars = [round(float(value), 6) for value in lines]
        self._commit_edit()
        self._refresh_timeline()
        self.statusBar().showMessage(message, 4000)

    def _playhead_bar(self) -> int:
        lines = self.timeline.bar_lines
        seconds = float(self.engine.position)
        for index in range(len(lines) - 1):
            if lines[index] <= seconds < lines[index + 1]:
                return index
        return 0

    def _edit_bar(self, index: int | None = None) -> None:
        """Type the start and end of a bar. The bars beside it stretch to meet it."""
        lines = self.timeline.bar_lines
        fallback = self.timeline.bar_fallback
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit bar")
        form = QFormLayout(dialog)
        number = QSpinBox()
        number.setRange(1, 9999)
        number.setValue((self._playhead_bar() if index is None else int(index)) + 1)
        start_edit = QLineEdit()
        end_edit = QLineEdit()
        for edit in (start_edit, end_edit):
            edit.setPlaceholderText("HH:MM:SS.cc")
        length = QLabel()

        def show_bar(value: int) -> None:
            start, end = bar_span(lines, value - 1, fallback)
            start_edit.setText(format_clock(start))
            end_edit.setText(format_clock(end))

        def show_length() -> None:
            try:
                length.setText(
                    f"{parse_clock(end_edit.text()) - parse_clock(start_edit.text()):.2f} s"
                )
            except ValueError:
                length.setText("—")

        number.valueChanged.connect(show_bar)
        start_edit.textChanged.connect(show_length)
        end_edit.textChanged.connect(show_length)
        show_bar(number.value())
        form.addRow("Bar", number)
        form.addRow("Starts at", start_edit)
        form.addRow("Ends at", end_edit)
        form.addRow("Length", length)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_bar_span(number.value() - 1, start_edit.text(), end_edit.text())

    def _apply_bar_span(self, index: int, start_text: str, end_text: str) -> bool:
        try:
            start, end = parse_clock(start_text), parse_clock(end_text)
            lines = set_span(
                self._current_bar_lines(), index, start, end, self.timeline.bar_fallback
            )
        except ValueError as exc:
            self.statusBar().showMessage(f"Bar {index + 1} not changed: {exc}", 5000)
            return False
        self._set_bars(
            lines, f"Bar {index + 1}: {format_clock(start)} – {format_clock(end)}"
        )
        return True

    def _current_bar_lines(self) -> list[float]:
        # The bars on screen, with the tempo grid made real when nothing is placed.
        return list(
            self.project.bars or self.timeline.bar_map or self.timeline.bar_lines
        )

    def _add_bar_line(self) -> None:
        seconds = float(self.engine.position)
        lines = insert_line(self._current_bar_lines(), seconds)
        self._set_bars(lines, f"Bar line added at {format_clock(seconds)}")

    def _remove_bar_line(self) -> None:
        lines = self._current_bar_lines()
        index = nearest_line(lines, float(self.engine.position))
        if index < 0 or len(lines) <= 2:
            self.statusBar().showMessage("Keep at least two bar lines", 3000)
            return
        removed = lines[index]
        self._set_bars(
            remove_line(lines, index), f"Bar line at {format_clock(removed)} removed"
        )

    def _mark_bars_from_drums(self) -> None:
        lines = self._drum_bar_lines()
        if not lines:
            self.statusBar().showMessage(
                "No drum downbeats yet. Mark a drum track and analyze it", 5000
            )
            return
        self._set_bars(lines, f"Marked {len(lines) - 1} bars at the drum downbeats")

    def _reset_bars(self) -> None:
        follows = "the drum downbeats" if self._drum_bar_lines() else "the tempo"
        self._set_bars([], f"Bars follow {follows}")

    def _bar_bounds(self, index: int) -> tuple[float, float] | None:
        lines, fallback = self.timeline.bar_lines, self.timeline.bar_fallback
        if not lines or index < 0:
            return None
        start, end = bar_span(lines, index, fallback)
        if end <= start:
            return None
        return float(start), float(end)

    def _play_bar(self, index: int, track: int = -1) -> None:
        """Play one bar. From a bar's ▶ on a track (``track`` >= 0) only that
        track (that stem) is heard; from a menu, the whole canvas in focus."""
        if 0 <= track < len(self.project.tracks):
            self._toggle_block(track, index)
            return
        bounds = self._bar_bounds(index)
        if bounds is None:
            return
        start, end = bounds
        self._engage_loop(start, end)
        self.engine.seek(start)
        self._play_hovered()
        self.statusBar().showMessage(f"Playing bar {index + 1}", 4000)

    # -- stem blocks played on their own -------------------------------------
    # A block's ▶ plays that bar of that stem as its own voice in the engine,
    # mixed with anything else playing. The transport, its playhead, the
    # loop and the scroll position are left alone; the playhead is hidden
    # while only blocks play.

    # A stem of a split song (see stem_group) is one voice on its split's
    # clock instead: the first ▶ loops that block (the picked blocks' span,
    # if several start together) until paused. Every stem loops the block
    # pressed on it, in step with the clock: a stem started on another bar
    # plays that bar at the same place in the loop (a shift on the clock),
    # and several bars of one stem can loop at once, each its own voice.
    # Picked bars side by side on one row play as one block (a run): one
    # voice from the first bar to the end of the last, keyed by the first.
    # A block shorter than the loop is silent for the rest of each cycle.
    # Blocks of different clips that carry on from each other (one stem
    # file, one row, touching in time and in the file, as when a stem was
    # cut into pieces) form a chain: they share the chain's place on the
    # clock, so they play one after the other as one block.
    # A split's synced loop (stem_loop) repeats its own region instead.

    def _voice_key(self, track: int, bar: int) -> tuple[int, int]:
        tracks = self.project.tracks
        if 0 <= track < len(tracks) and stem_group(tracks[track]):
            if stem_loop(tracks[track]) is not None:
                return int(track), STEM_VOICE
            for key, (_data, run) in self._preview_tracks.items():
                if key[0] == track and key[1] <= STEM_BLOCK and run and run[0] <= bar <= run[1]:
                    return key  # a bar of a run playing as one block
            return int(track), stem_block(bar)
        return int(track), int(bar)

    def _toggle_block(self, track: int, bar: int) -> None:
        picks = self._view_for(track).picks()
        if bar in picks.get(track, ()) and sum(len(bars) for bars in picks.values()) > 1:
            # ▶ on a picked block plays (or pauses) the whole selection.
            keys = {self._voice_key(index, other) for index, bars in picks.items() for other in bars}
            states = self.engine.previews()
            if any(states.get(key, (False,))[0] for key in keys):
                for key in keys:
                    self.engine.preview_pause(key)
                self._sync_previews()
            else:
                self._play_picked_blocks()
            return
        key = self._voice_key(track, bar)
        state = self.engine.previews().get(key)
        if state is not None and state[0]:
            self.engine.preview_pause(key)
            self._sync_previews()
            return
        self._start_blocks([(track, bar)])

    def _block_runs(self, keys) -> list[tuple[int, int, int]]:
        """``(track, first bar, last bar)`` for ``keys``: a stem's bars side
        by side become one run; every other block stands alone."""
        tracks = self.project.tracks
        bars: dict[int, set] = {}
        units = []
        for track, bar in keys:
            if 0 <= track < len(tracks) and stem_group(tracks[track]) and stem_loop(tracks[track]) is None:
                bars.setdefault(track, set()).add(int(bar))
            else:
                units.append((track, bar, bar))
        for track, found in bars.items():
            view = self._view_for(track)
            run = None
            for bar in sorted(found):
                times = view.block_times(track, bar)
                if times is None:
                    continue
                if run is not None and run[1] == bar - 1 and abs(run[2] - times[0]) < 1e-3:
                    run = (run[0], bar, times[1])
                    continue
                if run is not None:
                    units.append((track, run[0], run[1]))
                run = (bar, bar, times[1])
            if run is not None:
                units.append((track, run[0], run[1]))
        return units

    def _stem_chains(self, units) -> dict:
        """``(track, first bar) -> (chain number, song start, song end)`` for
        the stem block units (see _block_runs) that chain together; a unit
        alone is a chain of one."""
        tracks = self.project.tracks
        found = []
        for track, first, last in units:
            data = tracks[track] if 0 <= track < len(tracks) else None
            if data is None or stem_group(data) is None or stem_loop(data) is not None:
                continue
            view = self._view_for(track)
            head, tail = view.block_times(track, first), view.block_times(track, last)
            if head is None or tail is None:
                continue
            rows, _ = view.lane_rows(tracks)
            place = (stem_group(data), data.get("path"), track_canvas(data), rows[track])
            found.append((head[0], tail[1], self._song_time(data, head[0]), self._song_time(data, tail[1]), place, (track, first)))
        found.sort(key=lambda item: item[0])
        chains = []  # [place, arrangement end, song start, song end, units]
        for start, end, song, song_end, place, unit in found:
            for chain in chains:
                if chain[0] == place and abs(chain[1] - start) < 1e-3 and abs(chain[3] - song) < 1e-3:
                    chain[1], chain[3] = end, song_end
                    chain[4].append(unit)
                    break
            else:
                chains.append([place, end, song, song_end, [unit]])
        return {unit: (number, chain[2], chain[3]) for number, chain in enumerate(chains) for unit in chain[4]}

    def _with_neighbours(self, units) -> list:
        """A block pressed beside one of its chain that is already playing:
        that split's playing blocks stop and start again with it, so the
        chain plays as one from its start."""
        existing = {
            (key[0], run[0]): (key, (key[0], run[0], run[1]))
            for key, (_data, run) in self._preview_tracks.items()
            if key[1] <= STEM_BLOCK and run
        }
        asked = {(track, first) for track, first, _last in units}
        extra = [unit for unit_key, (_key, unit) in existing.items() if unit_key not in asked]
        if not extra:
            return units
        chains = self._stem_chains(units + extra)
        tracks = self.project.tracks
        restart = set()
        for unit_key in asked:
            if unit_key not in chains:
                continue
            number = chains[unit_key][0]
            if any(chains.get(other[:2], (None,))[0] == number for other in extra):
                restart.add(stem_group(tracks[unit_key[0]]))
        if not restart:
            return units
        units = list(units)
        for unit_key, (key, unit) in existing.items():
            if unit_key in asked or stem_group(tracks[key[0]]) not in restart:
                continue
            self.engine.preview_stop(key)
            del self._preview_tracks[key]
            units.append(unit)
        return units

    def _start_blocks(self, keys) -> None:
        """Start (or resume) blocks together, on the same audio sample.
        Stems join their split's clock; see _voice_key."""
        blocks = []
        groups: dict[str, list] = {}  # group -> [start, members, loop, end] (song seconds)
        spans: dict[tuple, tuple] = {}  # stem voice key -> (start, end) of its chain
        units = self._with_neighbours(self._block_runs(keys))
        chains = self._stem_chains(units)
        for track, bar, last in units:
            view = self._view_for(track)
            times = view.block_times(track, bar)
            if times is None:
                continue
            if last != bar:
                times = (times[0], view.block_times(track, last)[1])
            data = self.project.tracks[track]
            group = stem_group(data)
            if group is None:
                blocks.append(((track, bar), track, times[0], times[1]))
                self._preview_tracks[(track, bar)] = (data, times)
                continue
            # Where the block is in the song: the stem files' own time.
            song = self._song_time(data, times[0])
            end = self._song_time(data, times[1])
            loop = stem_loop(data)
            if loop is not None:
                group = loop_group(group)  # starts at the loop start
            entry = groups.setdefault(group, [song, {}, loop, end])
            entry[0] = min(entry[0], song)
            entry[3] = max(entry[3], end)
            if loop is None:
                key = (track, stem_block(bar))
                # Blocks of this stem inside the new run give way to it.
                for other, (_data, run) in list(self._preview_tracks.items()):
                    if other[0] == track and other[1] <= STEM_BLOCK and other != key and run and run[0] <= last and bar <= run[1]:
                        self.engine.preview_stop(other)
                        del self._preview_tracks[other]
                self._preview_tracks[key] = (data, (bar, last))
            else:
                key = (track, STEM_VOICE)
                self._preview_tracks[key] = (data, None)
            spans[key] = chains[(track, bar)][1:] if (track, bar) in chains else (song, end)
            entry[1][key] = (key, track, bool(data.get("muted", False)))
        if not blocks and not groups:
            return
        joining = {group for group in groups if self.engine.group_position(group) is not None}
        for group, entry in groups.items():
            if entry[2] is not None:
                continue  # the split's synced loop: one region for all
            members = entry[1]
            if group in joining:
                ref = self.engine.group_loop(group)
            else:
                # The earliest block sets the clock; the loop is as long
                # as the longest block.
                first = min(spans[key][0] for key in members)
                length = max(spans[key][1] - spans[key][0] for key in members)
                ref = entry[2] = (first, first + length)
                entry[0] = first
            for key, member in members.items():
                shift = 0.0 if ref is None else spans[key][0] - ref[0]
                members[key] = (*member, shift, spans[key][1] - spans[key][0])
        try:
            with self.engine.batch():  # everything starts on the same block
                if blocks:
                    self.engine.preview_play(blocks)
                for group, (start, members, loop, end) in groups.items():
                    self.engine.stem_play(group, list(members.values()), max(0.0, start), loop=loop)
        except Exception as exc:
            for key, *_rest in blocks:
                self.engine.preview_stop(key)
            for _start, members, _loop, _end in groups.values():
                for key, *_rest in members.values():
                    self.engine.preview_stop(key)
            self._sync_previews()
            QMessageBox.critical(self, "Audio output error", f"Could not play the block.\n\n{exc}")
            return
        self.stem_splitter.stop_preview()  # the splitter's own player is separate
        self._sync_previews()
        if not self.frame_timer.isActive():
            self.frame_timer.start()
        stems = [member for _start, members, _loop, _end in groups.values() for member in members.values()]
        if len(blocks) + len(stems) != 1:
            self.statusBar().showMessage(f"Playing {len(blocks) + len(stems)} blocks together", 3000)
        elif blocks:
            (track, bar), *_rest = blocks[0]
            name = self.project.tracks[track].get("name", "Track")
            self.statusBar().showMessage(f"Playing bar {bar + 1} of “{name}”", 3000)
        else:
            track = stems[0][1]
            name = self.project.tracks[track].get("name", "Track")
            how = "Joined the loop in sync" if joining else "Looping the block"
            if stem_loop(self.project.tracks[track]) is not None:
                how = "Joined the synced loop" if joining else "Looping in sync"
            self.statusBar().showMessage(f"{how}: “{name}”", 3000)

    # -- synced stem loops ---------------------------------------------------
    # Picked stem blocks of one split become its synced loop: one region of
    # the song, stored on each member track as "stem_loop" (song seconds).
    # Members play on the loop's own engine clock (loop_group), which the
    # audio callback wraps on the sample; the split's other stems are not in
    # it. Markers on each member clip move the region for all of them.

    @staticmethod
    def _song_time(track, seconds: float) -> float:
        """Arrangement ``seconds`` on a stem's clip, in its song's time."""
        return float(track.get("offset", 0.0) or 0.0) + float(seconds) - float(track.get("start", 0.0) or 0.0)

    def _loop_picked_stems(self) -> None:
        regions: dict[str, list] = {}  # split -> [start, end, member indices]
        tracks = self.project.tracks
        for view in self._timelines():
            for index, bars in view.picks().items():
                if not (0 <= index < len(tracks)) or stem_group(tracks[index]) is None:
                    continue
                for bar in bars:
                    times = view.block_times(index, bar)
                    if times is None:
                        continue
                    start = self._song_time(tracks[index], times[0])
                    end = self._song_time(tracks[index], times[1])
                    entry = regions.setdefault(stem_group(tracks[index]), [start, end, set()])
                    entry[0], entry[1] = min(entry[0], start), max(entry[1], end)
                    entry[2].add(index)
        if not regions:
            self.statusBar().showMessage("Pick blocks of split stems to loop them in sync", 4000)
            return
        self._remember_edit()
        for group, (start, end, members) in regions.items():
            for index, track in enumerate(tracks):
                if stem_group(track) != group:
                    continue
                if index in members:
                    track["stem_loop"] = [round(start, 6), round(end, 6)]
                else:
                    track.pop("stem_loop", None)
                    self._leave_stem_loop(index, group)
        self._commit_edit()
        self._push_stem_loops()
        self._show_stem_loops()
        count = sum(len(entry[2]) for entry in regions.values())
        start, end, _members = next(iter(regions.values()))
        self.statusBar().showMessage(
            f"{count} stem{'s' if count != 1 else ''} loop {start:.2f}–{end:.2f} s in sync. Press ▶ on any of them", 6000
        )

    def _remove_from_stem_loop(self, index: int) -> None:
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)) or stem_loop(tracks[index]) is None:
            return
        self._remember_edit()
        tracks[index].pop("stem_loop", None)
        self._leave_stem_loop(index, stem_group(tracks[index]))
        self._commit_edit()
        self._push_stem_loops()
        self._show_stem_loops()

    def _clear_stem_loop(self, index: int) -> None:
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)) or stem_loop(tracks[index]) is None:
            return
        group = stem_group(tracks[index])
        self._remember_edit()
        for other, track in enumerate(tracks):
            if stem_group(track) == group and track.pop("stem_loop", None) is not None:
                self._leave_stem_loop(other, group)
        self._commit_edit()
        self._push_stem_loops()
        self._show_stem_loops()

    def _leave_stem_loop(self, index: int, group: str) -> None:
        """A stem taken out of a loop stops if it was looping."""
        key = (index, STEM_VOICE)
        if key in self._preview_tracks:
            self.engine.preview_stop(key)
            self._sync_previews()

    def _push_stem_loops(self) -> None:
        """Each playing loop clock gets its split's current region."""
        regions: dict[str, tuple | None] = {}
        for track in self.project.tracks:
            group = stem_group(track)
            if group is not None:
                regions[group] = regions.get(group) or stem_loop(track)
        for group, loop in regions.items():
            self.engine.group_set_loop(loop_group(group), loop)

    def _show_stem_loops(self) -> None:
        for timeline in self._timelines():
            timeline.show_stem_loops(self.project.tracks)

    def _drag_stem_loop(self, index: int, which: int, seconds: float, finished: bool) -> None:
        """A loop marker on a member clip moved: the region moves for every
        member of that split, and a playing loop takes it at once."""
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return
        loop = stem_loop(tracks[index])
        if loop is None:
            return
        self._remember_edit()
        song = float(tracks[index].get("offset", 0.0) or 0.0) + float(seconds)
        start, end = loop
        if which == 0:
            start = max(0.0, min(song, end - MIN_STEM_LOOP))
        else:
            end = max(song, start + MIN_STEM_LOOP)
        group = stem_group(tracks[index])
        for track in tracks:
            if stem_group(track) == group and stem_loop(track) is not None:
                track["stem_loop"] = [round(start, 6), round(end, 6)]
        self.engine.group_set_loop(loop_group(group), (start, end))
        self._show_stem_loops()
        if finished:
            self._commit_edit()
            self.statusBar().showMessage(f"Synced loop: {start:.3f}–{end:.3f} s", 4000)

    def _push_stem_mutes(self) -> None:
        """A muted stem stays on its split's clock, silent."""
        for index, bar in list(self._preview_tracks):
            if bar < 0 and index < len(self.project.tracks):
                self.engine.preview_set_muted((index, bar), bool(self.project.tracks[index].get("muted", False)))

    def _stop_block(self, key) -> None:
        self.engine.preview_stop(key)
        self._sync_previews()

    def _stop_blocks(self) -> None:
        self.engine.preview_stop_all()
        self._sync_previews()

    def _play_picked_blocks(self) -> None:
        keys = [
            (track, bar)
            for view in self._timelines()
            for track, bars in sorted(view.picks().items())
            for bar in sorted(bars)
        ]
        self._start_blocks(keys)

    def _sync_previews(self, states: dict | None = None) -> dict:
        """Buttons, tints and cursors from the engine's own voice state."""
        states = self.engine.previews() if states is None else states
        for key in [key for key in self._preview_tracks if key not in states]:
            del self._preview_tracks[key]
        shown = self.engine.playing or not states
        blocks = {}
        for (track, bar), state in states.items():
            if bar == STEM_VOICE:
                # A looping stem shows on the block its clock is in now.
                bar = self._view_for(track).block_at_time(track, state[1])
                if bar is None:
                    continue
            elif bar < 0:
                # A block (or a run of them) shows on every bar it covers.
                entry = self._preview_tracks.get((track, bar))
                first = stem_block(bar)
                last = entry[1][1] if entry and entry[1] else first
                for covered in range(first, last + 1):
                    blocks[(track, covered)] = state
                continue
            blocks[(track, bar)] = state
        for timeline in self._timelines():
            timeline.set_previews(blocks)
            timeline.set_playhead_visible(shown)
        return states

    def _follow_preview_tracks(self) -> None:
        """Tracks were inserted or reordered: keep each block voice on its
        own track (found by identity, not index), or stop it if it is gone.
        Runs in the same engine batch as the reload."""
        where = {id(track): index for index, track in enumerate(self.project.tracks)}
        changes = {}
        for key, (track, times) in list(self._preview_tracks.items()):
            index = where.get(id(track))
            if index is None:
                self.engine.preview_stop(key)
                del self._preview_tracks[key]
            elif index != key[0]:
                new = (index, key[1])
                changes[key] = (new, index)
        if changes:
            self.engine.preview_retarget(changes)
            self._preview_tracks = {
                changes.get(key, (key,))[0]: value for key, value in self._preview_tracks.items()
            }

    def _prune_previews(self) -> None:
        """After an edit: stop voices whose track or block moved or went."""
        tracks = self.project.tracks
        for key, (track, times) in list(self._preview_tracks.items()):
            index, bar = key
            if bar < 0:
                # A stem follows its clock wherever its clip is moved.
                if index >= len(tracks) or tracks[index] is not track:
                    self.engine.preview_stop(key)
                continue
            current = self._view_for(index).block_times(index, bar) if index < len(tracks) else None
            if (
                index >= len(tracks)
                or tracks[index] is not track
                or current is None
                or abs(current[0] - times[0]) > 1e-3
                or abs(current[1] - times[1]) > 1e-3
            ):
                self.engine.preview_stop(key)
        self._sync_previews()

    def _loop_bar(self, index: int) -> None:
        bounds = self._bar_bounds(index)
        if bounds is None:
            return
        self._engage_loop(*bounds)
        self.statusBar().showMessage(f"Looping bar {index + 1}", 4000)

    def _export_one_bar(self, index: int) -> None:
        bounds = self._bar_bounds(index)
        if bounds is None:
            return
        self._engage_loop(*bounds)
        self._export_bars(index, index)

    def _add_bar_actions(self, menu: QMenu, index: int) -> None:
        menu.addAction("Play this bar", lambda i=index: self._play_bar(i))
        menu.addAction("Loop this bar", lambda i=index: self._loop_bar(i))
        menu.addAction("Export this bar…", lambda i=index: self._export_one_bar(i))
        menu.addAction("Edit bar…", lambda i=index: self._edit_bar(i))

    def _show_bar_menu(self, index: int, pos) -> None:
        if index < 0:
            return
        menu = QMenu(self)
        self._add_bar_actions(menu, index)
        menu.exec(pos)

    def _rename_track_at(self, index: int) -> None:
        if not (0 <= index < len(self.project.tracks)):
            return
        self._select_track(index)
        self._rename_track()

    def _select_bars(self, first: int, last: int) -> None:
        """Loop whole bars, which is also the range Export Bars offers."""
        lines, fallback = self.timeline.bar_lines, self.timeline.bar_fallback
        start, _end = bar_span(lines, first, fallback)
        _start, end = bar_span(lines, last, fallback)
        self._engage_loop(start, end)
        label = f"Bar {first + 1}" if first == last else f"Bars {first + 1}–{last + 1}"
        self.statusBar().showMessage(
            f"{label} selected · {format_clock(start)} – {format_clock(end)} · Ctrl+Shift+E to export",
            6000,
        )

    def _selected_bars(self) -> tuple[int, int]:
        """Bars covered by the loop, else the bar under the playhead."""
        lines, fallback = self.timeline.bar_lines, self.timeline.bar_fallback
        bounds = getattr(self, "_loop_bounds", None)
        if bounds:
            first = max(0, bar_index(lines, bounds[0] + 1e-6, fallback))
            last = max(first, bar_index(lines, bounds[1] - 1e-6, fallback))
            return first, last
        bar = self._playhead_bar()
        return bar, bar

    def _bar_export_dialog(self) -> tuple[int, int, str, str] | None:
        """Ask for bars, content and format. None means cancelled."""
        lines, fallback = self.timeline.bar_lines, self.timeline.bar_fallback
        settings = QSettings("SonicLab", "SonicLab")
        first, last = self._selected_bars()
        dialog = QDialog(self)
        dialog.setWindowTitle("Export bars")
        form = QFormLayout(dialog)
        start_bar = QSpinBox()
        end_bar = QSpinBox()
        for spin, value in ((start_bar, first), (end_bar, last)):
            spin.setRange(1, 9999)
            spin.setValue(value + 1)
        span = QLabel()

        def show_span() -> None:
            low, high = sorted((start_bar.value() - 1, end_bar.value() - 1))
            start, _end = bar_span(lines, low, fallback)
            _start, end = bar_span(lines, high, fallback)
            span.setText(
                f"{format_clock(start)} – {format_clock(end)}  ({end - start:.3f} s)"
            )

        start_bar.valueChanged.connect(show_span)
        end_bar.valueChanged.connect(show_span)
        show_span()
        content = QComboBox()
        content.addItem("Mix of all tracks", "mix")
        content.addItem("Selected tracks, mixed", "selected")
        content.addItem("Each track as its own file", "each")
        formats = QComboBox()
        for item in FORMATS:
            formats.addItem(item.label, item.key)
        stored = formats.findData(str(settings.value("export/format", "wav24")))
        formats.setCurrentIndex(max(0, stored))
        form.addRow("From bar", start_bar)
        form.addRow("To bar", end_bar)
        form.addRow("Time", span)
        form.addRow("Content", content)
        form.addRow("Format", formats)
        note = QLabel(
            "Mute, solo, volume, pan, fades and master level are applied as you hear them."
        )
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        low, high = sorted((start_bar.value() - 1, end_bar.value() - 1))
        settings.setValue("export/format", formats.currentData())
        return low, high, str(formats.currentData()), str(content.currentData())

    def _export_folder(self) -> str:
        stored = QSettings("SonicLab", "SonicLab").value("export/folder", "")
        if stored and os.path.isdir(str(stored)):
            return str(stored)
        if self.current_path:
            return os.path.dirname(self.current_path)
        return os.path.expanduser("~")

    def _export_bars(
        self, first=None, last=None, format_key=None, content="mix", target=None
    ) -> list:
        """Render bars ``first``..``last`` (zero-based) and write them.

        ``target`` is a file path, or a folder for ``content == "each"``.
        Missing arguments are asked for. Returns the written results.
        """
        if not self.project.tracks:
            self.statusBar().showMessage("Nothing to export. Import audio first", 4000)
            return []
        if first is None or last is None or format_key is None:
            chosen = self._bar_export_dialog()
            if chosen is None:
                return []
            first, last, format_key, content = chosen
        fmt = FORMATS_BY_KEY[format_key]
        if content == "selected":
            include = set(self._selection())
            if not include:
                self.statusBar().showMessage("Select the tracks to export", 4000)
                return []
            jobs = [(include, None)]
        elif content == "each":
            # Only tracks you would hear: muted ones, or others while one is soloed, are skipped.
            tracks = self.project.tracks
            soloed = any(track.get("solo") for track in tracks)
            jobs, used = [], set()
            for index, track in enumerate(tracks):
                if (
                    track.get("empty")
                    or track.get("muted")
                    or (soloed and not track.get("solo"))
                ):
                    continue
                label = track.get("name") or f"Track {index + 1}"
                if label.lower() in used:
                    label = f"{label} {index + 1}"
                used.add(label.lower())
                jobs.append(({index}, label))
            if not jobs:
                self.statusBar().showMessage("No audible tracks to export", 4000)
                return []
        else:
            jobs = [(None, None)]
        name = self.project.name or "Untitled"
        if target is None:
            folder = self._export_folder()
            if content == "each":
                target = QFileDialog.getExistingDirectory(
                    self, "Export each track to", folder
                )
            else:
                target, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export bars",
                    os.path.join(folder, default_name(name, first, last) + fmt.suffix),
                    f"{fmt.label} (*{fmt.suffix})",
                )
            if not target:
                return []
            QSettings("SonicLab", "SonicLab").setValue(
                "export/folder",
                target if content == "each" else os.path.dirname(target),
            )
        start, end = bar_frames(
            self.timeline.bar_lines,
            first,
            last,
            self.timeline.bar_fallback,
            OUTPUT_RATE,
        )
        results = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for include, track_name in jobs:
                audio = self.engine.render(start, end - start, include)
                path = (
                    os.path.join(target, default_name(name, first, last, track_name))
                    if track_name
                    else target
                )
                results.append(write_audio(path, audio, OUTPUT_RATE, fmt))
        except (OSError, ValueError, RuntimeError, sf.LibsndfileError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return results
        finally:
            QApplication.restoreOverrideCursor()
        bars = f"bar {first + 1}" if first == last else f"bars {first + 1}–{last + 1}"
        where = (
            os.path.basename(str(results[0].path))
            if len(results) == 1
            else f"{len(results)} files in {target}"
        )
        message = f"Exported {bars} ({(end - start) / OUTPUT_RATE:.3f} s) to {where}"
        clipped = sum(result.clipped for result in results)
        if clipped:
            message += f" · {clipped} samples clipped, lower the master level"
        self.statusBar().showMessage(message, 8000)
        return results

    def _downbeat_targets(self, source: int) -> list[int] | None:
        """Ask which tracks receive the source downbeats. None means cancelled."""
        tracks = self.project.tracks
        chosen = set(self._selected_indices()) - {source}
        dialog = QDialog(self)
        dialog.setWindowTitle("Apply downbeats")
        form = QFormLayout(dialog)
        name = tracks[source].get("name") or "Clip"
        form.addRow(QLabel(f"Copy the downbeat markers of “{name}” to:"))
        targets = QListWidget()
        for index, track in enumerate(tracks):
            if index == source:
                continue
            item = QListWidgetItem(track.get("name") or f"Track {index + 1}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = index in chosen if chosen else True
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            targets.addItem(item)
        form.addRow(targets)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return [
            int(targets.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(targets.count())
            if targets.item(row).checkState() == Qt.CheckState.Checked
        ]

    def _apply_downbeats_to_tracks(self, targets: list[int] | None = None) -> None:
        source = self._selected
        tracks = self.project.tracks
        if not (0 <= source < len(tracks)):
            self.statusBar().showMessage("Select a clip with downbeats", 3000)
            return
        events = self._track_events(tracks[source])
        if not events:
            self.statusBar().showMessage("This clip has no downbeats to apply", 4000)
            return
        if targets is None:
            targets = self._downbeat_targets(source)
            if targets is None:
                return
        targets = [
            index for index in targets if 0 <= index < len(tracks) and index != source
        ]
        if not targets:
            self.statusBar().showMessage("No tracks chosen", 3000)
            return
        origin = tracks[source]
        self._remember_edit()
        for index in targets:
            track = tracks[index]
            track["downbeats"] = transfer_events(
                events,
                float(origin.get("start", 0.0) or 0.0),
                float(origin.get("offset", 0.0) or 0.0),
                float(track.get("start", 0.0) or 0.0),
                float(track.get("offset", 0.0) or 0.0),
            )
        self._commit_edit()
        self._apply_detected_beats()
        self._apply_onsets()
        noun = "track" if len(targets) == 1 else "tracks"
        self.statusBar().showMessage(
            f"Applied downbeats to {len(targets)} {noun}", 4000
        )

    def _export_downbeats(self, path: str | None = None) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to export", 3000)
            return
        track = self.project.tracks[index]
        events = self._track_events(track)
        if not events:
            self.statusBar().showMessage("This clip has no downbeats to export", 4000)
            return
        name = track.get("name") or "Clip"
        if path is None:
            folder = default_folder(self.current_path)
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                folder = default_folder()
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export downbeats",
                str(folder / f"{name}{FILE_SUFFIX}"),
                f"SonicLab downbeats (*{FILE_SUFFIX})",
            )
            if not path:
                return
        result = self.beat_jobs.result_for(track.get("path"))
        document = export_document(
            events,
            float(track.get("offset", 0.0) or 0.0),
            float(track.get("duration", 0.0) or 0.0),
            name=name,
            tempo_bpm=getattr(result, "tempo_bpm", None) or float(self.bpm.value()),
            time_signature=parse_time_signature(self.signature.currentText()),
        )
        try:
            written = write_document(path, document)
        except OSError as exc:
            QMessageBox.critical(self, "Export error", str(exc))
            return
        self.statusBar().showMessage(f"Exported downbeats to {written.name}", 4000)

    def _import_downbeats(self, path: str | None = None) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            self.statusBar().showMessage("Select a clip to import into", 3000)
            return
        if path is None:
            folder = default_folder(self.current_path)
            if not folder.is_dir():
                folder = default_folder()
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Import downbeats",
                str(folder),
                f"SonicLab downbeats (*{FILE_SUFFIX} *.json)",
            )
            if not path:
                return
        try:
            document = read_document(path)
        except DownbeatFileError as exc:
            QMessageBox.warning(self, "Import error", str(exc))
            return
        track = self.project.tracks[index]
        events = import_events(document, float(track.get("offset", 0.0) or 0.0))
        if not events:
            self.statusBar().showMessage("The file has no downbeats", 4000)
            return
        self._remember_edit()
        track["downbeats"] = events
        self._commit_edit()
        self._apply_detected_beats()
        self._apply_onsets()
        self.statusBar().showMessage(f"Imported {os.path.basename(path)}", 4000)

    def _clear_applied_downbeats(self) -> None:
        index = self._selected
        if not (0 <= index < len(self.project.tracks)):
            return
        track = self.project.tracks[index]
        if not track.get("downbeats"):
            return
        self._remember_edit()
        track.pop("downbeats", None)
        self._commit_edit()
        self._apply_detected_beats()
        self._apply_onsets()
        self.statusBar().showMessage("Cleared applied downbeats", 3000)

    @staticmethod
    def _beat_status(result) -> str:
        if result is None or result.status == STATUS_ERROR:
            return getattr(result, "message", "") or "Beat analysis failed"
        if result.status == STATUS_OK:
            cached = " (cached)" if result.cached else ""
            return f"Detected {len(result.downbeats)} downbeats{cached}"
        if result.status == STATUS_EMPTY:
            return "No beats detected"
        if result.status == STATUS_UNAVAILABLE:
            return getattr(result, "message", "") or "Madmom was not found, so downbeats were not detected."
        if result.status == STATUS_CANCELLED:
            return "Beat analysis cancelled"
        if result.status == STATUS_TIMEOUT:
            return "Beat analysis timed out"
        return result.message or "Beat analysis failed"

    def _place_recording(self, path: str, audio) -> None:
        self._remember_edit()
        duration = len(audio) / OUTPUT_RATE
        name = os.path.splitext(os.path.basename(path))[0]
        armed = [
            index
            for index in self._armed_at_record
            if 0 <= index < len(self.project.tracks)
        ]
        placed = False
        for index in armed:
            track = self.project.tracks[index]
            if track.get("empty") or float(track.get("duration", 0.0) or 0.0) <= 0.0:
                track.update(
                    name=name,
                    path=path,
                    start=self._record_started_at,
                    duration=duration,
                    offset=0.0,
                    _audio=audio,
                )
                track.pop("empty", None)
                placed = True
                break
        if not placed:
            insert_at = (armed[0] + 1) if armed else len(self.project.tracks)
            self.project.tracks.insert(
                insert_at,
                {
                    "name": name,
                    "path": path,
                    "start": self._record_started_at,
                    "duration": duration,
                    "offset": 0.0,
                    "_audio": audio,
                    "armed": False,
                },
            )
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._queue_beat_sources([path])
        self.statusBar().showMessage(f"Recorded {name}", 4000)

    def _media_dir(self) -> str:
        if self.current_path:
            directory = os.path.join(
                os.path.dirname(os.path.abspath(self.current_path)), "Media"
            )
        else:
            directory = os.path.join(
                os.path.expanduser("~"), ".local", "share", "soniclab", "recordings"
            )
        try:
            os.makedirs(directory, exist_ok=True)
            return directory
        except OSError:
            fallback = os.path.join(tempfile.gettempdir(), "soniclab", "recordings")
            os.makedirs(fallback, exist_ok=True)
            return fallback

    def _unique_take(self, directory: str) -> str:
        index = 1
        while True:
            path = os.path.join(directory, f"Take {index:02d}.wav")
            if not os.path.exists(path):
                return path
            index += 1

    def _choose_input(self) -> None:
        devices, error = input_devices()
        if error or not devices:
            QMessageBox.warning(
                self, "No input", error or "No input device is available."
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Input device")
        form = QFormLayout(dialog)
        device_box = QComboBox()
        for index, name, channels in devices:
            device_box.addItem(f"{name}  ({channels} ch)", (index, channels))
        channel_box = QComboBox()

        def refill():
            _index, channels = device_box.currentData()
            channel_box.clear()
            for number in range(int(channels)):
                channel_box.addItem(f"Channel {number + 1}", (number, False))
            if int(channels) >= 2:
                channel_box.addItem("Stereo 1–2", (0, True))

        device_box.currentIndexChanged.connect(lambda _index: refill())
        refill()
        form.addRow("Device", device_box)
        form.addRow("Input", channel_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.input_device, self.input_channels = device_box.currentData()
        self.input_channel, self.input_stereo = channel_box.currentData()
        self.statusBar().showMessage(device_box.currentText(), 3000)

    def _set_record_lamp(self, recording: bool) -> None:
        self.record_btn.setProperty("recording", bool(recording))
        self.record_btn.setIcon(record_icon("#F4E4E4" if recording else "#A65A5A"))
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.record_btn.update()

    def _about(self):
        QMessageBox.about(
            self,
            "SonicLab",
            "SonicLab\n\n"
            "A multitrack arrangement desk.\n\n"
            "Import, playback, clip editing, undo, recording, the mixer, "
            "beat detection, downbeat looping, and the spectrum, scope, and "
            "spectrogram views are active.\n\n"
            "The metronome, effects, and the sample library are not available yet.",
        )

    def showEvent(self, event):
        super().showEvent(event)
        if not self._canvases_settled:
            # Sizes are only known once the window is on screen.
            self._canvases_settled = True
            QTimer.singleShot(0, self._settle_canvases)

    def _settle_canvases(self) -> None:
        if not self.splitter2.isVisible():
            return
        self._size_canvases()
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            state = QSettings("SonicLab", "SonicLab").value("view/canvasSplit")
            if state is not None:
                self.canvas_split.restoreState(state)

    def closeEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.timer.stop()
        for timeline in self._timelines():
            timeline.clear_scene()
        self.beat_jobs.cancel()
        self.onset_jobs.cancel()
        self.stem_jobs.cancel()
        self.loader.shutdown()
        self.stem_splitter.shutdown()
        self.visualizer.set_active(False)
        self.recorder.stop_stream()
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            settings = QSettings("SonicLab", "SonicLab")
            settings.setValue("windowState", self.saveState())
            if self.splitter2.isVisible():
                settings.setValue("view/canvasSplit", self.canvas_split.saveState())
            settings.setValue("visualizer", self.visualizer.state())
        self.engine.close()
        super().closeEvent(event)
