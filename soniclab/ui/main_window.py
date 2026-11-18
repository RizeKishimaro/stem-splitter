import bisect
import math
import os
import tempfile
import time
import uuid
from collections import deque

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
from soniclab.core.downbeat_loop import arrangement_downbeats, downbeat_loop_region, loop_shortcut
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
    remove_line,
    set_span,
)
from soniclab.core.drums import pick_drum_track
from soniclab.core.session_timeline import ORIGIN_DOWNBEATS, ORIGIN_MANUAL, ORIGIN_MIGRATED_DOWNBEATS, SessionTimeline
from soniclab.core.track_registry import (
    PAUSED as PLAYBACK_PAUSED,
    PLAYING as PLAYBACK_PLAYING,
    TRACK_ORIGINAL,
    TRACK_SEGMENT,
    TRACK_STEM,
    ActionHistory,
    PlaybackRegistry,
    PlayingBlockState,
    SelectionRegistry,
    TrackRegistry,
    block_frames,
    block_id,
    block_node,
    is_block_id,
)
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
from soniclab.ui.dancer_view import DancerView
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


def sync_group(track) -> str:
    """The clock a track's blocks play on: its source song's.

    Stems of one split share their split's group; re-adding the same stem
    files (a cached split) joins that group too (see _add_stem_tracks).
    A track that was not split plays on the clock of its own file, shared
    with every piece cut from it. Unrelated songs do not share a clock,
    except the blocks of one selection, which do (unify_selection_groups)."""
    return stem_group(track) or f"file:{track.get('path') or id(track)}"


def loop_group(group: str) -> str:
    """The engine clock of a split's synced loop. Its stems loop together;
    the split's other stems keep the split's own clock."""
    return f"{group}/loop"


def unify_selection_groups(groups: dict) -> None:
    """Give one selection that spans several songs a single shared clock.

    A selection that is already one song is left on that song's clock, so a
    block started later still joins it. A synced loop stays on its own clock.
    Anything not in this selection is not in ``groups`` and is not moved.
    """
    plain = [name for name, entry in groups.items() if entry[2] is None]
    if len(plain) < 2:
        return
    members = {}
    start = end = None
    for name in plain:
        song, found, _loop, last = groups.pop(name)
        start = song if start is None else min(start, song)
        end = last if end is None else max(end, last)
        members.update(found)
    tag = ",".join(f"{key[0]}:{key[1]}" for key in sorted(members))
    groups[f"selection:{tag}"] = [start, members, None, end]


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
        self._input_depth = 0
        self._chosen_at_input: set[int] = set()
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
        self._frame_times = deque(maxlen=240)  # playhead frames, for the GUI frame rate
        # The session's musical timeline: the one authority for bars, tempo
        # and meter (see _sync_session). ``project.bars`` mirrors its bars
        # in seconds for code not yet on it and for older project readers.
        self.session_timeline = SessionTimeline()
        # An old project whose bars followed the drum downbeats: they are
        # frozen into the timeline once those downbeats are known.
        self._freeze_drum_bars = False
        # The arrangement's database (see soniclab.core.track_registry): every
        # song, stem and segment by stable id, the selected ids, the segments
        # the transport plays as a selection, and the recorded actions. Each
        # track dict carries its node's id ("node_id"); _sync_registry keeps
        # them in step while the rest of the window still works on tracks.
        self.track_registry = TrackRegistry()
        self.selection_registry = SelectionRegistry()
        self.playback_registry = PlaybackRegistry()
        self.action_history = ActionHistory()
        # Selection playback through the transport: the blocks it plays, by
        # track identity and bar (indices change), and the loop it set.
        self._projecting = False  # drawing the registry's selection on the canvases
        # Where each playing block is now: identity -> (track, bar).
        self._member_blocks: dict = {}
        # The bar (frames) every playing block is fitted to, locked when
        # block playback begins and kept until it ends.
        self._member_bar = None
        # The selection before the current burst of edits (see
        # _follow_selection), or None between bursts.
        self._selection_before = None
        # The downbeat grid each track was last cut on, by track identity, so
        # new beat results recut the blocks once (see _apply_detected_beats).
        self._cut_grids: dict = {}
        # Selected and combined tracks always loop, each row on its own cycle.
        # The loop button is the song's loop region only.
        self._block_loop = True
        self._moved_ids: set[int] = set()  # tracks a drag just moved, by identity
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
        if os.environ.get("SONICLAB_AUDIO_DIAG"):
            # Optional diagnostics mode: the report on stdout every 5 s.
            self._diag_timer = QTimer(self)
            self._diag_timer.setInterval(5000)
            self._diag_timer.timeout.connect(self._print_diagnostics)
            self._diag_timer.start()
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
        self.pause_btn = self._tool_button(pause_icon(), "Pause", self._pause_pressed)
        self.stop_btn = self._tool_button(stop_icon(), "Stop", self._stop_pressed)
        self.return_btn = self._tool_button(
            return_icon(), "Return to start", self._return_to_start
        )
        for button in (self.play_btn, self.pause_btn, self.stop_btn, self.return_btn):
            bar.addWidget(button)
        # Off: Pause and Stop act on everything. On: on the selected tracks
        # only, the others play on.
        self.selected_only_btn = self._text_button(
            "Selected", "Pause, Stop and Space act on the selected tracks only; the others play on (Ctrl+Shift+P)",
            lambda: None,
        )
        self.selected_only_btn.setCheckable(True)
        bar.addWidget(self.selected_only_btn)
        # Two or more selected always share one ▶. On: the rows' shared ▶ are
        # joined by the yellow line too (off by default).
        self.link_btn = self._text_button(
            "Link", "Show the yellow line joining the selected tracks' shared ▶ (Ctrl+Shift+L)", lambda: None
        )
        self.link_btn.setCheckable(True)
        self.link_btn.toggled.connect(self._set_linking)
        bar.addWidget(self.link_btn)

        self.record_btn = self._tool_button(
            record_icon("#A65A5A"), "Record", self._toggle_record, "recordBtn"
        )
        bar.addWidget(self.record_btn)
        self.loop_btn = self._tool_button(
            loop_icon(), "Loop the downbeat region", self._toggle_loop, "loopBtn"
        )
        self.loop_btn.setCheckable(True)
        bar.addWidget(self.loop_btn)
        # The metronome clicks the grid that plays: the drums' beats, or
        # while blocks play, the bar they are fitted to (Ctrl+M).
        self.metronome_btn = self._tool_button(
            metronome_icon(), "Metronome: clicks on the drums' beats (Ctrl+M)", self._toggle_metronome
        )
        self.metronome_btn.setCheckable(True)
        bar.addWidget(self.metronome_btn)

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
        self.dancer = DancerView()
        self.dancer.set_clock(self._dancer_clock)
        self.dancer.set_audio(self._dancer_audio)
        self.dancer.enabled_changed.connect(lambda _on: self._sync_capture())
        self.dancer_dock = self._dock("Dancer", self.dancer, features)
        self.dancer_dock.setObjectName("dancerDock")
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
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dancer_dock)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.stem_splitter_dock
        )
        self.tabifyDockWidget(self.mixer_dock, self.visualizer_dock)
        self.tabifyDockWidget(self.visualizer_dock, self.dancer_dock)
        self.tabifyDockWidget(self.dancer_dock, self.stem_splitter_dock)
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
            self.dancer_dock,
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
        self._action(view_menu, "Sprite Animation &Editor…", self._open_sprite_editor)
        self._action(view_menu, "Heart &Dancer", self._restore_heart_dancer)
        view_menu.addSeparator()
        self.link_action = QAction("Show Link Line", self, checkable=True)
        self.link_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        self.link_action.setToolTip("Join the selected tracks' shared ▶ with the yellow line")
        self.link_action.toggled.connect(self.link_btn.setChecked)
        self.link_btn.toggled.connect(self.link_action.setChecked)
        view_menu.addAction(self.link_action)
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
        self._action(audio_menu, "Play / Pause Selection", self._toggle_group, QKeySequence("Shift+Space"))
        self.selected_only_action = QAction("Pause && Stop Selected Only", self, checkable=True)
        self.selected_only_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.selected_only_action.toggled.connect(self.selected_only_btn.setChecked)
        self.selected_only_btn.toggled.connect(self.selected_only_action.setChecked)
        self.selected_only_btn.toggled.connect(
            lambda on: self.statusBar().showMessage(
                "Pause and Stop act on the selected tracks only" if on else "Pause and Stop act on everything", 3000
            )
        )
        audio_menu.addAction(self.selected_only_action)
        self.play_both_action = self._action(audio_menu, "Play Both Canvases", lambda: self._play())
        self._action(audio_menu, "Pause", self._pause_pressed)
        self._action(audio_menu, "Stop", self._stop_pressed)
        self._action(audio_menu, "Playing Voices…", self._show_voice_report)
        self._action(audio_menu, "Audio Diagnostics…", self._show_diagnostics)
        self._action(audio_menu, "Return to Start", self._return_to_start, QKeySequence("Home"))
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
        self._action(audio_menu, "Toggle Loop", self._toggle_loop, QKeySequence("Ctrl+L"))
        self._action(audio_menu, "Clear Loop", self._clear_loop)
        self.metronome_action = QAction("Metronome", self, checkable=True)
        self.metronome_action.setShortcut(QKeySequence("Ctrl+M"))
        self.metronome_action.triggered.connect(lambda _checked: self._toggle_metronome())
        audio_menu.addAction(self.metronome_action)

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
            timeline.on_input = self._canvas_input
            timeline.seek_requested.connect(
                lambda seconds, canvas=timeline.canvas: self._seek_on_canvas(canvas, seconds)
            )
            timeline.track_selected.connect(lambda index, view=timeline: self._on_view_selected(view, index))
            timeline.selection_edited.connect(self._follow_selection)
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
            timeline.group_picks = lambda view=timeline: self._canvas_group(view)
            timeline.confirm_clear = self._confirm_unselect_all
            timeline.downbeat_lines = self._downbeat_grid
            timeline.selection_units = lambda: len(self.selection_registry.ids)
            timeline.group_playing = self._group_playing
            timeline.group_play_requested.connect(self._toggle_group)
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
            timeline.stem_unselect_requested.connect(
                lambda index, view=timeline: self._on_stem_unselect(view, index)
            )
            timeline.stem_half_split_requested.connect(self._split_stem_in_half)
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
            # A row's header speaks for every clip on that row (pieces moved
            # onto it share its lane; only the first has a header you see).
            headers.mute_changed.connect(lambda index, on: self._set_row(index, on, self._set_muted))
            headers.solo_changed.connect(lambda index, on: self._set_row(index, on, self._set_solo))
            headers.arm_changed.connect(self._set_armed)
            headers.edit_finished.connect(self._commit_edit)
            headers.resize_committed.connect(self._on_track_resized)
            headers.unavailable.connect(self._unavailable)
            headers.rename_requested.connect(self._rename_track_at)
            headers.menu_requested.connect(self._track_menu)
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
            # Otherwise Esc stops everything playing (the song and every stem
            # block, like Stop) and deselects the clips and picked blocks.
            if QApplication.activeModalWidget() is not None or isinstance(
                QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox, QComboBox)
            ):
                return False
            if not self._confirm_unselect_all():
                return True  # kept: nothing stops, nothing is unselected
            playing = self.engine.playing or bool(self.engine.previews())
            if playing:
                self._stop()
            if self._deselect_all() or playing:
                self.statusBar().showMessage("Stopped; nothing selected", 2000)
                return True
            return False
        if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if QApplication.activeModalWidget() is not None:
                return False
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QAbstractSpinBox, QComboBox)):
                return False
            if not event.isAutoRepeat():
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._toggle_group()  # Shift+Space: the selection's shared ▶ / ⏸
                else:
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
            self._pause_pressed()
        else:
            self._play_hovered()

    def _play_hovered(self) -> None:
        """Play the canvas the mouse is (or was last) over, from its own
        playhead. With one canvas this is plain Play. With blocks picked (or
        playing) it plays those blocks instead (see _play_members)."""
        if self._play_members():
            return
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
        # Point at the new track; do not select it whole (see _select_after_split).
        self._on_track_selected(len(self.project.tracks) - 1)
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
        # These stems share one playback clock. The same files added again
        # (a cached split of the same song) join the clock already there.
        paths = {os.path.abspath(stems[stem]) for stem in order}
        group = next(
            (stem_group(track) for track in self.project.tracks if stem_group(track) and track.get("path") in paths),
            None,
        ) or uuid.uuid4().hex
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
                    "stem_type": stem,
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
        kept = self._selection_by_track()
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
        self._select_after_split(kept, len(self.project.tracks) - len(added), select=False)
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
        kept = self._selection_by_track()
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
        self._select_after_split(kept, len(self.project.tracks) - len(added), select=False)
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
        self._sync_session()
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
        if self._preview_tracks:
            # Blocks were redrawn, maybe moved or cut: every playing block
            # voice must still match a block on screen, or stop.
            self._prune_previews()
        self._sync_registry()
        self._project_selection()  # the canvases show the registry's selection, at the new indices
        self._refresh_members()

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
            if self.engine.members:
                self._place_members_on_tracks()  # same batch: never heard on a stale clip index
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

    def _drum_grid(self) -> list[float] | None:
        """The drums' downbeats in arrangement seconds (the lead's bars), or
        None without a drum track that has them."""
        drum = self._drum_index()
        if drum is None:
            return None
        lines = self._downbeat_grid_preview(self.project.tracks[drum])  # reading only: no recut bookkeeping
        return lines if lines and len(lines) >= 2 else None

    def _drum_bar_at(self, grid, seconds: float):
        """``(bar start, bar length)`` of the drums' bar holding ``seconds``,
        the last bar's length carried on past the drums' end; None before
        the first downbeat."""
        if seconds < grid[0]:
            return None
        index = bisect.bisect_right(grid, seconds) - 1
        if index < len(grid) - 1:
            return grid[index], grid[index + 1] - grid[index]
        step = grid[-1] - grid[-2]
        laps = int((seconds - grid[-1]) // step)
        return grid[-1] + laps * step, step

    def _musical_position(self, seconds):
        """Bar and beat at ``seconds``. The beat is counted on the drums'
        own bars (they lead), the bar number from the session timeline, so
        it reads like the ruler; without drums, both from the timeline."""
        grid = self._drum_grid()
        held = self._drum_bar_at(grid, float(seconds)) if grid else None
        if held is not None:
            start, step = held
            numerator, _denominator = parse_time_signature(self.signature.currentText())
            beat = min(numerator, int((float(seconds) - start) / (step / numerator)) + 1)
            timeline = self.session_timeline
            bar = timeline.tick_to_position(timeline.seconds_to_tick(max(0.0, start + 1e-3))).bar
            return f"{max(0, bar):03d}.{max(1, beat)}"
        timeline = self.session_timeline
        position = timeline.tick_to_position(timeline.seconds_to_tick(max(0.0, float(seconds))))
        return f"{max(0, position.bar):03d}.{position.beat}"

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
        seconds = self.engine.heard_position() if playing else self.engine.position
        lead = self._lead_member_seconds(seconds)
        return seconds if lead is None else lead

    def _lead_member_seconds(self, seconds: float) -> float | None:
        """While blocks play, the clock, bar counter and ruler playhead show
        what plays, not the transport's running count: the arrangement time
        of the block that has played longest (the first started, then the
        top track), at transport time ``seconds``. None when no block sounds."""
        members = self.engine.members
        if not members:
            return None
        position = int(round(float(seconds) * OUTPUT_RATE))
        best = None
        for entry in members.values():
            offset = AudioEngine.member_offset(entry, position)
            if offset is None:
                continue
            rank = (entry[3], entry[0])
            if best is None or rank < best[0]:
                best = (rank, (entry[1] + offset) / OUTPUT_RATE)
        return None if best is None else best[1]

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
        self._update_transport_cursors(playing)
        return playing, seconds

    def _update_transport_cursors(self, playing: bool) -> None:
        """While the song plays, each block it plays shows its own cursor,
        placed from the one transport position (no clock of its own).
        Members show theirs with their ▶ / ⏸ (see _sync_previews). The
        canvas line is the ruler handle only, always."""
        lanes = None if self._members_active() else self._transport_lanes(playing)
        for timeline in self._timelines():
            if lanes is None:
                timeline.set_transport_cursors(None, None)
            else:
                timeline.set_transport_cursors(self._canvas_positions[timeline.canvas], lanes.get(timeline, {}))
            if getattr(timeline, "_playhead_bands", None) != []:
                timeline.set_playhead_bands([])

    def _transport_lanes(self, playing: bool):
        """``{view: {track index: bars or None}}`` of the blocks that show
        the transport cursor, or None for no cursors (the full line): while
        the arrangement plays, every block of every clip that is heard.
        Stopped, or a block auditioning on its own: none."""
        if not playing:
            return None
        tracks = self.project.tracks
        focus = self._track_focus
        focused = tracks[focus] if focus is not None and 0 <= focus < len(tracks) else None
        heard = []
        for index, track in enumerate(tracks):
            if self._canvas_focus is not None and track_canvas(track) != self._canvas_focus:
                continue
            if focused is not None and track is not focused:
                continue
            if not track.get("muted", False):
                heard.append((index, track))
        if any(track.get("solo", False) for _index, track in heard):
            heard = [(index, track) for index, track in heard if track.get("solo", False)]
        lanes: dict = {}
        for index, _track in heard:
            lanes.setdefault(self._view_for(index), {})[index] = None
        return lanes

    def _animate_playheads(self) -> None:
        """One playhead frame. Reads the engine's snapshot; never waits on audio."""
        now = time.monotonic()
        dt = min(0.1, now - self._last_frame) if self._last_frame else PLAYHEAD_FRAME_MS / 1000.0
        self._last_frame = now
        self._frame_times.append(now)
        self._retire_completed_members()
        shown = self._update_playheads()
        playing = shown[0]
        previews = self._sync_previews() if (self._preview_tracks or self.engine.previews() or self.engine.members) else {}
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
        # One controller at a time: the song plays every stem itself, so a
        # block voice still sounding would play its stem a second time.
        if self.engine.previews():
            self.engine.preview_stop_all()
            self._preview_tracks.clear()
            self._sync_previews()
        try:
            self.engine.play()
        except Exception as exc:
            QMessageBox.critical(
                self, "Audio output error", f"Could not start audio output.\n\n{exc}"
            )
        if self.engine.playing:
            self.frame_timer.start()
        self._sync_transport_lamp()

    def _linking(self) -> bool:
        return bool(getattr(self, "link_btn", None) and self.link_btn.isChecked())

    def _set_linking(self, on: bool) -> None:
        for view in self._timelines():
            view.show_link = bool(on)
            view._update_combined_play()
        self.statusBar().showMessage("Yellow link line shown" if on else "Yellow link line hidden", 3000)

    def _scope_keys(self) -> list | None:
        """With "Selected" on and blocks playing: the playing members of
        the selection, which Pause, Stop and Space then act on alone. None
        otherwise (they act on everything)."""
        if not self.selected_only_btn.isChecked() or not self._members_active():
            return None
        members = self.engine.members or {}
        keys = [key for key in self._group_members() if key in members]
        return keys or None

    def _pause_pressed(self) -> None:
        keys = self._scope_keys()
        if keys is None:
            self._pause()
            return
        for key in keys:
            self._hold_member(key, True)  # the others play on
        self._sync_previews()

    def _stop_pressed(self) -> None:
        keys = self._scope_keys()
        if keys is None:
            self._stop()
            return
        for key in keys:
            if key in (self.engine.members or {}):
                self._remove_member(key)  # the others play on

    def _pause(self):
        # The one clock stops: every block member holds where it is.
        self.engine.pause()
        self._sync_previews()
        self._sync_transport_lamp()
        self._sync_canvas_play_buttons()

    def _stop(self):
        # Stop is the panic button: the song and every stem block. Every
        # playing block stops too; the selection itself stays.
        self._end_members()
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
        if bounds is not None and not self._members_active():
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
        """The song's tempo from the session's stored bars, or None when the
        grid is regular from the tempo field."""
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
        # Bars keep their ticks; the tempo map follows the stretched audio.
        self._sync_session()
        self.session_timeline.scale_tempo(1.0 / ratio)
        self.project.bars = self.session_timeline.legacy_lines()
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
        self._remember_edit()
        self._sync_session()
        self.session_timeline.reset_tempo(float(self.bpm.value()))  # one tempo, then even bars on it
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

    def _selection_by_track(self) -> list:
        """The picked blocks and selected clips of each canvas, by track
        identity, so they survive an edit that inserts or reorders tracks."""
        tracks = self.project.tracks
        kept = []
        for view in self._timelines():
            picks = {id(tracks[i]): bars for i, bars in view.picks().items() if 0 <= i < len(tracks)}
            clips = {id(tracks[i]) for i in view.selected_indices() if 0 <= i < len(tracks)}
            kept.append((view, picks, clips))
        return kept

    def _restore_selection(self, kept, drop=()) -> bool:
        """Put back what ``_selection_by_track`` saved, at the tracks' new
        indices, leaving out the tracks in ``drop`` (and any that went).
        Returns whether anything is selected again."""
        where = {id(track): index for index, track in enumerate(self.project.tracks)}
        restored = False
        for view, picks, clips in kept:
            selected = {where[key] for key in clips if key in where and key not in drop}
            view.scene_obj.blockSignals(True)
            try:
                for clip in view._clip_items():
                    clip.setSelected(clip.clip_index in selected)
            finally:
                view.scene_obj.blockSignals(False)
            view.set_picks({where[key]: bars for key, bars in picks.items() if key in where and key not in drop})
            restored = restored or bool(selected or view.picks())
        return restored

    def _select_after_split(self, kept, index: int, drop=(), select: bool = True) -> None:
        """After a split: the tracks that were selected stay selected (and
        their loop keeps going); ``index`` becomes the track the clip
        commands act on. With nothing selected before, ``index`` is selected,
        unless ``select`` is False: new tracks (stems, imports) are only
        pointed at. Selecting a whole stem selects all its blocks, which
        trades their own ▶ for the shared one, so a fresh stem would show
        up without its play buttons."""
        if self._restore_selection(kept, drop) or not select:
            self._on_track_selected(index)
        else:
            self._select_track(index)

    def _reselect_after_move(self, kept, moved, sources=(), pick: bool = False) -> None:
        """After a move: the selection stays what it was, by track identity,
        and the moved clips join it in their new place (picked block by
        block with ``pick``, else selected whole). A move used to select only
        the moved clip and drop everything else, on both canvases, so stems
        selected before the drag were no longer played with it.

        ``sources`` are tracks whose old picks named the blocks that moved
        away; their picks are not carried over, nor the moved clips' own
        old picks, whose bar numbers belong to where they were."""
        dropped = {id(track) for track in moved} | {id(track) for track in sources}
        self._moved_ids = {id(track) for track in moved}
        self._restore_selection(kept, drop=dropped)
        tracks = self.project.tracks
        wanted = {id(track) for track in moved}
        indices = [i for i, track in enumerate(tracks) if id(track) in wanted]
        for index in indices:
            view = self._view_for(index)
            clip = view._clip_for(index)
            if clip is None:
                continue
            if pick:
                picks = view.picks()
                picks[index] = {int(span[0]) for span in view._spans_of(clip)}
                view.set_picks(picks)
            else:
                clip.setSelected(True)
        if indices:
            self._on_track_selected(min(indices))  # the clip commands act on it

    def _confirm_unselect_all(self) -> bool:
        """Ask before everything is unselected (a click on empty canvas, or
        Esc): unselected blocks stop playing. True with nothing selected."""
        if not any(view.selected_indices() or view.picks() for view in self._timelines()):
            return True
        answer = QMessageBox.question(
            self,
            "Unselect all",
            "Unselect every selected track and block?\nBlocks that are unselected stop playing.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _deselect_all(self) -> bool:
        """Clear the selected clips and picked blocks on both canvases, and
        stop the paused blocks (their grey tint and line go). Blocks still
        playing go on. Returns whether there was anything to clear."""
        views = self._timelines()
        selected = any(view.selected_indices() or view.picks() for view in views)
        paused = [key for key, state in self.engine.previews().items() if not state[0]]
        for view in views:
            view.set_picks({})
            self._clear_selection(view)
        for key in paused:
            self.engine.preview_stop(key)
        if paused:
            self._sync_previews()
        if selected:
            self._on_track_selected(-1)
        return selected or bool(paused)

    @staticmethod
    def _clear_selection(view) -> None:
        """Deselect everything in ``view`` without reporting "no selection".
        Its block controls are refreshed all the same: with the signals
        blocked, a shared ▶ would stay on a clip no longer selected, and a
        click on it would play that clip alone instead of with the rest."""
        if view is None:
            return
        view.scene_obj.blockSignals(True)
        view.scene_obj.clearSelection()
        view.scene_obj.blockSignals(False)
        view._update_combined_play()
        view.viewport().update()

    def _on_view_selected(self, view, index: int) -> None:
        # Selecting on one canvas adds to the selection: the other canvas
        # keeps its selected clips, as it keeps its picked blocks, so a clip
        # moved there (selected by the drop) still plays with the stems
        # selected here. _select_track still clears the other canvas when
        # one track is chosen on its own.
        if index < 0 and view.peer is not None and view.peer.selected_indices():
            return  # the selection is on the other canvas
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

    def _row_mates(self, index: int) -> list[int]:
        """Every track on the same row as track ``index`` (itself first)."""
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return [index]
        rows, _primaries = self._view_for(index).lane_rows(tracks)
        row = rows[index] if index < len(rows) else -1
        if row < 0:
            return [index]
        return [index] + [other for other, at in enumerate(rows) if at == row and other != index]

    def _set_row(self, index: int, on: bool, setter) -> None:
        """Mute or solo a row: every clip on it, as one undo step."""
        mates = self._row_mates(index)
        if len(mates) == 1:
            setter(index, on)
            return
        self._remember_edit()
        frozen, self._history_frozen = self._history_frozen, True  # each clip's own edit is part of this one
        try:
            for mate in mates:
                setter(mate, on)
        finally:
            self._history_frozen = frozen
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
        kept = self._selection_by_track()
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
        self._reselect_after_move(kept, [tracks[i] for i in group])
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
        kept = self._selection_by_track()
        self._remember_edit()
        try:
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
        except Exception:
            self._abandon_edit()
            raise
        grabbed_track = moving[indices.index(grabbed)] if grabbed in indices else None
        if grabbed_track is not None:
            # Dropped where the user can see it: no scroll. The clips moved
            # join the selection there; what else was selected stays so.
            self._reselect_after_move(kept, moving)
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
        # The top block of the group stops at the first row (as the drag
        # preview showed): one row shift for all, so their layout is kept.
        target_row = max(int(target_row), -min(source_rows[int(b[0])] - base for b in blocks))
        grouped: dict[int, list] = {}
        for index, local_start, local_duration, bar in blocks:
            grouped.setdefault(int(index), []).append((float(local_start), float(local_duration), int(bar)))
        # One shift for all of them: the earliest stops at the timeline's
        # start, so none is pushed on its own and their spacing is kept.
        earliest = min(float(tracks[int(b[0])].get("start", 0.0) or 0.0) + float(b[1]) for b in blocks)
        shift = max(float(shift), -earliest)
        self._remember_edit()
        try:
            self._move_block_pieces(
                tracks, grouped, source_rows, base, hosts, target_row, shift, canvas, target
            )
        except Exception:
            self._abandon_edit()
            raise

    def _move_block_pieces(self, tracks, grouped, source_rows, base, hosts, target_row, shift, canvas, target) -> None:
        kept = self._selection_by_track()
        sources = [tracks[index] for index in grouped]
        pieces = []  # (piece, relative row, whole)
        replaced = {}  # id(track) -> remaining segments
        for index, items in grouped.items():
            track = tracks[index]
            self._keep_stem_grid(track)
            relative = source_rows[index] - base
            start = float(track.get("start", 0.0) or 0.0)
            offset = float(track.get("offset", 0.0) or 0.0)
            duration = float(track.get("duration", 0.0) or 0.0)
            items.sort()
            # The whole clip moves as that same item. One bar of it moves
            # alone; the rest of the track stays.
            whole_clip = len(items) == 1 and items[0][0] <= 0.05 and items[0][1] >= duration - 0.05
            if whole_clip or track.get("bar_block"):
                pieces.append((track, relative, True))
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
                piece.pop("splits", None)
                pieces.append((piece, relative, False))
            segments = []
            cursor = 0.0
            for local_start, local_duration, _bar in items:
                if local_start - cursor >= 0.05:
                    segments.append((cursor, local_start - cursor))
                cursor = max(cursor, local_start + local_duration)
            if duration - cursor >= 0.05:
                segments.append((cursor, duration - cursor))
            replaced[id(track)] = (track, segments, start, offset)
        rebuilt = []
        for track in tracks:
            entry = replaced.get(id(track))
            if entry is None:
                rebuilt.append(track)
                continue
            track, segments, start, offset = entry
            if not segments:
                continue
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
                    part.pop("splits", None)
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
        # The moved bars stay picked in their new place, beside whatever else
        # was selected; dropped where the user can see it: no scroll.
        self._reselect_after_move(kept, moved, sources, pick=True)
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

    def _on_stem_unselect(self, view, index: int) -> None:
        clip = view._clip_for(index)
        view._unselect_handled = False
        if clip is None or not self._right_click_unselects(view, clip):
            return
        view._unselect_handled = True
        self._drop_stem_from_selection(view, clip)

    def _track_menu(self, index: int, global_pos) -> None:
        """Right-click on a track's name: what can be done to the track."""
        if not (0 <= index < len(self.project.tracks)):
            return
        self._select_track(index)
        menu = QMenu(self)
        menu.addAction("Stop", lambda: self._stop_track_voices(index))
        menu.addAction("Stop All", self._stop)
        menu.addSeparator()
        for action in (self.rename_action, self.mute_action, self.solo_action, self.move_start_action):
            menu.addAction(action)
        menu.addSeparator()
        for action in (self.duplicate_action, self.analyze_action, self.onset_action, self.reference_action):
            menu.addAction(action)
        self._sync_clip_actions()
        menu.addMenu(self.downbeat_menu)
        for action in (self.canvas_move_action, self.move_up_action, self.move_down_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.delete_action)
        menu.exec(global_pos)

    def _block_menu(self, view, clip, block, pos) -> None:
        """Right-click on a block: the few things done most, and the rest
        under More."""
        index, bar = block
        picked = bar in view.picks().get(index, ())
        menu = QMenu(self)
        if picked or clip.isSelected():
            menu.addAction("Unselect", lambda: self._unselect_block(view, clip, bar))
        else:
            menu.addAction("Select", lambda: view.toggle_pick(index, bar))
        if sum(len(bars) for other in self._timelines() for bars in other.picks().values()) > 1:
            menu.addAction("Unselect All", self._unselect_all_blocks)
        menu.addSeparator()
        span = next((span for span in view._spans_of(clip) if int(span[0]) == bar), None)
        if span is not None:
            menu.addAction(
                "Split in Half",
                lambda: self._split_stem_in_half(index, float(span[3]), float(span[4]) - float(span[3])),
            )
        menu.addAction("Repeat / Loop", lambda: self._loop_block(view, index, bar))
        menu.addAction("Stop", lambda: self._stop_block_at(index, bar))
        menu.addAction("Stop All", self._stop)
        menu.addSeparator()
        more = menu.addMenu("More")
        self._fill_more_menu(more, view, clip, block, view.bar_under(pos))
        # Opening the menu leaves the selection as it is. Only More, whose
        # commands act on the selected track, selects this one when opened.
        more.aboutToShow.connect(lambda index=clip.clip_index: self._target_track(index))
        menu.exec(view.viewport().mapToGlobal(pos))

    def _target_track(self, index: int) -> None:
        """Make ``index`` the track the clip commands act on, unless it is
        already selected (then the selection stays as it is)."""
        if index not in self._selection():
            self._select_track(index)
        self._sync_clip_actions()

    def _unselect_block(self, view, clip, bar: int) -> None:
        before = self._chosen_tracks()
        self._unselect_block_only(view, clip, bar)
        self._stop_unselected(before - self._chosen_tracks())

    def _unselect_block_only(self, view, clip, bar: int) -> None:
        index = clip.clip_index
        picks = view.picks()
        if bar in picks.get(index, ()):
            view.toggle_pick(index, bar)
        if not view.picks().get(index) and clip.isSelected():
            clip.setSelected(False)

    def _unselect_all_blocks(self) -> None:
        before = self._chosen_tracks()
        for view in self._timelines():
            view.set_picks({})
            view.scene_obj.clearSelection()
        self._stop_unselected(before - self._chosen_tracks())

    def _loop_block(self, view, index: int, bar: int) -> None:
        """Repeat this block: a split stem loops it (with any other picked
        blocks) in sync; another track loops the song over it."""
        tracks = self.project.tracks
        if 0 <= index < len(tracks) and stem_group(tracks[index]):
            if bar not in view.picks().get(index, ()):
                view.add_pick(index, bar)
            self._loop_picked_stems()
            return
        times = view.block_times(index, bar)
        if times is not None:
            self._engage_loop(*times)

    def _fill_more_menu(self, menu, view, clip, block, bar) -> None:
        """Everything the block menu used to hold."""
        states = self.engine.previews()
        key = self._block_voice(*block, states)
        state = states.get(key)
        playing = state is not None and state[0]
        menu.addAction(f"{'Pause' if playing else 'Play'} This Block", lambda: self._toggle_block(*block))
        if any(other.picks() for other in self._timelines()):
            menu.addAction("Play Picked Blocks Together", self._play_picked_blocks)
            menu.addAction("Move Picked Blocks to Other Canvas", self._move_selected_to_other_canvas)
        if stem_loop(self.project.tracks[block[0]]) is not None:
            menu.addAction("Remove This Stem from the Synced Loop", lambda index=block[0]: self._remove_from_stem_loop(index))
            menu.addAction("Clear the Synced Loop", lambda index=block[0]: self._clear_stem_loop(index))
        menu.addSeparator()
        if bar >= 0:
            self._add_bar_actions(menu, bar)
            menu.addSeparator()
        if self.project.tracks[clip.clip_index].get("bars"):
            menu.addAction("Cut This Stem on the Song's Bars", lambda index=clip.clip_index: self._use_song_bars(index))
        for action in (self.rename_action, self.move_start_action, self.mute_action, self.solo_action):
            menu.addAction(action)
        menu.addSeparator()
        for action in (self.split_action, self.duplicate_action, self.analyze_action, self.onset_action,
                       self.reference_action, self.nudge_action):
            menu.addAction(action)
        menu.addMenu(self.downbeat_menu)
        for action in (self.canvas_move_action, self.move_up_action, self.move_down_action):
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self.delete_action)

    def _right_click_unselects(self, view, clip) -> bool:
        """A right-click drops a split stem that is selected, and one of
        several selected tracks. A track that is not selected keeps its menu."""
        index = clip.clip_index
        tracks = self.project.tracks
        picked = index in view.picks()
        if not clip.isSelected() and not picked:
            return False
        if 0 <= index < len(tracks) and stem_group(tracks[index]):
            return True
        selected = [item for other in self._timelines() for item in other.selected_indices()]
        return clip.isSelected() and len(selected) > 1

    def _drop_stem_from_selection(self, view, clip) -> None:
        index = clip.clip_index
        picks = view.picks()
        if index in picks:
            del picks[index]
            view.set_picks(picks)
        was_selected = clip.isSelected()
        if was_selected:
            clip.setSelected(False)
        tracks = self.project.tracks
        name = tracks[index].get("name", "Track") if 0 <= index < len(tracks) else "Track"
        self.statusBar().showMessage(f"“{name}” is no longer selected", 3000)
        if was_selected and self._selected != index:
            return
        if self._selected == index:
            rest = [item for other in self._timelines() for item in other.selected_indices()]
            if not rest:
                for other in self._timelines():
                    if other.picks():
                        rest = [min(other.picks())]
                        break
            self._on_track_selected(rest[0] if rest else -1)

    def _clip_context_menu(self, pos, view=None):
        view = view or self.timeline
        if getattr(view, "_skip_context_menu", False):
            view._skip_context_menu = False
            return
        bar = view.bar_under(pos)
        clip = view.clip_at(pos)
        block = view.block_at(pos)
        if clip is not None and block is not None:
            self._block_menu(view, clip, block, pos)
            return
        menu = QMenu(self)
        states = self.engine.previews()
        if block is not None:
            key = self._block_voice(*block, states)
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
        if self.project.tracks[clip.clip_index].get("bars"):
            menu.addAction("Cut This Stem on the Song's Bars", lambda index=clip.clip_index: self._use_song_bars(index))
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
        self.session_timeline = SessionTimeline()
        self._freeze_drum_bars = False
        self._end_members()
        self.track_registry = TrackRegistry()
        self.selection_registry = SelectionRegistry()
        self.playback_registry = PlaybackRegistry()
        self.action_history = ActionHistory()
        self._sync_history()
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
        self._sync_session()
        saved_project.timeline = self.session_timeline.to_dict()
        saved_project.bars = self.session_timeline.legacy_lines()  # for older SonicLab builds
        self._sync_registry()
        serial_tracks = [
            {key: value for key, value in track.items() if not key.startswith("_")}
            for track in self.project.tracks
        ]  # again: the sync gave every track its node id
        saved_project.registry = {"tracks": self.track_registry.to_dict(), "history": self.action_history.to_dict()}
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
            self._open_session_timeline()
            self._open_registry()
            self._refresh_timeline()
            if self._selected >= 0:
                # Point at it; a whole selected stem would trade its own ▶ for the shared one.
                self._on_track_selected(self._selected)
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

    @property
    def history(self) -> ActionHistory:
        """The project's one undo history (saved with it)."""
        return self.action_history

    # In-memory track fields an edit can change and undo must restore. The
    # other "_" fields (audio, analysis caches) are rebuilt, not recorded.
    _RECORDED_RUNTIME = ("_alignment",)

    def _snapshot(self) -> dict:
        """The project as an edit leaves it, in a form that can be saved:
        tracks without their audio (see _with_runtime)."""
        return {
            "tracks": [
                {key: value for key, value in track.items() if not key.startswith("_") or key in self._RECORDED_RUNTIME}
                for track in self.project.tracks
            ],
            "master_gain": self.engine.output_gain,
            "master_muted": self.engine.master_muted,
            "bpm": float(self.bpm.value()),
            "signature": self.signature.currentText(),
            "name": self.project.name,
            "bars": list(self.project.bars),
            "timeline": self.session_timeline.to_dict(),
        }

    def _snapshot_changed(self, before: dict, after: dict) -> bool:
        if (
            before["master_gain"] != after["master_gain"]
            or before["master_muted"] != after["master_muted"]
            or before["bpm"] != after["bpm"]
            or before["signature"] != after["signature"]
            or before["name"] != after["name"]
            or before.get("bars") != after.get("bars")
            or before.get("timeline") != after.get("timeline")
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
            "bars",
            "splits",
            "node_id",
            "_alignment",
        )
        for left, right in zip(before["tracks"], after["tracks"]):
            for key in keys:
                if left.get(key) != right.get(key):
                    return True
        return False

    def _with_runtime(self, tracks) -> list[dict]:
        """Tracks from a recorded snapshot, with their in-memory fields
        (audio, analysis) back: from the track with the same node id or
        file now in the project, else read from disk as a load does."""
        current = self.project.tracks
        by_node = {track.get("node_id"): track for track in current if track.get("node_id")}
        by_path = {track.get("path"): track for track in current if track.get("path")}
        for track in tracks:
            if track.get("_audio") is not None:
                continue
            donor = by_node.get(track.get("node_id")) or by_path.get(track.get("path"))
            if donor is not None:
                for key, value in donor.items():
                    if key.startswith("_") and key not in self._RECORDED_RUNTIME and key not in track:
                        track[key] = value
                continue
            source = track.get("path")
            audio = np.zeros((1, 2), dtype=np.float32)
            if source and os.path.exists(source):
                try:
                    raw, rate = sf.read(source, dtype="float32", always_2d=True)
                    audio = AudioEngine.resample(raw, rate)
                except Exception:
                    pass
            factor = self._stretch_of(track)
            if factor != 1.0 and len(audio) > 1:
                track["_source_audio"] = audio
                audio = stretch_audio(audio, factor)
            track["_audio"] = audio
        return tracks

    def _remember_edit(self) -> None:
        if self._history_frozen or self._gesture_before is not None:
            return
        self._gesture_before = self._snapshot()
        self._gesture_actions = self.action_history.done
        self.action_history.begin()  # what this gesture records is one undo step

    def _commit_edit(self) -> None:
        if self._history_frozen or self._gesture_before is None:
            return
        before = self._gesture_before
        self._gesture_before = None
        after = self._snapshot()
        recorded = self.action_history.recorded_since(getattr(self, "_gesture_actions", self.action_history.done))
        if self._snapshot_changed(before, after) or recorded > 0:
            # One gesture, one undo step: the project before and after it,
            # with any registry actions it recorded (a split) in its group.
            self.action_history.record_edit(before, after)
        self.action_history.end()
        self._sync_history()

    def _abandon_edit(self) -> None:
        """A transfer that failed before it was committed: put the tracks back."""
        snap = self._gesture_before
        if snap is None:
            return
        self._gesture_before = None
        # Registry actions it recorded are taken back and dropped, so the
        # history never holds half an edit.
        while self.action_history.recorded_since(getattr(self, "_gesture_actions", self.action_history.done)) > 0:
            self.action_history.undo(self.track_registry)
            del self.action_history.actions[self.action_history.done:]
        self.action_history.end()
        self._restore_snapshot(snap)

    def _restore_snapshot(self, snap: dict) -> None:
        self._history_frozen = True
        self._gesture_before = None
        try:
            self.project.tracks = self._with_runtime([dict(item) for item in snap["tracks"]])
            self.project.name = snap.get("name") or "Untitled"
            self.project.bars = list(snap.get("bars") or [])
            if snap.get("timeline"):
                self.session_timeline = SessionTimeline.from_dict(snap["timeline"])
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
            # Point at the same track again without selecting it whole.
            self._on_track_selected(self._selected if self._selected >= 0 else -1)
        finally:
            self._history_frozen = False

    def _undo(self) -> None:
        """Undo the last gesture: its registry actions (the same node ids
        come back) and the project as it was before it."""
        undone = self.action_history.undo(self.track_registry)
        edit = next((action for action in undone if action.snapshot_before is not None), None)
        if edit is not None:
            self._restore_snapshot(edit.snapshot_before)
        elif undone:
            self._refresh_timeline()
        self._sync_history()

    def _redo(self) -> None:
        redone = self.action_history.redo(self.track_registry)
        edit = next((action for action in redone if action.snapshot_after is not None), None)
        if edit is not None:
            self._restore_snapshot(edit.snapshot_after)
        elif redone:
            self._refresh_timeline()
        self._sync_history()

    def _sync_history(self) -> None:
        if not hasattr(self, "undo_action"):
            return
        self.undo_action.setEnabled(self.action_history.can_undo)
        self.redo_action.setEnabled(self.action_history.can_redo)

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
        self._keep_stem_grid(track)
        start = float(track.get("start", 0.0))
        duration = float(track.get("duration", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        local_start = max(0.0, min(float(local_start), duration))
        local_end = max(local_start, min(local_start + float(local_duration), duration))
        if local_end - local_start < 0.05 or duration <= 0:
            return
        self._remember_edit()
        try:
            self._apply_block_move(
                track, index, start, offset, duration, local_start, local_end,
                new_start, share_lane, insert_at, bar, canvas, host,
            )
        except Exception:
            self._abandon_edit()
            raise

    def _apply_block_move(
        self, track, index, start, offset, duration, local_start, local_end,
        new_start, share_lane, insert_at, bar, canvas, host,
    ) -> None:
        kept = self._selection_by_track()
        if track.get("bar_block") or local_end - local_start >= duration - 0.05:
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
            self._reselect_after_move(kept, [track])
            return
        # Only this bar leaves. The audio stays the same array; nothing is
        # written out. The rest of the track stays on its row.
        piece = dict(track)
        piece["start"] = max(0.0, float(new_start))
        piece["offset"] = offset + local_start
        piece["duration"] = local_end - local_start
        piece["armed"] = False
        piece["bar_block"] = True
        piece.pop("splits", None)
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
            tail.pop("splits", None)
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
        if host is not None:
            self._join_row(piece, host)
        self._reload_engine()
        self._commit_edit()
        self._refresh_timeline()
        self._reselect_after_move(kept, [piece], [track])
        self.statusBar().showMessage(f"Moved bar {int(bar) + 1}", 2000)

    def _keep_stem_grid(self, track: dict) -> None:
        """Before a stem is cut into pieces, give it a bar grid of its own.

        The grid is in the stem's audio-file time, so every piece copied
        from the track carries the same grid and it moves with each piece.
        Applied downbeats of the stem itself give it; otherwise the song's
        bars where the stem sits now. A stem that has a grid keeps it.
        """
        if track.get("bars"):
            return
        shift = float(track.get("offset", 0.0) or 0.0) - float(track.get("start", 0.0) or 0.0)
        own = [float(time) for time, position in (track.get("downbeats") or []) if int(position) == 1]
        if len(own) >= 2:
            track["bars"] = sorted(round(time, 6) for time in own)
            return
        # By identity: comparing track dicts compares their audio arrays.
        index = next((i for i, other in enumerate(self.project.tracks) if other is track), None)
        view = self._view_for(index) if index is not None else self.timeline
        lines = view.bar_lines
        if len(lines) < 2:
            return
        start = float(track.get("start", 0.0) or 0.0)
        end = start + float(track.get("duration", 0.0) or 0.0)
        # The lines over the clip, and one either side so the grid has an edge.
        first = max(0, bisect.bisect_right(lines, start) - 1)
        last = min(len(lines), bisect.bisect_left(lines, end) + 1)
        kept = [round(line + shift, 6) for line in lines[first : last + 1]]
        if len(kept) < 2:
            return
        track["bars"] = kept
        # The same file is the same stem: its other tracks share the grid.
        path = track.get("path")
        for other in self.project.tracks:
            if path and other is not track and other.get("path") == path and not other.get("bars"):
                other["bars"] = list(kept)

    def _use_song_bars(self, index: int) -> None:
        """Forget a stem's own grid (all its pieces): cut it on the song's bars."""
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return
        path = tracks[index].get("path")
        self._remember_edit()
        for track in tracks:
            if track is tracks[index] or (path and track.get("path") == path):
                track.pop("bars", None)
        self._commit_edit()
        self._refresh_timeline()
        self.statusBar().showMessage("This stem follows the song's bars again", 3000)

    def _mark_block_edges(self, track: dict, *file_times: float) -> None:
        """Mark block edges on ``track``, in its audio-file time.

        The track stays the one clip it is: same audio, same name. A time
        inside the clip becomes a block edge (``track["splits"]``). Nothing
        is copied and nothing is written to disk.
        """
        offset = float(track.get("offset", 0.0) or 0.0)
        duration = float(track.get("duration", 0.0) or 0.0)
        if track.get("bar_block"):
            # Drawn as one block until it has marks. The marks cut it.
            track.pop("bar_block", None)
            track["bars"] = [round(offset, 6), round(offset + duration, 6)]
        else:
            self._keep_stem_grid(track)
        marks = {round(float(mark), 6) for mark in track.get("splits") or ()}
        lo = offset + 1e-3
        hi = offset + duration - 1e-3
        for raw in file_times:
            mark = round(float(raw), 6)
            if lo < mark < hi:
                marks.add(mark)
        track["splits"] = sorted(marks)

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
        offset = float(track.get("offset", 0.0) or 0.0)
        self._mark_block_edges(track, offset + local)
        self._commit_edit()
        self._refresh_timeline()
        name = track.get("name", "Track")
        self.statusBar().showMessage(f"“{name}” stays one track. Marked a block edge at {offset + local:.2f} s", 2500)

    def _split_stem_in_half(self, index: int, local_start: float = 0.0, local_duration: float = -1.0) -> None:
        """Cut the block under the scissor in half: a mark on the same track.

        The track stays one track (one clip, one name, the same audio); the
        mark, kept in its audio-file time (``track["splits"]``), is a block
        edge, so the block shows as two, each with its own ▶ and each its own
        voice. Nothing is copied or written to disk.
        """
        tracks = self.project.tracks
        index = int(index)
        if not (0 <= index < len(tracks)):
            return
        track = tracks[index]
        start = float(track.get("start", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        duration = float(track.get("duration", 0.0) or 0.0)
        if float(local_duration) < 0:
            local_start, local_duration = 0.0, duration
        region_start = max(0.0, min(duration, float(local_start)))
        region_end = max(region_start, min(duration, region_start + float(local_duration)))
        if region_end - region_start <= 0.12:
            self.statusBar().showMessage("This block is too short to split in half", 3000)
            return
        mid = (region_start + region_end) / 2.0
        if mid - region_start <= 0.05 or region_end - mid <= 0.05:
            self.statusBar().showMessage("This block is too short to split in half", 3000)
            return
        view = self._view_for(index)
        # The block's identity as the selection may hold it (selection is by
        # identity, so the mark renumbering the blocks after it is harmless).
        bar_here = view.block_at_time(index, start + region_start + 1e-4)
        before_identity = self._block_identity(index, bar_here) if bar_here is not None else None
        self._remember_edit()
        mark = round(offset + mid, 6)
        # In the Track Registry, before the mark renumbers the blocks: the
        # exact block (its segment, made now if it was never needed) is split
        # at the mark's source frame into two segments with ids of their own.
        block = self._segment_of_block(index, start + region_start, start + region_end)
        self._mark_block_edges(track, mark)
        if block is not None:
            timeline = self.session_timeline

            def frame_to_tick(frame, track=track):
                seconds = float(track.get("start", 0.0) or 0.0) + frame / OUTPUT_RATE - float(track.get("offset", 0.0) or 0.0)
                return round(timeline.seconds_to_tick(seconds))

            action = self.action_history.split(
                self.track_registry, block.id, [int(round(mark * OUTPUT_RATE))], frame_to_tick
            )
            # Through the action's result: a selected block that was split is
            # selected as its two halves.
            ids = self.selection_registry.ids
            for old in {before_identity, block.id}:
                if old in ids:
                    at = ids.index(old)
                    ids[at : at + 1] = [child for child in action.created if child not in ids]
            self.selection_registry.replace(ids)
        self._commit_edit()
        self._refresh_timeline()  # draws the selection from the registry; voices follow their blocks
        self._on_track_selected(index)
        name = track.get("name", "Track")
        self.statusBar().showMessage(f"Split a block of “{name}” in half at {offset + mid:.2f} s", 2000)

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
        visible = self.visualizer.isVisible() or (
            self.dancer.isVisible() and self.dancer.enabled
        )
        if visible == self._capture_shown:
            return
        self._capture_shown = visible
        self.engine.set_capture(visible)
        self.visualizer.set_active(self.visualizer.isVisible())

    def _open_sprite_editor(self) -> None:
        editor = getattr(self, "_sprite_editor", None)
        if editor is None:
            from soniclab.ui.sprite_editor import SpriteEditorWindow

            editor = SpriteEditorWindow()
            editor.set_song_clock(self._sprite_song_clock)
            editor.send_requested.connect(self._send_sprite_to_dancer)
            self._sprite_editor = editor
        editor.show()
        editor.raise_()
        editor.activateWindow()

    def _sprite_song_clock(self) -> dict:
        seconds, bpm, downs, _loop, playing = self._dancer_clock()
        pose = self.dancer.pose
        return {
            "seconds": seconds,
            "bpm": bpm,
            "playing": playing,
            "downbeats": list(downs),
            "clock_frame": pose.frame,
            "events": self.dancer.live_events(),
            "levels": {
                "kick": pose.layers.kick,
                "snare": pose.layers.snare,
                "hat": pose.layers.hat,
                "bass": pose.layers.bass,
                "perc": pose.layers.perc,
                "downbeat": 1.0 if pose.on_downbeat else 0.0,
            },
        }

    def _restore_heart_dancer(self) -> None:
        self.dancer.restore_heart()
        self.dancer_dock.show()
        self.dancer_dock.raise_()
        self.statusBar().showMessage("Heart dancer", 3000)

    def _send_sprite_to_dancer(self, data: dict) -> None:
        self.dancer.apply_sprite_project(data)
        self.dancer_dock.show()
        self.dancer_dock.raise_()
        self.statusBar().showMessage("Dancer is using the sprite animation", 4000)

    def _dancer_clock(self):
        """Playback position, tempo, downbeats and loop. Not a wall clock.

        With the song stopped and stems or blocks playing on their own, the
        dancer follows them: the arrangement time a playing block is heard
        at, which lines up with the song's downbeats."""
        bpm = float(self.project.bpm or 120.0)
        playing = bool(self.engine.playing)
        if not playing:
            heard = self._dancer_preview_time()
            if heard is not None:
                return heard, bpm, self._dancer_downbeats(), None, True
        seconds = self._shown_position(playing)  # what plays: the blocks, when blocks play
        return seconds, bpm, self._dancer_downbeats(), self.engine.loop, playing

    def _dancer_preview_time(self) -> float | None:
        """Arrangement seconds heard from the playing block voices, or None.
        Stems of a split share one clock, so any sounding one gives the same."""
        return self.engine.sounding_time()

    def _dancer_audio(self):
        """Latest master block and the arrangement time of its first sample.

        The render head is ahead of ``heard_position`` by the output latency,
        so a hit stamped here is drawn when it is actually heard.
        """
        audio = self.engine.read_master(2048)
        count = int(audio.shape[0]) if getattr(audio, "ndim", 0) else 0
        end = float(self.engine.position)
        if not self.engine.playing:
            heard = self._dancer_preview_time()
            if heard is not None:
                end = heard
        origin = end - count / OUTPUT_RATE
        return audio, origin

    def _dancer_downbeats(self) -> tuple[float, ...]:
        now = time.monotonic()
        cached = getattr(self, "_dancer_marks", None)
        if cached is not None and now - cached[0] < 0.25:
            return cached[1]
        marks = self._dancer_downbeat_times()
        self._dancer_marks = (now, marks)
        return marks

    def _dancer_downbeat_times(self) -> tuple[float, ...]:
        tracks = self.project.tracks
        if not tracks:
            return ()
        drum = self._drum_index()
        index = drum if drum is not None else 0
        if not (0 <= index < len(tracks)):
            return ()
        track = tracks[index]
        events = self._track_events(track)
        if not events:
            return ()
        return arrangement_downbeats(
            events,
            float(track.get("start", 0.0) or 0.0),
            float(track.get("offset", 0.0) or 0.0),
            float(track.get("alignment", 0.0) or 0.0),
        )

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

        self._remember_edit()
        track["_alignment"] = float(updated)  # recorded in the edit (see _RECORDED_RUNTIME)
        self._commit_edit()
        self._apply_onsets()
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

    def _downbeat_grid(self, track: dict) -> list[float] | None:
        """Where a track's blocks are cut: its downbeat markers (the
        triangles: applied downbeats, else the drum track's, else its own
        analysis) in arrangement seconds. Before the first marker the grid
        steps back by the first bar's length, so the clip's opening is a
        block too. None with fewer than two markers: the song's bars then."""
        start = float(track.get("start", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        downs = sorted({round(start + float(time) - offset, 6) for time, position in self._track_events(track) if int(position) == 1})
        grid = None
        if len(downs) >= 2 and downs[1] - downs[0] > 0.05:
            step = downs[1] - downs[0]
            while downs[0] > start + 1e-4:
                downs.insert(0, downs[0] - step)
            grid = downs
        self._cut_grids[id(track)] = tuple(grid or ())
        return grid

    def _toggle_metronome(self) -> None:
        on = not self.engine.metronome
        self.engine.set_metronome(on)
        self.metronome_btn.setChecked(on)
        self.metronome_action.setChecked(on)
        self.metronome_btn.setIcon(metronome_icon("#D7B36A" if on else "#5E6772"))
        if on:
            self._push_metronome_grid()
            self._push_member_meter()
        self.statusBar().showMessage("Metronome on" if on else "Metronome off", 2000)

    def _push_metronome_grid(self) -> None:
        """The beats the metronome clicks while the arrangement plays: the
        drums' bars cut into beats (accent on each downbeat), carried on to
        the end of the song; without drums, the session timeline's beats."""
        numerator, _denominator = parse_time_signature(self.signature.currentText())
        tracks = self.project.tracks
        until = max(
            (float(t.get("start", 0.0) or 0.0) + float(t.get("duration", 0.0) or 0.0) for t in tracks), default=0.0
        ) + 4.0
        beats, accents = [], []
        grid = self._drum_grid()
        if grid:
            lines = list(grid)
            step = lines[-1] - lines[-2]
            while lines[-1] < until and step > 0:
                lines.append(lines[-1] + step)
            for a, b in zip(lines, lines[1:]):
                for k in range(numerator):
                    beats.append(int(round((a + (b - a) * k / numerator) * OUTPUT_RATE)))
                    accents.append(k == 0)
        else:
            for seconds, downbeat in self.session_timeline.get_beat_lines(until):
                beats.append(int(round(float(seconds) * OUTPUT_RATE)))
                accents.append(bool(downbeat))
        order = sorted(range(len(beats)), key=beats.__getitem__)
        self.engine.set_metronome_grid([beats[i] for i in order], [accents[i] for i in order])

    def _push_member_meter(self) -> None:
        """While blocks play the metronome counts the bar they are fitted
        to, one beat of it apart, from where their playback began."""
        numerator, _denominator = parse_time_signature(self.signature.currentText())
        bar = self._member_bar
        self.engine.set_member_meter(bar / numerator if bar else None, numerator)

    def _apply_detected_beats(self) -> None:
        visible = self.timeline.show_detected
        tracks = self.project.tracks
        if self.engine.metronome:
            self._push_metronome_grid()  # the drums' beats may have moved
        if not self._canvases_rebuilding and any(
            id(track) in self._cut_grids
            and self._cut_grids[id(track)] != tuple(self._downbeat_grid_preview(track) or ())
            for track in tracks
        ):
            # New beat results moved the triangles: recut the blocks there,
            # once, after this event.
            if not getattr(self, "_recut_pending", False):
                self._recut_pending = True
                QTimer.singleShot(0, self._recut_blocks)
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

    def _recut_blocks(self) -> None:
        self._recut_pending = False
        self._refresh_timeline()

    def _downbeat_grid_preview(self, track: dict):
        """``_downbeat_grid`` without recording it."""
        saved = self._cut_grids.get(id(track))
        grid = self._downbeat_grid(track)
        if saved is None:
            self._cut_grids.pop(id(track), None)
        else:
            self._cut_grids[id(track)] = saved
        return grid

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
        """The session timeline's stored bars as bar-line seconds, for the
        canvases (compatibility adapter while they still work in seconds).

        Empty means the grid is regular from the tempo and meter. Detected
        downbeats never make bars here: analysis does not change the grid.
        """
        self._sync_session()
        return self.session_timeline.legacy_lines()

    def _sync_session(self) -> None:
        """Keep the timeline and the fields that feed it in step.

        The BPM field is the session tempo while no tempo map is stored; the
        time signature is the default meter (stored bars keep their own).
        ``project.bars`` changed outside the timeline (older code, a test, a
        project from before it) is taken in, as an old project's bars are."""
        timeline = self.session_timeline
        numerator, denominator = parse_time_signature(self.signature.currentText())
        timeline.session.numerator, timeline.session.denominator = numerator, denominator
        if not timeline.tempos and abs(timeline.session.bpm - float(self.bpm.value())) > 1e-9:
            timeline.session.bpm = float(self.bpm.value())
            timeline._changed()
        legacy = [round(float(value), 6) for value in self.project.bars or ()]
        if legacy != timeline.legacy_lines():
            self.session_timeline = SessionTimeline.from_legacy_lines(
                legacy, float(self.bpm.value()), numerator, denominator
            ) if len(legacy) >= 2 else SessionTimeline(timeline.session)
            self.project.bars = self.session_timeline.legacy_lines()

    def _open_session_timeline(self) -> None:
        """A loaded project's timeline: as saved, or migrated from an older
        project's bar lines. An old project whose bars followed the drum
        downbeats has them frozen in (now, or when the downbeats are known),
        so it looks as it did; after that, analysis never moves its bars."""
        numerator, denominator = parse_time_signature(self.signature.currentText())
        bpm = float(self.bpm.value())
        self._freeze_drum_bars = False
        if self.project.timeline:
            self.session_timeline = SessionTimeline.from_dict(self.project.timeline)
        elif len(self.project.bars or ()) >= 2:
            self.session_timeline = SessionTimeline.from_legacy_lines(self.project.bars, bpm, numerator, denominator)
        else:
            self.session_timeline = SessionTimeline()
            self.session_timeline.session.bpm = bpm
            self._freeze_drum_bars = True
            self._follow_drum_bars()
        self.project.bars = self.session_timeline.legacy_lines()

    def _open_registry(self) -> None:
        """A loaded project's Track Registry and action history, as saved.
        An older project has none: its tracks get nodes on the next sync.
        Selection and playback are not restored: no voice starts by itself."""
        self._end_members()
        saved = self.project.registry or {}
        self.track_registry = TrackRegistry.from_dict(saved.get("tracks"))
        self.action_history = ActionHistory.from_dict(saved.get("history"))
        self._sync_history()  # undo after reopening undoes the last saved action
        self.selection_registry = SelectionRegistry()
        self.playback_registry = PlaybackRegistry()

    def _follow_drum_bars(self) -> None:
        """Detected downbeats changed. They never move the session's bars;
        only an old project still waiting for its drum bars takes them in,
        once (see _open_session_timeline)."""
        if not self._freeze_drum_bars:
            return
        lines = self._drum_bar_lines()
        if len(lines) < 2:
            return
        self._freeze_drum_bars = False
        numerator, denominator = parse_time_signature(self.signature.currentText())
        self.session_timeline = SessionTimeline.from_legacy_lines(
            lines, float(self.bpm.value()), numerator, denominator, origin=ORIGIN_MIGRATED_DOWNBEATS
        )
        self.project.bars = self.session_timeline.legacy_lines()
        self._refresh_timeline()

    def _set_bars(self, lines, message: str, origin: str = ORIGIN_MANUAL) -> None:
        """New bar lines (seconds, from the ruler and bar commands) into the
        session timeline, through its tempo map. Empty: the regular grid."""
        self._remember_edit()
        self._sync_session()
        self.session_timeline.set_bar_lines_seconds(lines, origin)
        self.project.bars = self.session_timeline.legacy_lines()
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
        # The user's command: detected downbeats become bars because they ask.
        self._set_bars(lines, f"Marked {len(lines) - 1} bars at the drum downbeats", ORIGIN_DOWNBEATS)

    def _reset_bars(self) -> None:
        self._set_bars([], "Bars follow the tempo and time signature")

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

    # One clock per source song (see sync_group): the stems of a split, a
    # cached re-split of it, every piece cut from them, or the pieces of one
    # unsplit file. The engine keeps it in sample frames of that song; the
    # GUI only reads it. Each block is one voice on its song's clock: it
    # plays its own stretch of the song (a shift on the clock) and loops on
    # its own length, so blocks started together sound together, a block
    # started later joins at the clock's place, and none waits for another.
    # A selection that spans more than one song is the exception: those
    # blocks share one clock for that play (see unify_selection_groups).
    # A song's own clock is left where it is for anything not selected.
    # A mute keeps the voice on the clock, silent, so unmuting lands on the
    # song's current place. Picked bars side by side on one clip play as one
    # block (a run): one voice, keyed by the first bar. Separate clips, the
    # pieces of a cut stem among them, are separate voices.
    # A split's synced loop (stem_loop) repeats its own region instead.

    def _voice_key(self, track: int, bar: int) -> tuple[int, int]:
        tracks = self.project.tracks
        if 0 <= track < len(tracks):
            if stem_loop(tracks[track]) is not None:
                return int(track), STEM_VOICE
            for key, (_data, run) in self._preview_tracks.items():
                if key[0] == track and key[1] <= STEM_BLOCK and run and run[0] <= bar <= run[1]:
                    return key  # a bar of a run playing as one block
            return int(track), stem_block(bar)
        return int(track), int(bar)

    def _block_voice(self, track: int, bar: int, states: dict) -> tuple[int, int]:
        """The voice that plays this block now (``_voice_key``: its own, or
        the run it is part of), so ▶ / ⏸ act on that one and never start a
        second voice for the same block. A ``(track, bar)`` key is a voice
        made through the engine's plain ``preview_play``, which the canvas
        no longer uses; it is still found, so none is left out of reach."""
        own = self._voice_key(track, bar)
        if own not in states and (int(track), int(bar)) in states:
            return int(track), int(bar)
        return own

    def _toggle_block(self, track: int, bar: int) -> None:
        """▶ / ⏸ on a block. On one of several selected blocks it plays the
        whole selection combined, as one loop; on any other block it plays
        just that block."""
        # A stem in a synced loop (Repeat / Loop) plays that loop as its own
        # voice; any other block is a member on the transport clock.
        synced = 0 <= track < len(self.project.tracks) and stem_loop(self.project.tracks[track]) is not None
        if synced:
            if self._toggle_stem_loop(track):
                return
        elif self._toggle_member(track, bar):
            return
        states = self.engine.previews()
        key = self._block_voice(track, bar, states)
        state = states.get(key)
        chosen = id(self.project.tracks[track]) in self._chosen_tracks()
        # The voice this click is about: a later voice under the same key
        # (restarted meanwhile) is not paused or resumed by it.
        generation = self.engine.voice_state(key)[1] or None
        if state is not None and state[0]:
            if chosen:
                self._pause_selected()  # the selected tracks pause together
            else:
                self.engine.preview_pause(key, generation)  # an unselected track: just this one
                self._sync_previews()
            return
        if state is not None:
            # Selected tracks paused together play on together, in sync. An
            # unselected track plays on alone and starts nothing else.
            paused = [key]
            if chosen:
                paused = [
                    other for other, (playing, _seconds) in states.items()
                    if not playing and self._key_chosen(other)
                ]
            try:
                self._yield_transport()
                self.engine.preview_resume(paused, {key: generation} if generation else None)
            except Exception as exc:
                QMessageBox.critical(self, "Audio output error", f"Could not play the block.\n\n{exc}")
            self._sync_previews()
            if not self.frame_timer.isActive():
                self.frame_timer.start()
            return
        self._start_blocks([(track, bar)])

    def _canvas_group(self, view) -> dict:
        """The blocks selected one by one on ``view`` (a box, Shift or Ctrl
        click), ``{track: bars}``, from the Selection Registry: the group a
        drag moves. A track selected whole is not a group of its blocks."""
        picks: dict = {}
        for index, bar in self._picked_blocks():
            if self._view_for(index) is view:
                picks.setdefault(index, set()).add(bar)
        return picks

    def _selected_blocks(self) -> list[tuple[int, int]]:
        """Every selected block, ``(track, bar)``, from the Selection
        Registry: its blocks, and every block of a track selected whole (a
        parent resolves to its blocks)."""
        found = set()
        for identity in self.selection_registry.ids:
            if is_block_id(identity) or self._node_track(identity) is None:
                block = self._identity_block(identity)
                if block is not None:
                    found.add(block)
                continue
            index = self._node_track(identity)
            view = self._view_for(index)
            clip = view._clip_for(index)
            if clip is not None:
                found.update((index, int(span[0])) for span in view._spans_of(clip))
        return sorted(found)

    # -- stable identities -----------------------------------------------------
    # The Selection Registry holds identities, never track indices: a block's
    # identity (block_id: its track's node and its source frames, or the
    # node id of a segment a split made) and a whole track's node id. Indices
    # and bar numbers are worked out from them when needed.

    def _node_track(self, node_id) -> int | None:
        """The index of the track whose node is ``node_id``, or None."""
        for index, track in enumerate(self.project.tracks):
            if track.get("node_id") == node_id:
                return index
        return None

    def _block_identity(self, index: int, bar: int) -> str | None:
        """The stable identity of one block: a split result's own node id,
        else its track's node and its source frames."""
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return None
        times = self._view_for(index).block_times(index, bar)
        node = self.track_registry.get(tracks[index].get("node_id"))
        if times is None or node is None:
            return None
        track = tracks[index]
        clip_start = float(track.get("start", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        first = int(round((offset + float(times[0]) - clip_start) * OUTPUT_RATE))
        last = int(round((offset + float(times[1]) - clip_start) * OUTPUT_RATE))
        for child in self.track_registry.descendants(node.id):
            if child.source_start_frame == first and child.source_end_frame == last:
                return child.id
        return block_id(node.id, first, last)

    def _identity_block(self, identity: str) -> tuple[int, int] | None:
        """``(track, bar)`` of a block identity now, or None if no block
        on any track is that stretch any more."""
        registry = self.track_registry
        if is_block_id(identity):
            owner = block_node(identity)
            first, last = block_frames(identity)
        else:
            node = registry.get(identity)
            if node is None or not node.active:
                return None
            first, last = node.source_start_frame, node.source_end_frame
            owner = node.id
            while owner is not None and self._node_track(owner) is None:
                owner = registry.get(owner).parent_id if registry.get(owner) else None
            if owner is None:
                return None
        family = {owner} | {child.id for child in registry.descendants(owner)}
        for index, track in enumerate(self.project.tracks):
            if track.get("node_id") not in family:
                continue
            clip_start = float(track.get("start", 0.0) or 0.0)
            offset = float(track.get("offset", 0.0) or 0.0)
            at = clip_start + first / OUTPUT_RATE - offset
            view = self._view_for(index)
            bar = view.block_at_time(index, at + 1e-4)
            if bar is None:
                continue
            times = view.block_times(index, bar)
            if times is not None and abs((offset + float(times[1]) - clip_start) * OUTPUT_RATE - last) <= 2:
                return index, bar
        return None

    def _selection_from_views(self) -> None:
        """A selection edit on a canvas, by any means, into the Selection
        Registry (the authority): picked blocks by block identity, clips
        selected whole by their node id."""
        if self._projecting:
            return
        ids = []
        for view in self._timelines():
            picks = view.picks()
            for index, bars in sorted(picks.items()):
                for bar in sorted(bars):
                    identity = self._block_identity(index, bar)
                    if identity is not None:
                        ids.append(identity)
            for index in view.selected_indices():
                if index not in picks and 0 <= index < len(self.project.tracks):
                    node_id = self.project.tracks[index].get("node_id")
                    if node_id:
                        ids.append(node_id)
        self.selection_registry.replace(ids)

    def _project_selection(self) -> None:
        """Draw the Selection Registry on the canvases (picks and Qt's
        selection are its picture), dropping identities nothing resolves
        to any more, so no stale id stays selected."""
        picks_by_view = {view: {} for view in self._timelines()}
        whole = {}
        kept = []
        for identity in self.selection_registry.ids:
            index = None if is_block_id(identity) else self._node_track(identity)
            if index is not None:
                whole.setdefault(self._view_for(index), set()).add(index)
                kept.append(identity)
                continue
            block = self._identity_block(identity)
            if block is None:
                continue
            index, bar = block
            picks_by_view[self._view_for(index)].setdefault(index, set()).add(bar)
            kept.append(identity)
        self.selection_registry.replace(kept)
        self._projecting = True
        try:
            for view, picks in picks_by_view.items():
                chosen = whole.get(view, set())
                view.scene_obj.blockSignals(True)
                try:
                    for clip in view._clip_items():
                        if clip.isSelected() != (clip.clip_index in chosen):
                            clip.setSelected(clip.clip_index in chosen)
                finally:
                    view.scene_obj.blockSignals(False)
                view.set_picks(picks)
                view._update_combined_play()
        finally:
            self._projecting = False

    # -- block playback on the one transport clock -----------------------------
    # A block's ▶ starts that block, and Play (toolbar or Space) with blocks
    # picked starts each of them. Every one is a member of its own in the
    # Playback Registry, played by the transport from its own first frame,
    # all at once, on the transport's one clock (see members_begin): none
    # waits for its place on the arrangement or for another block, and each
    # stops at its own end. A block started later joins at the clock's place
    # now; the others go on where they are. A block's ▶ / ⏸ pauses or
    # resumes that block only; Pause, Play and Stop act on all of them.
    # A change of selection stops every playing block it does not hold (see
    # _stop_deselected), one started with its own ▶ unselected too.
    # A synced stem loop still plays as its own voice (see _start_blocks).

    def _members_active(self) -> bool:
        """Whether the transport plays (or holds) blocks, not the song."""
        return self.engine.members is not None

    def _picked_blocks(self) -> list[tuple[int, int]]:
        """Blocks picked themselves, ``(track, bar)``: not every block of a
        track selected whole (a click on a clip, to edit it)."""
        found = set()
        for identity in self.selection_registry.ids:
            if is_block_id(identity) or self._node_track(identity) is None:
                block = self._identity_block(identity)
                if block is not None:
                    found.add(block)
        return sorted(found)

    def _play_members(self) -> bool:
        """Play / Space: every member plays on, or every picked block
        starts. False when neither applies (the song plays)."""
        if self._members_active():
            scope = self._scope_keys()
            for key in list(self.playback_registry.playing_blocks) if scope is None else scope:
                self._hold_member(key, False)
            self._run_clock()
            return True
        if self.engine.playing:
            return False  # the song plays: it goes on
        if len(self.selection_registry.ids) > 1:
            self._start_member_entries(self._group_units())
            return True
        blocks = self._picked_blocks()
        if not blocks:
            return False  # nothing picked: the song plays
        self._start_members(blocks)
        return True

    def _toggle_member(self, track: int, bar: int) -> bool:
        """A block's own ▶ / ⏸: pause or resume this block only, or start it
        as one more member. False while the song plays (it is in the song
        already) or when it has no identity."""
        identity = self._block_identity(track, bar)
        if identity is None:
            return False
        identity = self._member_key(track, bar, identity) or identity
        return self._toggle_unit(identity, lambda: self._start_members([(track, bar)]))

    def _toggle_stem_loop(self, track: int) -> bool:
        """▶ / ⏸ on a split stem in a synced loop (Repeat / Loop): its loop
        is a member like any block, on the same clock and grid, in time with
        everything else that plays, and it always loops. Keyed by the
        stem's node, so a selection holding the stem plays the same member."""
        identity = self.project.tracks[track].get("node_id") if 0 <= track < len(self.project.tracks) else None
        place = self._unit_place(identity) if identity else None
        if place is None:
            return False
        return self._toggle_unit(identity, lambda: self._start_member_entries({identity: place}))

    def _toggle_unit(self, identity: str, start) -> bool:
        """Pause or resume the member ``identity``, or ``start`` it as one
        more (with the clock stopped, it plays alone: the others hold)."""
        member = self.playback_registry.playing_blocks.get(identity)
        if member is not None and identity in (self.engine.members or {}):
            if member.state == PLAYBACK_PLAYING and self.engine.playing:
                self._hold_member(identity, True)
            else:
                if not self.engine.playing:
                    self._hold_others(identity)  # the clock restarts for this block alone
                self._hold_member(identity, False)
                self._run_clock()
            self._sync_previews()
            return True
        if self.engine.playing and not self._members_active():
            self.statusBar().showMessage("That block already plays with the song", 3000)
            return True
        if not self.engine.playing:
            self._hold_others(identity)
        start()
        return True

    def _loops_always(self, identity) -> bool:
        """A split stem's synced loop loops whatever the loop button says."""
        if is_block_id(identity):
            return False
        index = self._node_track(identity)
        return index is not None and stem_loop(self.project.tracks[index]) is not None

    def _hold_others(self, identity) -> None:
        keep = {identity} if isinstance(identity, str) else set(identity)
        for key, block in self.playback_registry.playing_blocks.items():
            if key not in keep and block.state == PLAYBACK_PLAYING:
                self._hold_member(key, True)

    # The selection as one: two or more selected things (blocks picked by a
    # box, Shift or Ctrl, or tracks selected whole, on either canvas) share
    # one ▶ that starts, pauses and resumes them together, on the one clock.
    # A track selected whole plays as one member over its whole clip, its
    # bars one after another, as the single track it is. Each keeps its own
    # audio, identity and place; nothing is merged.

    def _group_units(self) -> dict:
        """``{identity: (track, first frame, end frame)}`` of everything
        selected, from the Selection Registry."""
        units = {}
        for identity in self.selection_registry.ids:
            place = self._unit_place(identity)
            if place is not None:
                units[identity] = place
        return units

    def _unit_place(self, identity: str):
        """``(track, first frame, end frame)`` of a selected or playing
        thing now: a whole track by its node id, else a block. None when it
        is gone."""
        if not is_block_id(identity):
            index = self._node_track(identity)
            if index is not None:
                track = self.project.tracks[index]
                start = float(track.get("start", 0.0) or 0.0)
                duration = float(track.get("duration", 0.0) or 0.0)
                first, end = start, start + duration
                loop = stem_loop(track)
                if loop is not None:
                    # A split stem in a synced loop plays that loop: its song
                    # seconds on this clip, inside the clip.
                    offset = float(track.get("offset", 0.0) or 0.0)
                    first = max(first, start + loop[0] - offset)
                    end = min(end, start + loop[1] - offset)
                    if end <= first:
                        return None
                return index, int(round(first * OUTPUT_RATE)), int(round(end * OUTPUT_RATE))
        block = self._identity_block(identity)
        if block is None:
            return self._place_by_source(identity)  # a run of several blocks
        entries = self._block_entries([block])
        return next(iter(entries.values()), None)

    def _group_members(self) -> dict:
        """The selection as the members that play it: each track's selected
        stretches in order, keyed as the members already playing them."""
        units = self._group_units()
        return self._as_member_keys(self._chain_by_track(units)) if units else {}

    def _as_member_keys(self, entries: dict) -> dict:
        """The same entries, keyed by the member already playing each place
        where there is one (an edit can give a block a new identity while
        its member keeps the one it started with): never a second member."""
        by_place = {(entry[0], entry[1], entry[2]): key for key, entry in (self.engine.members or {}).items()}
        return {by_place.get(tuple(entry[:3]), identity): entry for identity, entry in entries.items()}

    def _group_playing(self) -> bool:
        """Whether everything selected plays now (the shared ▶ shows ⏸)."""
        if not self.engine.playing or not self._members_active():
            return False
        members = self.playback_registry.playing_blocks
        keys = self._group_members()
        return bool(keys) and all(
            members.get(key) is not None and members[key].state == PLAYBACK_PLAYING for key in keys
        )

    def _toggle_group(self) -> None:
        """The shared ▶ / ⏸: pause every selected thing, or play them all
        together (those not playing yet from their own start, the paused
        ones on from where they are). Other playing blocks are left alone,
        except that a stopped clock restarts for the selection only."""
        units = self._group_members()
        if not units:
            return
        if self._group_playing():
            for identity in units:
                self._hold_member(identity, True)
            self._sync_previews()
            return
        if not self.engine.playing and self._members_active():
            self._hold_others(units)
        if self.engine.playing and not self._members_active():
            self.statusBar().showMessage("Those tracks already play with the song", 3000)
            return
        self._start_member_entries(units)
        count = len(units)
        self.statusBar().showMessage(f"Playing {count} selected together", 3000)

    def _hold_member(self, key: str, held: bool) -> None:
        self.engine.member_hold(key, held)
        self.playback_registry.set_state(key, PLAYBACK_PAUSED if held else PLAYBACK_PLAYING)

    def _run_clock(self) -> None:
        if not self.engine.playing:
            self._play(keep_focus=True)
        self.frame_timer.start()
        self._sync_transport_lamp()

    def _start_members(self, blocks) -> None:
        """Each block becomes a member, from its own first frame now (see
        _start_member_entries)."""
        entries = {}
        for identity, entry in self._block_entries(blocks).items():
            block = self._identity_block(identity)
            known_key = self._member_key(*block, identity) if block is not None else None
            entries[known_key or identity] = entry  # one member per block, never a second
        self._start_member_entries(entries)

    def _start_member_entries(self, entries: dict, play: bool = True) -> None:
        """``{identity: (track, first frame, end frame)}`` become members, in
        one step for the audio thread: each track's stretches play in order,
        as that one track (see _chain_by_track), every track from now.
        Members already playing are not touched (a paused one plays on from
        its place); a block auditioning carries on from where it is."""
        if not entries:
            return
        if all(len(entry) == 3 for entry in entries.values()):
            entries = self._chain_by_track(entries)
        entries = self._as_member_keys(entries)
        self._hear_every_member()
        adopted, into, voices = self._adopt_auditions()
        entries = {**adopted, **entries}
        loop = self._block_loop
        known = self.engine.members or {}
        replaced = self._overlapped_members(entries, known)
        try:
            with self.engine.batch():
                self.engine.members_begin()
                for key in voices:
                    self.engine.preview_stop(key)
                    self._preview_tracks.pop(key, None)
                for key in replaced:
                    self.engine.member_remove(key)  # its track plays again as the new sequence
                for identity, (track, first, end, *rest) in entries.items():
                    wait, period, rate = (list(rest) + [0, None, 1.0][len(rest):])[:3]
                    if identity in known:
                        self.engine.member_hold(identity, False)
                    else:
                        loop = self._block_loop or self._loops_always(identity)
                        # In time with what already plays (an audition taken
                        # over carries on exactly where it is instead).
                        self.engine.member_add(
                            identity, track, first, end, loop, into.get(identity, -wait), period,
                            sync=identity not in into, rate=rate,
                        )
                if play:
                    self.engine.play()
        except Exception as exc:
            QMessageBox.critical(self, "Audio output error", f"Could not start audio output.\n\n{exc}")
        for key in replaced:
            self.playback_registry.remove(key)
        members = self.engine.members or {}
        for identity, (track, first, end, *_rest) in entries.items():
            entry = members.get(identity)
            if entry is None:
                continue
            existing = self.playback_registry.playing_blocks.get(identity)
            if existing is not None:
                existing.state = PLAYBACK_PLAYING
                continue
            data = self.project.tracks[track]
            shift = self._file_shift(data)
            node = self.track_registry.get(data.get("node_id"))
            self.playback_registry.add(PlayingBlockState(
                id=identity,
                source_id=node.source_asset_id if node is not None else None,
                source_start=first + shift,
                source_end=end + shift,
                source_cursor=first + shift,
                global_start=entry[3],
                loop=entry[4],
                rate=entry[7],
            ))
        self._update_member_blocks()
        if not self._member_bar and self._members_active():
            # The first blocks of this playback set the bar for everything
            # after: joins, resumes and the metronome all keep to it.
            self._member_bar = getattr(self, "_last_chain_bar", None)
            self._push_member_meter()
        if play:
            self._run_clock()
        self._sync_previews()

    # Most a row is sped up or slowed down to fit the lead's bars (8 %);
    # past that it keeps its speed and rests until its next bar instead.
    MAX_FIT = 0.08

    def _chain_by_track(self, entries: dict) -> dict:
        """The selection as players that never wait for each other: every
        row starts at once and loops on its own, its selected stretches one
        after another (gaps left out; stretches of one clip side by side are
        one run). The drums lead: each row is fitted to a whole number of
        the drums' bars, a little faster when it runs long, a little slower
        when it runs short, so its downbeats land on the drums' downbeats on
        every pass. ``{key: (track, first, end, frames to wait, loop period
        or None, rate)}``; a run of several blocks of a clip is keyed by the
        logical block that spans it."""
        tracks = self.project.tracks
        rows_of: dict = {}
        by_row: dict[tuple, list] = {}
        for identity, (track, first, end) in entries.items():
            view = self._view_for(track)
            if view not in rows_of:
                rows_of[view] = view.lane_rows(tracks)[0]
            rows = rows_of[view]
            row = rows[track] if 0 <= track < len(rows) and rows[track] >= 0 else ("track", track)
            by_row.setdefault((view.canvas, row), []).append((first, end, track, identity))
        groups: list[list] = []
        for items in by_row.values():
            items.sort()
            runs: list[list] = []
            for first, end, track, identity in items:
                last = runs[-1] if runs else None
                if last is not None and last[3] == track and first <= last[1] + 2:
                    last[1] = max(last[1], end)
                    last[2].append(identity)
                else:
                    runs.append([first, end, [identity], track])
            groups.append(runs)
        # Every block is fitted to one bar, a lone one too: a block played on
        # its own ▶ loops on whole drum bars, so one that joins later, or that
        # others join, stays on the same grid instead of drifting by the few
        # samples its own length is off a bar on every pass.
        bar = self._lead_bar_frames(groups)
        self._last_chain_bar = bar
        chained = {}
        for runs in groups:
            natural = sum(end - first for first, end, _ids, _track in runs)
            rate, cycle = 1.0, natural
            # The drums set the bar (their length over their whole bars), so
            # fitting leaves them at exactly their own speed; they only
            # change speed when they join a bar locked before them.
            if bar:
                bars = max(1, int(round(natural / bar)))
                fitted = natural / (bars * bar)
                if abs(fitted - 1.0) <= self.MAX_FIT:
                    rate, cycle = fitted, int(round(bars * bar))
                else:
                    cycle = int(math.ceil(natural / bar) * bar)  # its own speed, resting to the next bar
            period = None if (len(runs) == 1 and cycle == natural and rate == 1.0) else cycle
            wait = 0.0
            for first, end, ids, track in runs:
                data = tracks[track]
                node = data.get("node_id")
                shift = self._file_shift(data)
                key = ids[0] if len(ids) == 1 or not node else block_id(node, first + shift, end + shift)
                chained[key] = (track, first, end, int(round(wait)), period, rate)
                wait += (end - first) / rate
        return chained

    def _lead_bar_frames(self, groups) -> float | None:
        """The bar every block is fitted to: the one locked when this
        playback began, so all of it keeps one grid; else worked out now
        (see _measure_lead_bar)."""
        if self._members_active() and self._member_bar:
            return self._member_bar
        return self._measure_lead_bar(groups)

    def _measure_lead_bar(self, groups) -> float | None:
        """One bar of the lead, in frames: the drums' own selected row when
        it is in the selection (its length over its whole bars, so the drums
        play as recorded), else the drums' downbeat spacing, else the
        song's bar lines, else the tempo and meter."""
        grid = None
        drum = self._drum_index()
        if drum is not None:
            lines = self._downbeat_grid(self.project.tracks[drum]) or []
            steps = [b - a for a, b in zip(lines, lines[1:]) if b > a]
            if steps:
                grid = float(np.median(steps)) * OUTPUT_RATE
        if grid is None:
            lines = self._bar_lines()
            steps = [b - a for a, b in zip(lines, lines[1:]) if b > a]
            if steps:
                grid = float(np.median(steps)) * OUTPUT_RATE
        if grid is None:
            numerator, _denominator = parse_time_signature(self.signature.currentText())
            grid = 60.0 / max(1.0, float(self.bpm.value())) * numerator * OUTPUT_RATE
        if drum is not None:
            for runs in groups:
                if any(track == drum for *_rest, track in runs):
                    natural = sum(end - first for first, end, _ids, _track in runs)
                    return natural / max(1, int(round(natural / grid)))
        return grid

    def _overlapped_members(self, entries: dict, known: dict) -> list:
        """Members a new run takes over: on the same track, over the same
        audio, under another key (a run grew by a block, say). Without this
        that audio would play twice."""
        gone = []
        for identity, (track, first, end, *_rest) in entries.items():
            if identity in known:
                continue
            for key, entry in known.items():
                if key not in entries and key not in gone and entry[0] == track and entry[1] < end and first < entry[2]:
                    gone.append(key)
        return gone

    @staticmethod
    def _file_shift(track) -> int:
        """Arrangement frame to the track's file frame: add this."""
        return int(round((float(track.get("offset", 0.0) or 0.0) - float(track.get("start", 0.0) or 0.0)) * OUTPUT_RATE))

    def _block_entries(self, blocks) -> dict:
        """``{identity: (clip index, first frame, end frame)}`` for
        ``blocks``: each block's stretch in arrangement frames on its clip
        (so its file frames, gain, pan and fades are the clip's own)."""
        entries = {}
        for track, bar in blocks:
            times = self._view_for(track).block_times(track, bar)
            identity = self._block_identity(track, bar)
            if times is None or identity is None:
                continue
            entries[identity] = (track, int(round(float(times[0]) * OUTPUT_RATE)), int(round(float(times[1]) * OUTPUT_RATE)))
        return entries

    def _adopt_auditions(self):
        """Blocks auditioning now, as members that carry on from where they
        are: ``(entries, frames already in, voice keys)``, measured at what
        is mixed next, not what is heard. A synced stem loop and a paused
        voice are not taken over."""
        entries, into, voices = {}, {}, []
        if not self._preview_tracks:
            return entries, into, voices  # nothing auditioning: nothing to take over
        states = self.engine.previews(time.monotonic() + self.engine.output_delay)
        for key, (data, run) in list(self._preview_tracks.items()):
            state = states.get(key)
            if state is None or not state[0] or not run or key[1] > STEM_BLOCK:
                continue
            track = key[0]
            if not (0 <= track < len(self.project.tracks)) or self.project.tracks[track] is not data:
                continue
            view = self._view_for(track)
            for bar in range(run[0], run[1] + 1):
                times = view.block_times(track, bar)
                identity = self._block_identity(track, bar)
                if times is None or identity is None:
                    continue
                entries[identity] = (track, int(round(times[0] * OUTPUT_RATE)), int(round(times[1] * OUTPUT_RATE)))
                # A later bar of a run comes in when the run reaches it.
                into[identity] = int(round((float(state[1]) - float(times[0])) * OUTPUT_RATE))
            voices.append(key)
        return entries, into, voices

    def _hear_every_member(self) -> None:
        """Members can be on either canvas and any track: a canvas or track
        focus (which silences the rest) does not apply to them."""
        self._clear_track_focus()
        if self._canvas_focus is not None:
            self._set_canvas_focus(None)

    def _remove_member(self, identity: str) -> None:
        """One block stops; the others play on, untouched."""
        self.engine.member_remove(identity)
        self.playback_registry.remove(identity)
        if not self.engine.members:
            self._end_members(pause=True)
        self._sync_previews()

    def _retire_completed_members(self) -> None:
        """Blocks that played to their own end leave; the rest go on."""
        done = self.engine.members_completed
        if not done:
            return
        for key in done:
            self.engine.member_remove(key)
            self.playback_registry.remove(key)
        if not self.engine.members:
            self._end_members(pause=True)

    def _end_members(self, pause: bool = False) -> None:
        """Member playback ends (Stop, or nothing left): the transport is
        back where it began, playing the song. The selection stays."""
        if not self._members_active() and not self.playback_registry.playing_blocks:
            return
        if pause and self.engine.playing:
            self._pause()
        origin = self.engine.members_end()
        if origin is not None:
            self.engine.seek(origin / OUTPUT_RATE)
        self.playback_registry.stop()
        self._member_blocks = {}
        self._member_bar = None
        self._set_loop_button(self.engine.loop is not None)  # the arrangement's loop again
        self._sync_previews()
        self._sync_transport_lamp()

    def _sync_member_loop(self) -> None:
        """Every playing block wraps within its own stretch, or plays once,
        as ``_block_loop`` says."""
        if not self._members_active():
            return
        loop = self._block_loop
        keys = [key for key in (self.engine.members or {}) if not self._loops_always(key)]
        self.engine.member_set_loop(loop, keys)
        for key in keys:
            block = self.playback_registry.playing_blocks.get(key)
            if block is not None:
                block.loop = loop

    def _update_member_blocks(self) -> None:
        """Where each member is now, ``identity -> (track, bar)`` (rows,
        indices and bars may have changed)."""
        blocks = {}
        for identity in self.playback_registry.playing_blocks:
            whole = None if is_block_id(identity) else self._node_track(identity)
            if whole is not None:
                blocks[identity] = (whole, None)  # a whole track: the bar its cursor is in, each frame
                continue
            block = self._identity_block(identity)
            if block is None:
                place = self._place_by_source(identity)  # a run: the bar its cursor is in, each frame
                block = None if place is None else (place[0], None)
            if block is not None:
                blocks[identity] = block
        self._member_blocks = blocks

    def _place_members_on_tracks(self) -> None:
        """Each playing block onto the track that holds its source frames
        now (tracks were reordered, cut or moved), from the tracks alone,
        before any canvas redraws. Its place in its own stretch is kept."""
        for identity in list(self.engine.members or {}):
            whole = not is_block_id(identity) and self._node_track(identity) is not None
            place = self._unit_place(identity) if whole else self._place_by_source(identity)
            if place is not None:
                self.engine.member_place(identity, *place)

    def _place_by_source(self, identity: str):
        """``(track, first, end)`` arrangement frames of the stretch an
        identity names, found by its source frames on the tracks of its
        node's family: a block, a run of blocks, or a segment. None when no
        track holds it now."""
        registry = self.track_registry
        if is_block_id(identity):
            owner, (first, last) = block_node(identity), block_frames(identity)
        else:
            node = registry.get(identity)
            if node is None:
                return None
            owner, first, last = node.id, node.source_start_frame, node.source_end_frame
            while owner is not None and self._node_track(owner) is None:
                parent = registry.get(owner)
                owner = parent.parent_id if parent is not None else None
            if owner is None:
                return None
        family = {owner} | {child.id for child in registry.descendants(owner)}
        for index, track in enumerate(self.project.tracks):
            if track.get("node_id") not in family:
                continue
            shift = self._file_shift(track)
            offset = int(round(float(track.get("offset", 0.0) or 0.0) * OUTPUT_RATE))
            length = int(round(float(track.get("duration", 0.0) or 0.0) * OUTPUT_RATE))
            if offset - 2 <= first and last <= offset + length + 2:
                return index, first - shift, last - shift
        return None

    def _member_key(self, track: int, bar: int, identity: str | None) -> str | None:
        """The member playing this block, if any: by its identity, or the
        member shown on it (a move can give a block a new identity while
        its member keeps the one it started with)."""
        if identity in self.playback_registry.playing_blocks:
            return identity
        for key, block in self._member_blocks.items():
            if block == (track, bar) and key in self.playback_registry.playing_blocks:
                return key
        times = self._view_for(track).block_times(track, bar) if 0 <= track < len(self.project.tracks) else None
        if times is not None:
            # A block inside a run (or a whole track) playing as one member.
            middle = int(round((float(times[0]) + float(times[1])) / 2 * OUTPUT_RATE))
            for key, entry in (self.engine.members or {}).items():
                if entry[0] == track and entry[1] <= middle < entry[2] and key in self.playback_registry.playing_blocks:
                    return key
        return None

    def _refresh_members(self) -> None:
        """Tracks were reloaded: each member follows its block (by identity)
        and keeps its place; a block that is gone stops."""
        if not self._members_active():
            return
        for identity in list(self.playback_registry.playing_blocks):
            entry = self._unit_place(identity)
            if entry is None:
                self.engine.member_remove(identity)
                self.playback_registry.remove(identity)
                continue
            self.engine.member_place(identity, *entry)
        if not self.engine.members:
            self._end_members(pause=True)
            return
        self._update_member_blocks()

    def _stop_deselected(self) -> None:
        """The selection changed: every playing block it does not hold stops
        and leaves the Playback Registry, like any unselected block (one
        started with its own ▶ without being selected too). A playing run
        that lost only some of its blocks plays on as the ones still
        selected, in time. Blocks still selected play on."""
        before, self._selection_before = self._selection_before, None
        members = self.engine.members
        if before is None or not members:
            return
        ids = list(self.selection_registry.ids)
        if set(ids) == before:
            return
        kept = {identity: place for identity in ids if (place := self._unit_place(identity)) is not None}

        def overlap(place, clip, first, end) -> int:
            return max(0, min(end, place[2]) - max(first, place[1])) if place[0] == clip else 0

        stop, rejoin = [], {}
        for key, entry in members.items():
            clip, first, end = entry[0], entry[1], entry[2]
            covered = sum(overlap(place, clip, first, end) for place in kept.values())
            if key in kept or covered >= (end - first) - 2:
                continue  # still selected all over (it moved, or was picked again)
            stop.append(key)
            rejoin.update({identity: place for identity, place in kept.items() if overlap(place, clip, first, end)})
        if rejoin:
            # The run's selected rest replaces it, on the grid (the old run
            # is taken out as it starts, so its audio never doubles).
            self._start_member_entries(rejoin, play=self.engine.playing)
        for key in stop:
            if key in (self.engine.members or {}):
                self._remove_member(key)
        self._sync_previews()

    def _member_marks(self) -> dict:
        """``(track, bar) -> (playing, arrangement seconds)`` for each member
        with a place to show: its own start sample on the one clock, never
        its place on the arrangement, decides how far in it is."""
        members = self.engine.members
        if not members:
            return {}
        playing = self.engine.playing
        now = self.engine.heard_position() if playing else self.engine.position
        position = int(round(now * OUTPUT_RATE))
        marks = {}
        for key, entry in members.items():
            block = self._member_blocks.get(key)
            offset = AudioEngine.member_offset(entry, position)
            if block is None or offset is None:
                continue
            seconds = (entry[1] + offset) / OUTPUT_RATE
            if block[1] is None:
                bar = self._view_for(block[0]).block_at_time(block[0], seconds)
                if bar is None:
                    continue
                block = (block[0], bar)
            marks[block] = (playing and entry[5] is None, seconds)
            record = self.playback_registry.playing_blocks.get(key)
            if record is not None:
                record.source_cursor = record.source_start + offset
        return marks

    def _follow_selection(self) -> None:
        """A selection edit on a canvas, by any means: into the Selection
        Registry. What left the selection stops playing once the edit is
        done (an edit that clears the selection and picks again counts as
        one), see _stop_deselected."""
        if self._projecting:
            return
        if self._selection_before is None:
            self._selection_before = set(self.selection_registry.ids)
            QTimer.singleShot(0, self._stop_deselected)
        self._selection_from_views()
        for view in self._timelines():
            view._update_combined_play()

    # -- the Track Registry --------------------------------------------------

    def _sync_registry(self) -> None:
        """Keep the Track Registry in step with the tracks (the adapter while
        the window still edits track dicts).

        Every track gets a node with a stable id ("node_id" on the dict): a
        song or file is an original, a split's stems are stems under one
        original for the split, and a track copied from another (a block
        dragged out, an old split piece) is a segment under the node it came
        from, so no id is ever active twice. A node keeps its id through
        moves, saves and undo; one whose track is gone is deactivated, not
        deleted. Detected downbeats go to the source asset in sample frames.
        """
        registry = self.track_registry
        timeline = self.session_timeline
        tracks = self.project.tracks
        live = set()
        first_of_stem: dict[tuple, str] = {}
        for track in tracks:
            path = str(track.get("path") or "")
            audio = track.get("_audio")
            total = int(len(audio)) if audio is not None else 0
            asset = registry.assets.get(track.get("asset_id") or "")
            if asset is None:
                asset = registry.asset_for(path, OUTPUT_RATE, total) if path else registry.asset_for(f"memory:{id(track)}", OUTPUT_RATE, total)
                track["asset_id"] = asset.id
            events = track.get("downbeats") or ()
            if events:
                asset.beats = [int(round(float(time) * OUTPUT_RATE)) for time, _position in events]
                asset.downbeats = [int(round(float(time) * OUTPUT_RATE)) for time, position in events if int(position) == 1]
            start = float(track.get("start", 0.0) or 0.0)
            offset = float(track.get("offset", 0.0) or 0.0)
            duration = float(track.get("duration", 0.0) or 0.0)
            source = (int(round(offset * OUTPUT_RATE)), int(round((offset + duration) * OUTPUT_RATE)))
            place = (int(round(timeline.seconds_to_tick(start))), int(round(timeline.seconds_to_tick(start + duration))))
            node = registry.get(track.get("node_id"))
            group = stem_group(track)
            stem_key = (group, path) if group and path else None  # an old piece shares its stem's file
            if node is None or node.id in live:
                parent = None
                kind = TRACK_ORIGINAL
                if node is not None:  # a copy of a track already seen: a stretch of it
                    parent, kind = node.id, TRACK_SEGMENT
                elif stem_key in first_of_stem:  # an old split piece of a stem
                    parent, kind = first_of_stem[stem_key], TRACK_SEGMENT
                elif group:
                    song = f"song-{group}"
                    if registry.get(song) is None:
                        registry.create(TRACK_ORIGINAL, None, 0, 0, 0, 0, node_id=song)
                    parent, kind = song, TRACK_STEM
                node = registry.create(
                    kind, asset.id, source[0], source[1], place[0], place[1],
                    parent_id=parent, stem_type=track.get("stem_type"),
                )
                track["node_id"] = node.id
            node.active = True
            node.source_asset_id = asset.id
            node.source_start_frame, node.source_end_frame = source
            node.arrangement_start_tick, node.arrangement_end_tick = place
            live.add(node.id)
            if stem_key and stem_key not in first_of_stem:
                first_of_stem[stem_key] = node.id
        managed = self.action_history.managed()
        for node in registry.nodes.values():
            if node.id in live or node.id in managed:
                continue  # managed: a split's original and halves follow the history
            if node.track_type == TRACK_ORIGINAL and any(child.id in live for child in registry.children(node.id, False)):
                node.active = True  # a split's original stays while its stems do
            elif node.created_by_action_id == "selection":
                node.active = node.parent_id in live  # a block's segment lives with its track
            else:
                node.active = False
        self.selection_registry.reconcile(registry)
        self.playback_registry.reconcile(registry, self.selection_registry)

    def _segment_of_block(self, track_index: int, start: float, end: float):
        """The registry segment of one block of a track (arrangement seconds
        ``start`` to ``end``), made when first needed."""
        tracks = self.project.tracks
        if not (0 <= track_index < len(tracks)):
            return None
        track = tracks[track_index]
        node = self.track_registry.get(track.get("node_id"))
        if node is None:
            self._sync_registry()
            node = self.track_registry.get(track.get("node_id"))
            if node is None:
                return None
        clip_start = float(track.get("start", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        first = int(round((offset + start - clip_start) * OUTPUT_RATE))
        last = int(round((offset + end - clip_start) * OUTPUT_RATE))
        timeline = self.session_timeline
        segment = self.track_registry.segment_for(
            node.id, first, last, int(round(timeline.seconds_to_tick(start))), int(round(timeline.seconds_to_tick(end)))
        )
        if segment.created_by_action_id is None and segment.id != node.id:
            segment.created_by_action_id = "selection"
        return segment

    def _chosen_tracks(self) -> set[int]:
        """The tracks with anything selected, from the Selection Registry
        (as track objects' ids, for code that matches tracks by object)."""
        tracks = self.project.tracks
        return {id(tracks[index]) for index, _bar in self._selected_blocks() if 0 <= index < len(tracks)}

    def _key_chosen(self, key) -> bool:
        tracks = self.project.tracks
        return 0 <= key[0] < len(tracks) and id(tracks[key[0]]) in self._chosen_tracks()

    def _pause_selected(self) -> None:
        """Pausing one selected track pauses every selected track (and the
        song if it plays). Unselected tracks keep playing."""
        chosen = self._chosen_tracks()
        tracks = self.project.tracks
        for key, (playing, _seconds) in self.engine.previews().items():
            if playing and 0 <= key[0] < len(tracks) and id(tracks[key[0]]) in chosen:
                self.engine.preview_pause(key)
        if self.engine.playing:
            self._pause()
        self._sync_previews()
        self.statusBar().showMessage("Paused the selected tracks. ▶ plays them on", 2500)

    def _canvas_input(self, starting: bool) -> None:
        """Around each mouse press or release on a canvas: a track the user
        just unselected stops playing at once."""
        if starting:
            if self._input_depth == 0:
                self._chosen_at_input = self._chosen_tracks()
            self._input_depth += 1
            return
        self._input_depth = max(0, self._input_depth - 1)
        if self._input_depth == 0:
            self._stop_unselected(self._chosen_at_input - self._chosen_tracks())

    def _stop_unselected(self, gone: set[int]) -> None:
        if not gone:
            return
        tracks = self.project.tracks
        stopped = False
        for key in list(self.engine.previews()):
            if 0 <= key[0] < len(tracks) and id(tracks[key[0]]) in gone:
                self.engine.preview_stop(key)
                stopped = True
        if stopped:
            self._sync_previews()

    def _pause_everything(self) -> None:
        """Pausing one track pauses every track: each playing block and the
        song. Each keeps its place, so ▶ plays them on in sync."""
        for key, (playing, _seconds) in self.engine.previews().items():
            if playing:
                self.engine.preview_pause(key)
        if self.engine.playing:
            self._pause()
        self._sync_previews()
        self.statusBar().showMessage("Paused everything. ▶ plays it on", 2500)

    def _is_split_piece(self, index: int) -> bool:
        """True when this clip is one piece of a stem that was cut in two.

        An uncut stem is the whole file and still plays one bar at a time.
        A piece covers only part of the file and has a sibling from the
        same split, so its ▶ plays the piece straight through.
        """
        tracks = self.project.tracks
        if not (0 <= index < len(tracks)):
            return False
        track = tracks[index]
        group = stem_group(track)
        path = track.get("path")
        if not group or not path or stem_loop(track) is not None:
            return False
        audio = track.get("_audio")
        if audio is None or len(audio) == 0:
            return False
        file_length = len(audio) / OUTPUT_RATE
        offset = float(track.get("offset", 0.0) or 0.0)
        duration = float(track.get("duration", 0.0) or 0.0)
        if offset <= 0.05 and duration >= file_length - 0.05:
            return False
        # A bar grabbed and moved is one block. The stem it was taken from
        # still plays one bar at a time; only a cut at the playhead plays
        # the whole piece.
        if track.get("bar_block"):
            return False
        siblings = [
            other for other in tracks
            if other is not track and stem_group(other) == group and other.get("path") == path
        ]
        if not siblings or any(other.get("bar_block") for other in siblings):
            return False
        return True

    def _whole_pieces(self, keys) -> list[tuple[int, int]]:
        """A cut stem plays every bar of the piece under the pointer.

        One bar of that piece is half of it when the cut left two bars, and
        looping that bar restarts in the middle. The piece is one run.
        """
        found = []
        seen = set()
        for track, bar in keys:
            bars = [int(bar)]
            if self._is_split_piece(track):
                view = self._view_for(track)
                clip = view._clip_for(track)
                if clip is not None:
                    bars = [int(span[0]) for span in view._spans_of(clip)] or bars
            for other in bars:
                if (track, other) not in seen:
                    seen.add((track, other))
                    found.append((track, other))
        return found

    def _arrangement_time(self, index: int, song: float) -> float:
        """A track's song-file time ``song`` as arrangement seconds."""
        track = self.project.tracks[index]
        return float(track.get("start", 0.0) or 0.0) + float(song) - float(track.get("offset", 0.0) or 0.0)

    def _split_at(self, index: int, seconds: float) -> bool:
        """Whether a split mark of this track sits at arrangement ``seconds``."""
        track = self.project.tracks[index]
        at = float(track.get("offset", 0.0) or 0.0) + float(seconds) - float(track.get("start", 0.0) or 0.0)
        return any(abs(float(mark) - at) < 1e-3 for mark in track.get("splits") or ())

    def _block_runs(self, keys) -> list[tuple[int, int, int]]:
        """``(track, first bar, last bar)`` for ``keys``: a stem's bars side
        by side become one run; every other block stands alone."""
        tracks = self.project.tracks
        bars: dict[int, set] = {}
        units = []
        for track, bar in keys:
            if 0 <= track < len(tracks) and stem_loop(tracks[track]) is None:
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
                # Bars that touch, including across a split mark, are one run.
                # A combined play of the track is one voice, not a voice per bar.
                if run is not None and abs(run[2] - times[0]) < 1e-3:
                    run = (run[0], bar, times[1])
                    continue
                if run is not None:
                    units.append((track, run[0], run[1]))
                run = (bar, bar, times[1])
            if run is not None:
                units.append((track, run[0], run[1]))
        return units

    def _yield_transport(self) -> None:
        """Blocks are about to sound: pause the song, which would otherwise
        play the same stems a second time, from another place."""
        if self.engine.playing:
            self._pause()

    def _playback_anchor(self, blocks, clicked):
        """The block playback starts from: a clip this drag moved, else the
        block whose ▶ was pressed."""
        tracks = self.project.tracks

        def moved(pair) -> bool:
            track = pair[0]
            return 0 <= track < len(tracks) and id(tracks[track]) in self._moved_ids

        if clicked is not None and moved(clicked):
            return clicked
        for pair in blocks:
            if moved(pair):
                return pair
        return clicked

    def _arrange_moved_selection(self, groups, placed, anchor) -> bool:
        """Play the selected pieces of one stem as one track.

        Each scissor piece is its own clip. Giving every piece a loop of its
        own makes them all start together, so one drums stem sounds like
        several tracks. One clock walks the timeline instead: each piece
        keeps its place, source offset and duration, and sounds only while
        the clock is over it. A single block is left on its song's clock.
        """
        del anchor  # the track starts at its first piece, not at one block
        plain = [name for name, entry in groups.items() if entry[2] is None]
        keys = [key for name in plain for key in groups[name][1]]
        if len(keys) < 2:
            return False
        for name in plain:
            for key, member in groups[name][1].items():
                arr0, arr1, _song, _song_end = placed[key]
                track = self.project.tracks[member[1]]
                start = float(track.get("start", 0.0) or 0.0)
                duration = float(track.get("duration", 0.0) or 0.0)
                covers = bool(track.get("bar_block")) or (
                    abs(arr0 - start) <= 0.05 and abs((arr1 - arr0) - duration) <= 0.05
                )
                if not covers:
                    return False  # a bar inside a longer stem keeps its own slice
        origin = min(placed[key][0] for key in keys)
        members = {}
        loop_end = origin
        for name in plain:
            for key, member in groups.pop(name)[1].items():
                arr0, arr1, song, _song_end = placed[key]
                # shift = offset - start: the clock is arrangement time.
                members[key] = (*member, song - arr0, None)
                loop_end = max(loop_end, arr1)
        length = max(loop_end - origin, 1.0 / OUTPUT_RATE)
        tag = ",".join(f"{key[0]}:{key[1]}" for key in sorted(members))
        groups[f"selection:{tag}"] = [origin, members, (origin, origin + length), loop_end]
        return True

    def _start_blocks(self, keys, retire_keys=(), anchor=None) -> None:
        """Start (or resume) blocks together, on the same audio sample.
        Stems of one song join that song's clock; a selection that spans
        songs shares one clock (see unify_selection_groups). A clip a drag
        moved shares that clock too, starting at its timeline position.

        Pieces of one cut stem play as that one track: one clock walks their
        timeline, and each piece sounds only in its own place. A single
        block stays on its song's clock. A voice this start replaces is
        stopped after the new voices exist, so the same audio is not heard
        twice. Voices outside the selection are not stopped and their clocks
        are not moved."""
        groups: dict[str, list] = {}  # group -> [start, members, loop, end] (song seconds)
        spans: dict[tuple, tuple] = {}  # stem voice key -> (start, end) of its own block, song seconds
        placed: dict[tuple, tuple] = {}  # voice key -> arrangement and song span of the block
        retire = []  # voices a new run replaces; stopped after the run exists, so the clock stays
        units = self._block_runs(self._whole_pieces(keys))
        for track, bar, last in units:
            view = self._view_for(track)
            times = view.block_times(track, bar)
            if times is None:
                continue
            if last != bar:
                times = (times[0], view.block_times(track, last)[1])
            data = self.project.tracks[track]
            group = sync_group(data)
            # Where the block is in the song: the source file's own time.
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
                # So does a voice left under a bar number the clip no longer
                # has: a move renumbers the block, and the old voice would
                # keep playing on its own.
                for other, (_data, run) in list(self._preview_tracks.items()):
                    if other[0] == track and other[1] <= STEM_BLOCK and other != key and run and run[0] <= last and bar <= run[1]:
                        retire.append(other)
                        del self._preview_tracks[other]
                for other in list(self.engine.previews()):
                    if (
                        isinstance(other, tuple)
                        and len(other) == 2
                        and other[0] == track
                        and other != key
                        and other[1] <= STEM_BLOCK
                        and other not in retire
                        and view.block_times(track, stem_block(other[1])) is None
                    ):
                        retire.append(other)
                        self._preview_tracks.pop(other, None)
                self._preview_tracks[key] = (data, (bar, last))
                placed[key] = (times[0], times[1], song, end)
            else:
                key = (track, STEM_VOICE)
                self._preview_tracks[key] = (data, None)
            spans[key] = (song, end)
            entry[1][key] = (key, track, bool(data.get("muted", False)))
        if not groups:
            return
        self._yield_transport()
        # A moved clip joins the selection on one arrangement clock. Otherwise
        # several songs in one selection share one clock, and one song keeps
        # its own so a later block can still join it.
        anchor = self._playback_anchor([(track, bar) for track, bar, _last in units], anchor)
        if not self._arrange_moved_selection(groups, placed, anchor):
            unify_selection_groups(groups)
        joining = {group for group in groups if self.engine.group_position(group) is not None}
        for group, entry in groups.items():
            if entry[2] is not None:
                continue  # the split's synced loop: one region for all
            members = entry[1]
            if group in joining:
                ref = self.engine.group_loop(group)
                # A longer stem joins a clock a shorter one already started.
                # Grow the loop so the long stem plays to its end. The clock
                # stays where it is, and the short stem keeps its own bar.
                if ref is not None:
                    current = ref[1] - ref[0]
                    needed = max(spans[key][1] - spans[key][0] for key in members)
                    if needed > current + 1e-4:
                        ref = (ref[0], ref[0] + needed)
                        entry[2] = ref
            else:
                # The earliest block sets the clock; the loop is as long
                # as the longest block.
                first = min(spans[key][0] for key in members)
                length = max(spans[key][1] - spans[key][0] for key in members)
                ref = entry[2] = (first, first + length)
                entry[0] = first
            for key, member in members.items():
                shift = 0.0 if ref is None else spans[key][0] - ref[0]
                # A block that ends at a split mark does not reach past it.
                sealed = self._split_at(member[1], self._arrangement_time(member[1], spans[key][1]))
                members[key] = (*member, shift, spans[key][1] - spans[key][0], sealed)
        try:
            member_keys = {key for _start, members, _loop, _end in groups.values() for key in members}
            with self.engine.batch():  # everything starts on the same block
                for group, (start, members, loop, end) in groups.items():
                    self.engine.stem_play(group, list(members.values()), max(0.0, start), loop=loop)
                for key in (*retire, *retire_keys):
                    if key not in member_keys:
                        self.engine.preview_stop(key)
        except Exception as exc:
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
        if len(stems) != 1:
            self.statusBar().showMessage(f"Playing {len(stems)} blocks together", 3000)
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
        units = self._group_units()
        if units and (not self.engine.playing or self._members_active()):
            # Repeat means play it now: the selection loops, on the shared
            # clock, joining what already plays.
            self._start_member_entries(units)
        self.statusBar().showMessage(
            f"{count} stem{'s' if count != 1 else ''} loop {start:.2f}–{end:.2f} s in sync", 6000
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
        members = self.engine.members or {}
        for track in tracks:
            key = track.get("node_id")
            if stem_group(track) == group and key in members:
                place = self._unit_place(key)
                if place is not None:
                    self.engine.member_place(key, *place)  # a playing loop takes the new region at once
        self._show_stem_loops()
        if finished:
            self._commit_edit()
            self.statusBar().showMessage(f"Synced loop: {start:.3f}–{end:.3f} s", 4000)

    def _push_stem_mutes(self) -> None:
        """A muted stem stays on its split's clock, silent."""
        for index, bar in list(self._preview_tracks):
            if bar < 0 and index < len(self.project.tracks):
                self.engine.preview_set_muted((index, bar), bool(self.project.tracks[index].get("muted", False)))

    def _stop_block(self, key, generation: int | None = None) -> None:
        """Take one voice out of the mix. ``generation`` (see
        AudioEngine.voice_state) keeps a late request from stopping a newer
        voice under the same key. Other voices are not touched."""
        self.engine.preview_stop(key, generation)
        self._preview_tracks.pop(key, None)
        self._sync_previews()

    def _stop_block_at(self, track: int, bar: int) -> None:
        """Stop the voice playing this block, whichever kind it is."""
        identity = self._member_key(track, bar, self._block_identity(track, bar))
        if identity is not None:
            self._remove_member(identity)  # the rest play on
            return
        key = self._block_voice(track, bar, self.engine.previews())
        self._stop_block(key, self.engine.voice_state(key)[1] or None)

    def _stop_track_voices(self, index: int) -> None:
        """Stop every block voice of one track; the rest play on."""
        for key in [key for key in self.engine.previews() if key[0] == index]:
            self.engine.preview_stop(key)
            self._preview_tracks.pop(key, None)
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
        if self.engine.playing and not self._members_active():
            self.statusBar().showMessage("Those blocks already play with the song", 3000)
            return
        self._start_members(keys)

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
        # Playing blocks show the same way: tint, ▶ / ⏸, their own cursor.
        blocks.update(self._member_marks())
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
        """After an edit: stop voices whose track or block moved or went.
        A split stem's block follows its audio instead (see
        _follow_stem_blocks), so no voice is left playing under a block
        number that no longer shows it."""
        tracks = self.project.tracks
        spans = {row["key"]: row["file_span"] for row in self.engine.voice_report()}
        for key, (track, times) in list(self._preview_tracks.items()):
            index, bar = key
            if bar <= STEM_BLOCK:
                self._follow_stem_block(key, track, spans.get(key))
                continue
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

    def _follow_stem_block(self, key, track, span) -> None:
        """Keep a split stem's block voice on the block that holds its audio
        after an edit, or remove it from the mixer.

        Its key names a block by number. A move (another canvas, another
        time) renumbers the blocks, and a cut gives part of the track to new
        pieces. The voice plays a stretch of its stem file (``span``); the
        clip that holds that stretch now (the same track first, else a piece
        of it on the same row) and the block there become its key; a piece
        must hold the very stretch it played. Otherwise (no such clip, or the
        block was cut) the voice stops: nothing would show it playing."""
        tracks = self.project.tracks
        if span is None or not any(other is track for other in tracks):
            self.engine.preview_stop(key)
            self._preview_tracks.pop(key, None)
            return
        song = float(span[0])
        if any(song + 1e-3 < float(mark) < song + float(span[1]) - 1e-3 for mark in track.get("splits") or ()):
            # Split in Half cut the block it plays: that block is gone, so is
            # its voice. ▶ plays either half.
            self.engine.preview_stop(key)
            self._preview_tracks.pop(key, None)
            return

        def holds(other) -> bool:
            offset = float(other.get("offset", 0.0) or 0.0)
            return offset - 1e-6 <= song < offset + float(other.get("duration", 0.0) or 0.0) - 1e-6

        same = [other for other in tracks if stem_group(other) == stem_group(track) and other.get("path") == track.get("path")]
        place = (track_canvas(track), track.get("lane"))
        ranked = sorted(
            (other for other in same if holds(other)),
            key=lambda other: (other is not track, (track_canvas(other), other.get("lane")) != place),
        )
        if not ranked:
            self.engine.preview_stop(key)
            self._preview_tracks.pop(key, None)
            return
        holder = ranked[0]
        index = next(i for i, other in enumerate(tracks) if other is holder)
        at = float(holder.get("start", 0.0) or 0.0) + song - float(holder.get("offset", 0.0) or 0.0)
        view = self._view_for(index)
        bar = view.block_at_time(index, at + 1e-4)
        times = view.block_times(index, bar) if bar is not None else None
        start = float(holder.get("start", 0.0) or 0.0)
        offset = float(holder.get("offset", 0.0) or 0.0)
        if times is None or (
            holder is not track
            and (
                abs(offset + times[0] - start - song) > 1e-3
                or abs((times[1] - times[0]) - float(span[1])) > 1e-3
            )
        ):
            # Its audio went to another clip whose block is not the stretch
            # it played (the block was cut): its voice leaves the mix, and ▶
            # plays the new blocks. A track moved whole keeps its voice, on
            # the same clock, whatever bars it now lands on.
            self.engine.preview_stop(key)
            self._preview_tracks.pop(key, None)
            return
        new = (index, stem_block(bar))
        if new == key:
            if holder is not track:
                self._preview_tracks[key] = (holder, self._preview_tracks[key][1])
            return
        if new in self.engine.previews():
            self.engine.preview_stop(key)  # that block already has its voice
            self._preview_tracks.pop(key, None)
            return
        run = self._preview_tracks.pop(key)[1]
        width = (run[1] - run[0]) if run else 0
        self.engine.preview_retarget({key: (new, index)})
        self._preview_tracks[new] = (holder, (bar, bar + width))

    def _voice_report_text(self) -> str:
        """Every voice the mixer holds: key, generation, state, whether it
        is mixed into the output, and the block and track it plays."""
        tracks = self.project.tracks
        lines = []
        for row in sorted(self.engine.voice_report(), key=lambda row: (str(row["key"]))):
            index, bar = row["key"] if isinstance(row["key"], tuple) and len(row["key"]) == 2 else (-1, 0)
            name = tracks[index].get("name", "Track") if 0 <= index < len(tracks) else "?"
            if bar == STEM_VOICE:
                block = "synced loop"
            elif bar <= STEM_BLOCK:
                block = f"bar {stem_block(bar) + 1}"
            else:
                block = f"bar {bar + 1}"
            span = row["file_span"]
            where = f" · file {span[0]:.3f}+{span[1]:.3f}s" if span else ""
            mixed = "in the mix" if row["mixed"] else "not mixed"
            lines.append(
                f"{row['key']}  #{row['generation']}  {row['state']:<9}  {mixed:<11}  {name} {block}{where}"
            )
        return "\n".join(lines) if lines else "No voices. Nothing is playing outside the song."

    def _gui_fps(self) -> float:
        """Playhead frames drawn per second over the last second."""
        times = [t for t in self._frame_times if t >= time.monotonic() - 1.0]
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / max(1e-6, times[-1] - times[0])

    def _diagnostics_text(self) -> str:
        """Audio callback timing, misses and underruns, voices (and any
        duplicated), the stream's real format, GUI frame rate and cache use."""
        from soniclab.core.waveform import CACHE_STATS
        from soniclab.ui.timeline import TILE_STATS

        d = self.engine.diagnostics()
        latency = "not open" if d["latency_s"] is None else f"{d['latency_s'] * 1000:.1f} ms"
        lines = [
            f"Audio callback: mean {d['mean_ms']:.3f} ms · p95 {d['p95_ms']:.3f} · p99 {d['p99_ms']:.3f} · "
            f"max {d['max_ms']:.3f} ms of {d['budget_ms']:.1f} ms ({d['load'] * 100:.1f}% load, {d['callbacks']} callbacks)",
            f"Deadline misses: {d['deadline_misses']} · underruns: {d['underruns']}",
            f"Voices: {d['voices']} ({d['mixed_voices']} in the mix) · duplicates: {len(d['duplicates'])}"
            + (f" {d['duplicates']}" if d["duplicates"] else ""),
            f"Output: {d['sample_rate']:.0f} Hz · {d['block_size']} frames a block · latency {latency}"
            f" (asked for {d['requested_latency']})",
            f"GUI: {self._gui_fps():.1f} frames a second",
            f"Waveform caches: envelopes {CACHE_STATS['hits']} hits / {CACHE_STATS['misses']} misses · "
            f"tiles {TILE_STATS['hits']} hits / {TILE_STATS['misses']} misses",
        ]
        return "\n".join(lines)

    def _show_diagnostics(self) -> None:
        text = self._diagnostics_text()
        print(text, flush=True)
        box = QMessageBox(self)
        box.setWindowTitle("Audio Diagnostics")
        box.setText(text)
        box.exec()

    def _print_diagnostics(self) -> None:
        print(f"--- SonicLab diagnostics {time.strftime('%H:%M:%S')}\n{self._diagnostics_text()}", flush=True)

    def _show_voice_report(self) -> None:
        text = self._voice_report_text()
        print(text, flush=True)
        box = QMessageBox(self)
        box.setWindowTitle("Playing Voices")
        box.setText(f"{len(self.engine.voice_report())} voice(s) in the mixer")
        box.setDetailedText(text)
        box.exec()

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
            "beat detection, downbeat looping, the metronome, and the spectrum, "
            "scope, and spectrogram views are active.\n\n"
            "Effects and the sample library are not available yet.",
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
