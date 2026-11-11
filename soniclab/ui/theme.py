"""Shared colors and the application stylesheet.

Surfaces stay near-black. Amber is reserved for time, selection, and the
playhead so it reads as the instrument's only highlight.
"""

from PySide6.QtGui import QColor, QPalette

WINDOW = "#0C0D10"
PANEL = "#12151A"
PANEL_RAISED = "#181C22"
RULER = "#101318"
LINE = "#2A3038"
LINE_SOFT = "#1E232B"
TEXT = "#E6E8EC"
TEXT_DIM = "#8B939F"
TEXT_DISABLED = "#5E6772"
AMBER = "#D7B36A"
AMBER_DIM = "#8A7040"
LCD = "#E7F2EA"
SELECT = "#243044"

DARK_STYLE = """
QMainWindow, QDialog, QMessageBox, QInputDialog {
    background: #0C0D10;
    color: #E6E8EC;
    font-family: "DejaVu Sans", sans-serif;
    font-size: 12px;
}
QMenuBar {
    background: #101318;
    color: #C5CAD3;
    border-bottom: 1px solid #2A3038;
    spacing: 0;
    padding: 0 4px;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
}
QMenuBar::item:selected { background: #222831; }
QMenu {
    background: #161B21;
    color: #E6E8EC;
    border: 1px solid #2A3038;
    padding: 4px 0;
}
QMenu::item { padding: 5px 28px 5px 18px; }
QMenu::item:selected { background: #243044; }
QMenu::item:disabled { color: #5E6772; }
QMenu::separator { height: 1px; background: #2A3038; margin: 4px 8px; }
QToolBar#transportBar {
    background: #101318;
    border: 0;
    border-bottom: 1px solid #2A3038;
    spacing: 4px;
    padding: 4px 8px;
}
QToolButton {
    background: transparent;
    color: #C5CAD3;
    border: 0;
    border-radius: 0;
    padding: 4px 8px;
    font-size: 11px;
}
QToolButton:hover { background: #222831; color: #F2F4F7; }
QToolButton:pressed { background: #0C0D10; }
QToolButton:disabled { color: #5E6772; background: transparent; }
QToolButton#playBtn[playing="true"] {
    background: #3A3424;
    color: #F4E7CC;
}
QToolButton#loopBtn { padding: 2px; min-width: 28px; min-height: 26px; }
QToolButton#loopBtn[looping="true"] {
    background: #3A3424;
    color: #F4E7CC;
}
QToolButton#iconBtn { padding: 2px; min-width: 28px; min-height: 26px; }
QToolButton#mixBtn {
    padding: 0;
    min-width: 18px;
    max-width: 18px;
    min-height: 16px;
    max-height: 16px;
    font-size: 9px;
    font-weight: 700;
    color: #8B939F;
    border: 1px solid #2A3038;
}
QToolButton#mixBtn:hover { color: #E6E8EC; background: #222831; }
QToolButton#mixBtn:checked { color: #F4E7CC; background: #3A3424; border-color: #8A7040; }
QToolButton#mixBtn:disabled { color: #4E5660; border-color: #23282F; }
QToolButton#futureBtn {
    color: #5E6772;
    border: 1px solid #23282F;
    padding: 2px;
}
QToolButton#futureBtn:hover { background: #1A1E24; color: #8B939F; }
QToolButton#armBtn:checked { color: #F4E4E4; background: #6A3034; border-color: #A65A5A; }
QToolButton#recordBtn[recording="true"] { background: #6A3034; }
QToolButton#panelBtn {
    border: 1px solid #2A3038;
    padding: 4px 10px;
}
QToolButton#panelBtn:hover { background: #222831; color: #F2F4F7; }
QToolButton#panelBtn:disabled { color: #4E5660; border-color: #23282F; }
QToolButton#primaryBtn {
    background: #3A3424;
    color: #F4E7CC;
    border: 1px solid #8A7040;
    padding: 4px 14px;
}
QToolButton#primaryBtn:hover { background: #4A4230; }
QToolButton#primaryBtn:disabled { background: transparent; color: #4E5660; border-color: #23282F; }
QToolButton#playStemBtn {
    border: 1px solid #2A3038;
    padding: 0;
    min-width: 22px;
    max-width: 22px;
    min-height: 18px;
    max-height: 18px;
    font-size: 9px;
}
QToolButton#playStemBtn:hover { background: #222831; }
QToolButton#playStemBtn:checked { color: #F4E7CC; background: #3A3424; border-color: #8A7040; }
QToolButton#playStemBtn:disabled { color: #4E5660; border-color: #23282F; }
QLabel { background: transparent; color: #E6E8EC; }
QLabel#wordmark {
    color: #B7BCC4;
    font-size: 12px;
    font-weight: 600;
    padding: 0 6px 0 2px;
}
QLabel#sectionLabel {
    color: #8B939F;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
}
QLabel#position {
    color: #E7F2EA;
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 16px;
    font-weight: 700;
}
QLabel#musical {
    color: #D7B36A;
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 13px;
    font-weight: 700;
}
QLabel#dim { color: #8B939F; font-size: 11px; }
QLabel#trackIndex {
    color: #D7B36A;
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 10px;
}
QFrame#lcd {
    background: #0C0F13;
    border: 1px solid #2A3038;
}
QFrame#trackHeader { background: #12151A; border-bottom: 1px solid #1E232B; }
QFrame#trackHeader[selected="true"] { background: #1A2330; }
QFrame#rulerCorner { background: #101318; border-bottom: 1px solid #2A3038; }
QFrame#mixerStrip { background: transparent; border-right: 1px solid #1E232B; }
QFrame#mixerStrip[selected="true"] { background: #1A2330; }
QLineEdit#projectName {
    background: transparent;
    color: #E7E4DC;
    border: 0;
    border-bottom: 1px solid transparent;
    font-weight: 600;
    padding: 2px 6px;
    min-width: 140px;
}
QLineEdit#projectName:focus { border-bottom: 1px solid #8A7040; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #0E1114;
    color: #E6E8EC;
    border: 1px solid #2A3038;
    border-radius: 0;
    padding: 3px 6px;
    selection-background-color: #3A3424;
    selection-color: #F4E7CC;
}
QDoubleSpinBox, QComboBox { font-family: "DejaVu Sans Mono", monospace; }
QComboBox::drop-down { border: 0; width: 14px; }
QComboBox QAbstractItemView {
    background: #161B21;
    color: #E6E8EC;
    border: 1px solid #2A3038;
    selection-background-color: #243044;
}
QDockWidget { color: #C5CAD3; }
QDockWidget::title {
    background: #101318;
    border: 0;
    border-bottom: 1px solid #2A3038;
    padding: 5px 8px;
    text-align: left;
}
QDockWidget > QWidget { background: #12151A; }
QTreeWidget {
    background: #12151A;
    color: #E6E8EC;
    border: 0;
    outline: 0;
    font-size: 12px;
}
QTreeWidget::item { height: 22px; padding: 1px 4px; }
QTreeWidget::item:selected { background: #243044; color: #F4F6F8; }
QTreeWidget::item:hover { background: #1C2128; }
QTreeWidget::item:disabled { color: #5E6772; }
QHeaderView::section {
    background: #101318;
    color: #8B939F;
    border: 0;
    border-bottom: 1px solid #2A3038;
    padding: 4px 6px;
    font-size: 10px;
}
QScrollArea { background: #12151A; border: 0; }
QScrollBar:horizontal, QScrollBar:vertical {
    background: #101318;
    margin: 0;
    border: 0;
}
QScrollBar:horizontal { height: 10px; }
QScrollBar:vertical { width: 10px; }
QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
    background: #3A424E;
    min-width: 24px;
    min-height: 24px;
}
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: #2A3038; }
QStatusBar {
    background: #101318;
    color: #8B939F;
    border-top: 1px solid #2A3038;
}
QStatusBar::item { border: 0; }
QFrame#stemBanner {
    background: #1B2430;
    border-bottom: 1px solid #3A4A5E;
}
QLabel#stemLabel { color: #E6E8EC; font-weight: 600; }
QProgressBar#stemProgress {
    background: #0E1116;
    border: 1px solid #2A3038;
    border-radius: 4px;
}
QProgressBar#stemProgress::chunk { background: #D7B36A; border-radius: 3px; }
QToolTip {
    background: #161B21;
    color: #E6E8EC;
    border: 1px solid #2A3038;
    padding: 4px 6px;
}
QSlider::groove:horizontal { height: 2px; background: #2C333C; }
QSlider::handle:horizontal {
    background: #C8CED6;
    width: 8px;
    margin: -5px 0;
    border: 0;
}
QSlider::groove:vertical { width: 2px; background: #2C333C; }
QSlider::handle:vertical {
    background: #C8CED6;
    height: 8px;
    margin: 0 -6px;
    border: 0;
}
QCheckBox { color: #E6E8EC; spacing: 6px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #3A424E; background: #0E1114; }
QCheckBox::indicator:checked { background: #8A7040; border-color: #D7B36A; }
QCheckBox:disabled { color: #5E6772; }
QCheckBox::indicator:disabled { background: #12151A; border-color: #23282F; }
"""


def build_palette() -> QPalette:
    palette = QPalette()
    window = QColor(WINDOW)
    panel = QColor(PANEL)
    text = QColor(TEXT)
    disabled = QColor(TEXT_DISABLED)
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, QColor("#0E1114"))
    palette.setColor(QPalette.ColorRole.AlternateBase, panel)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(PANEL_RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(SELECT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#F4E7CC"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    return palette
