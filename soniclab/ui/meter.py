from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


class PeakMeter(QWidget):
    """Vertical bar driven by peak values from the mixer, not by a timer animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = 0.0
        self._display = 0.0
        self.setFixedWidth(8)
        self.setMinimumHeight(48)

    def set_level(self, level: float) -> None:
        self._target = max(0.0, min(1.0, float(level)))
        if self._target >= self._display:
            self._display = self._target
        else:
            self._display *= 0.72
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0E1114"))
        height = int(self.height() * self._display)
        if height <= 0:
            return
        color = QColor("#C4A46A") if self._display >= 0.92 else QColor("#8E9AAD")
        painter.fillRect(0, self.height() - height, self.width(), height, color)
        painter.end()
