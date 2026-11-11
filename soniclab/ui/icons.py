"""Small geometric transport icons drawn at 16px.

One stroke weight and one fill so play, stop, and the unavailable
transport marks belong to the same set.
"""

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF


def _canvas(color: str):
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    return pixmap, painter


def _finish(pixmap, painter) -> QIcon:
    painter.end()
    return QIcon(pixmap)


def play_icon(color: str = "#E6E8EC") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawPolygon(QPolygonF([QPointF(4, 2.5), QPointF(4, 13.5), QPointF(13, 8)]))
    return _finish(pixmap, painter)


def pause_icon(color: str = "#E6E8EC") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawRect(QRectF(3.5, 2.5, 3, 11))
    painter.drawRect(QRectF(9.5, 2.5, 3, 11))
    return _finish(pixmap, painter)


def stop_icon(color: str = "#E6E8EC") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawRect(QRectF(3.5, 3.5, 9, 9))
    return _finish(pixmap, painter)


def return_icon(color: str = "#E6E8EC") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawRect(QRectF(2.5, 3, 2, 10))
    painter.drawPolygon(QPolygonF([QPointF(13, 3), QPointF(6, 8), QPointF(13, 13)]))
    return _finish(pixmap, painter)


def record_icon(color: str = "#5E6772") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawEllipse(QRectF(3.5, 3.5, 9, 9))
    return _finish(pixmap, painter)


def loop_icon(color: str = "#5E6772") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(color), 1.4))
    painter.drawArc(QRectF(3, 3, 10, 10), 40 * 16, 280 * 16)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawPolygon(QPolygonF([QPointF(11, 2), QPointF(14.5, 5), QPointF(10, 5.5)]))
    return _finish(pixmap, painter)


def metronome_icon(color: str = "#5E6772") -> QIcon:
    pixmap, painter = _canvas(color)
    painter.drawPolygon(QPolygonF([QPointF(8, 1.5), QPointF(12.5, 14), QPointF(3.5, 14)]))
    painter.setBrush(QColor("#12151A"))
    painter.drawPolygon(QPolygonF([QPointF(8, 5), QPointF(10.6, 13), QPointF(5.4, 13)]))
    painter.setPen(QPen(QColor(color), 1.3))
    painter.drawLine(QPointF(8, 12), QPointF(11.5, 4))
    return _finish(pixmap, painter)
