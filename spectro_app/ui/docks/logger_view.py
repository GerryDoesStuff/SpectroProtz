from __future__ import annotations

from typing import Iterable

from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QDockWidget, QTextEdit

class LoggerDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Logger", parent)
        self.text = QTextEdit(); self.text.setReadOnly(True)
        self.setWidget(self.text)

    def clear(self):
        self.text.clear()

    def append_line(self, line: str):
        self.text.append(line)
        self.text.moveCursor(QTextCursor.MoveOperation.End)

    def stream_lines(self, lines: Iterable[str]):
        for line in lines:
            self.append_line(line)
