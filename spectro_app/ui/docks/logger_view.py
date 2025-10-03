from PyQt6.QtWidgets import QDockWidget, QTextEdit

class LoggerDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Logger", parent)
        self.text = QTextEdit(); self.text.setReadOnly(True)
        self.setWidget(self.text)
