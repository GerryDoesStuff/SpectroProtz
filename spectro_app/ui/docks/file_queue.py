from PyQt6.QtWidgets import QDockWidget, QListWidget

class FileQueueDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("File Queue", parent)
        self.list = QListWidget()
        self.setWidget(self.list)
