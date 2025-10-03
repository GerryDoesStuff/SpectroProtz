from PyQt6.QtWidgets import QDockWidget, QLabel

class PreviewDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Preview", parent)
        self.setWidget(QLabel("Plot preview goes here"))
