from PyQt6.QtWidgets import QDockWidget, QLabel

class QCDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("QC Panel", parent)
        self.setWidget(QLabel("QC table and charts go here"))
