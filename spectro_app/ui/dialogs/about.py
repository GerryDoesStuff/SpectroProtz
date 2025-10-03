from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SpectroApp")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("SpectroApp\nVersion 0.1.0\n© 2025 SpectroLab"))
