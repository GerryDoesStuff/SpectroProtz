from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextBrowser

class HelpViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        lay = QVBoxLayout(self)
        br = QTextBrowser()
        br.setHtml("""<h3>Getting Started</h3><p>Load spectra, choose a module, set a recipe, then Run.</p>""") 
        lay.addWidget(br)
