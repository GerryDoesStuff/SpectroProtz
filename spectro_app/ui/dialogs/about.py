from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel

from spectro_app.io.opus import opus_optional_reader_status

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About SpectroApp")
        lay = QVBoxLayout(self)
        status = opus_optional_reader_status()
        readers_text = (
            "Optional OPUS readers:\n"
            f"  - spectrochempy: {'available' if status['spectrochempy'] else 'not installed'}\n"
            f"  - brukeropusreader: {'available' if status['brukeropusreader'] else 'not installed'}"
        )
        lay.addWidget(
            QLabel(
                "SpectroApp\n"
                "Version 0.1.0\n"
                "© 2025 SpectroLab\n\n"
                f"{readers_text}"
            )
        )
