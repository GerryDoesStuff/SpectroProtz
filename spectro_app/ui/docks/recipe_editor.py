from PyQt6.QtWidgets import QDockWidget, QWidget, QFormLayout, QComboBox, QSpinBox, QCheckBox

class RecipeEditorDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Recipe Editor", parent)
        w = QWidget(); lay = QFormLayout(w)
        self.module = QComboBox(); self.module.addItems(["uvvis", "ftir", "raman", "pees"])
        self.smooth_enable = QCheckBox(); self.smooth_window = QSpinBox(); self.smooth_window.setRange(0, 999)
        lay.addRow("Module", self.module)
        lay.addRow("Smoothing enabled", self.smooth_enable)
        lay.addRow("Smoothing window", self.smooth_window)
        self.setWidget(w)
