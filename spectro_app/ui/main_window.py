from PyQt6 import QtWidgets, QtCore
from spectro_app.ui.menus import build_menus
from spectro_app.ui.docks.file_queue import FileQueueDock
from spectro_app.ui.docks.recipe_editor import RecipeEditorDock
from spectro_app.ui.docks.preview_widget import PreviewDock
from spectro_app.ui.docks.qc_widget import QCDock
from spectro_app.ui.docks.logger_view import LoggerDock
from spectro_app.ui.dialogs.about import AboutDialog
from spectro_app.ui.dialogs.help_viewer import HelpViewer
from spectro_app.engine.run_controller import RunController
from spectro_app.engine.recipe_model import Recipe

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, appctx):
        super().__init__()
        self.appctx = appctx
        self.setWindowTitle("Spectroscopy Processing App")
        self.resize(1280, 800)
        build_menus(self)
        self._init_docks()
        self._init_status()
        self.restore_state()
        self.runctl = RunController(self)

    def _init_docks(self):
        self.fileDock = FileQueueDock(self); self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.fileDock)
        self.recipeDock = RecipeEditorDock(self); self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.recipeDock)
        self.previewDock = PreviewDock(self); self.setCentralWidget(self.previewDock)
        self.qcDock = QCDock(self); self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.qcDock)
        self.loggerDock = LoggerDock(self); self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.loggerDock)

    def _init_status(self):
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.status.addPermanentWidget(self.progress)

    # Menu slots (placeholders)
    def on_open(self): pass
    def on_save_recipe(self): pass
    def on_save_recipe_as(self): pass
    def on_export(self): pass
    def on_reset_layout(self): self.restoreState(self.saveState())
    def on_run(self): pass
    def on_cancel(self): pass
    def on_help(self): dlg = HelpViewer(self); dlg.exec()
    def on_about(self): dlg = AboutDialog(self); dlg.exec()

    def closeEvent(self, e):
        if not self.appctx.maybe_close():
            e.ignore(); return
        self.save_state(); super().closeEvent(e)

    def save_state(self):
        s = self.appctx.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())

    def restore_state(self):
        s = self.appctx.settings
        g = s.value("geometry"); w = s.value("windowState")
        if g: self.restoreGeometry(g)
        if w: self.restoreState(w)
