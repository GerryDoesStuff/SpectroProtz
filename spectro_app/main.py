import sys
from PyQt6.QtWidgets import QApplication
from spectro_app.ui.main_window import MainWindow
from spectro_app.app_context import AppContext

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpectroApp")
    app.setOrganizationName("SpectroLab")
    ctx = AppContext()
    win = MainWindow(ctx)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
