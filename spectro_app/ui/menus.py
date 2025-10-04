from PyQt6.QtGui import QAction

def build_menus(window):
    menubar = window.menuBar()

    file_menu = menubar.addMenu("&File")
    file_menu.addAction(_act(window, "Open...", shortcut="Ctrl+O", slot=window.on_open))
    file_menu.addAction(_act(window, "Save Recipe", shortcut="Ctrl+S", slot=window.on_save_recipe))
    file_menu.addAction(_act(window, "Save Recipe As...", slot=window.on_save_recipe_as))
    file_menu.addSeparator()
    file_menu.addAction(_act(window, "Export Workbook...", slot=window.on_export))
    file_menu.addSeparator()
    file_menu.addAction(_act(window, "Exit", shortcut="Ctrl+Q", slot=window.close))

    edit_menu = menubar.addMenu("&Edit")
    edit_menu.addAction(_act(window, "Undo", shortcut="Ctrl+Z"))
    edit_menu.addAction(_act(window, "Redo", shortcut="Ctrl+Y"))
    edit_menu.addSeparator()
    edit_menu.addAction(_act(window, "Cut", shortcut="Ctrl+X"))
    edit_menu.addAction(_act(window, "Copy", shortcut="Ctrl+C"))
    edit_menu.addAction(_act(window, "Paste", shortcut="Ctrl+V"))
    edit_menu.addAction(_act(window, "Delete", shortcut="Del"))
    edit_menu.addAction(_act(window, "Select All", shortcut="Ctrl+A"))

    view_menu = menubar.addMenu("&View")
    view_menu.addAction(_act(window, "Reset Layout", slot=window.on_reset_layout))

    process_menu = menubar.addMenu("&Process")
    process_menu.addAction(_act(window, "Run", shortcut="F5", slot=window.on_run))
    process_menu.addAction(_act(window, "Cancel", shortcut="Esc", slot=window.on_cancel))

    tools_menu = menubar.addMenu("&Tools")
    tools_menu.addAction(_act(window, "Settings...", slot=window.on_settings))
    tools_menu.addAction(
        _act(window, "Check for Updates...", slot=window.on_check_updates)
    )
    tools_menu.addAction(
        _act(window, "Open Log Folder", slot=window.on_open_log_folder)
    )

    help_menu = menubar.addMenu("&Help")
    help_menu.addAction(_act(window, "Help Contents", slot=window.on_help))
    help_menu.addAction(_act(window, "About", slot=window.on_about))

def _act(parent, text, shortcut=None, slot=None):
    a = QAction(text, parent)
    if shortcut:
        a.setShortcut(shortcut)
    if slot:
        a.triggered.connect(slot)
    return a
