from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget, QListWidget


class FileQueueDock(QDockWidget):
    paths_dropped = QtCore.pyqtSignal(list)
    inspect_requested = QtCore.pyqtSignal(str)
    preview_requested = QtCore.pyqtSignal(str)
    locate_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        plugin_resolver: Optional[Callable[[List[str]], object]] = None,
    ) -> None:
        super().__init__("File Queue", parent)
        self.list = QListWidget()
        self.setWidget(self.list)
        self.setAcceptDrops(True)
        self._plugin_resolver = plugin_resolver

    def set_plugin_resolver(
        self, resolver: Optional[Callable[[List[str]], object]]
    ) -> None:
        self._plugin_resolver = resolver

    # ------------------------------------------------------------------
    # Drag and drop handling
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # type: ignore[name-defined]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # type: ignore[name-defined]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # type: ignore[name-defined]
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()

        urls = event.mimeData().urls()
        ordered_paths: List[Path] = []
        seen: set[Path] = set()

        for url in urls:
            if not url.isLocalFile():
                continue
            resolved = Path(url.toLocalFile()).expanduser().resolve()
            if resolved in seen:
                continue
            if resolved.is_dir():
                for path in sorted(self._iter_directory(resolved)):
                    if path not in seen:
                        ordered_paths.append(path)
                        seen.add(path)
                continue
            if resolved.is_file():
                ordered_paths.append(resolved)
                seen.add(resolved)

        if not ordered_paths:
            return

        manifest_paths: List[Path] = []
        data_paths: List[Path] = []
        plugin = self._resolve_plugin_for_paths(ordered_paths)

        if plugin and hasattr(plugin, "_is_manifest_file"):
            for path in ordered_paths:
                try:
                    is_manifest = bool(plugin._is_manifest_file(path))  # type: ignore[attr-defined]
                except Exception:
                    is_manifest = False
                if is_manifest:
                    manifest_paths.append(path)
                else:
                    data_paths.append(path)
        else:
            data_paths = list(ordered_paths)

        include_manifests = True
        if manifest_paths and data_paths:
            include_manifests = self._confirm_manifest_inclusion(manifest_paths)
            if include_manifests is None:
                return

        final_paths: List[str]
        if include_manifests:
            final_paths = [str(p) for p in ordered_paths]
        else:
            final_paths = [str(p) for p in ordered_paths if p in data_paths]

        if final_paths:
            self.paths_dropped.emit(final_paths)

    def _iter_directory(self, directory: Path) -> Iterable[Path]:
        for candidate in directory.rglob("*"):
            if candidate.is_file():
                yield candidate

    def _confirm_manifest_inclusion(self, manifest_paths: List[Path]) -> Optional[bool]:
        names = "\n".join(path.name for path in manifest_paths[:5])
        if len(manifest_paths) > 5:
            names += "\n…"
        message = (
            "Manifest files were detected along with spectra.\n\n"
            "Include the manifests when updating the queue?"
        )
        if names:
            message += f"\n\nDetected manifests:\n{names}"

        choice = QtWidgets.QMessageBox.question(
            self,
            "Include Manifest Files?",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )

        if choice == QtWidgets.QMessageBox.StandardButton.Cancel:
            return None
        if choice == QtWidgets.QMessageBox.StandardButton.No:
            return False
        return True

    def _resolve_plugin_for_paths(self, paths: Iterable[Path]):
        if not self._plugin_resolver:
            return None
        try:
            return self._plugin_resolver([str(p) for p in paths])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Context menu actions
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:  # type: ignore[name-defined]
        menu = QtWidgets.QMenu(self)
        selected_path = self._selected_path()

        inspect_action = menu.addAction("Inspect Header…")
        preview_action = menu.addAction("Preview…")
        menu.addSeparator()
        locate_action = menu.addAction("Show in File Manager")

        enabled = bool(selected_path)
        inspect_action.setEnabled(enabled)
        preview_action.setEnabled(enabled)
        locate_action.setEnabled(enabled)

        action = menu.exec(event.globalPos()) if menu.actions() else None
        if not action or not selected_path:
            return
        path = selected_path
        if action is inspect_action:
            self.inspect_requested.emit(path)
        elif action is preview_action:
            self.preview_requested.emit(path)
        elif action is locate_action:
            self.locate_requested.emit(path)

    def _selected_path(self) -> Optional[str]:
        items = self.list.selectedItems()
        if not items:
            return None
        return items[0].text()
