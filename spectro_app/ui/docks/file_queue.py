from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QDockWidget, QListWidget


from .file_queue_models import QueueEntry, order_queue_entries


__all__ = ["FileQueueDock", "QueueEntry", "order_queue_entries"]


QUEUE_ENTRY_ROLE = QtCore.Qt.ItemDataRole.UserRole


@dataclass
class QueueBadge:
    text: str
    category: str
    tooltip: str = ""


class _QueueItemDelegate(QtWidgets.QStyledItemDelegate):
    BADGE_PADDING = QtCore.QMargins(8, 3, 8, 3)
    BADGE_SPACING = 6
    TEXT_MARGIN = QtCore.QMargins(8, 6, 8, 6)

    _CATEGORY_COLORS: Dict[str, QtGui.QColor] = {
        "technique": QtGui.QColor(45, 101, 191),
        "mode": QtGui.QColor(66, 133, 244),
        "role-sample": QtGui.QColor(15, 157, 88),
        "role-blank": QtGui.QColor(217, 48, 37),
        "role-standard": QtGui.QColor(244, 180, 0),
        "role-unknown": QtGui.QColor(95, 99, 104),
        "manifest-file": QtGui.QColor(52, 168, 83),
        "manifest-ok": QtGui.QColor(15, 157, 88),
        "manifest-missing": QtGui.QColor(217, 48, 37),
        "manifest-error": QtGui.QColor(217, 98, 48),
        "manifest-na": QtGui.QColor(120, 120, 120),
        "default": QtGui.QColor(95, 99, 104),
    }

    def __init__(self, dock: "FileQueueDock", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._dock = dock

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        entry = index.data(QUEUE_ENTRY_ROLE)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        style.drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_PanelItemViewItem, opt, painter, opt.widget)

        painter.save()
        rect = opt.rect.marginsRemoved(self.TEXT_MARGIN)
        metrics = QtGui.QFontMetrics(opt.font)
        badges = self._dock.badges_for_entry(entry) if isinstance(entry, QueueEntry) else []
        badge_layout = self._prepare_badges(metrics, badges, rect)

        text_rect = QtCore.QRect(rect)
        if badge_layout:
            total_width = badge_layout.total_width + self.BADGE_SPACING
            text_rect.setRight(rect.right() - total_width)

        display_text = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        if isinstance(entry, QueueEntry):
            display_text = entry.display_name

        painter.setFont(opt.font)
        if opt.state & QtWidgets.QStyle.StateFlag.State_Selected:
            painter.setPen(opt.palette.color(QtGui.QPalette.ColorRole.HighlightedText))
        else:
            painter.setPen(opt.palette.color(QtGui.QPalette.ColorRole.Text))
        elided = metrics.elidedText(display_text or "", QtCore.Qt.TextElideMode.ElideMiddle, max(text_rect.width(), 10))
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, elided)

        if badge_layout:
            self._draw_badges(painter, opt, metrics, badge_layout)

        painter.restore()

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        metrics = QtGui.QFontMetrics(opt.font)
        height = metrics.height() + self.TEXT_MARGIN.top() + self.TEXT_MARGIN.bottom()
        return QtCore.QSize(opt.rect.width(), height)

    @dataclass
    class _BadgeLayout:
        badges: List[QueueBadge]
        widths: List[int]
        total_width: int
        badge_height: int

    def _prepare_badges(
        self,
        metrics: QtGui.QFontMetrics,
        badges: List[QueueBadge],
        rect: QtCore.QRect,
    ) -> Optional["_QueueItemDelegate._BadgeLayout"]:
        if not badges:
            return None

        badge_height = metrics.height() + self.BADGE_PADDING.top() + self.BADGE_PADDING.bottom()
        widths: List[int] = []
        total_width = 0
        for badge in badges:
            text_width = metrics.horizontalAdvance(badge.text)
            width = text_width + self.BADGE_PADDING.left() + self.BADGE_PADDING.right()
            widths.append(width)
            total_width += width
        total_width += max(len(badges) - 1, 0) * self.BADGE_SPACING
        if total_width >= rect.width():
            # Trim badges if they won't fit; keep the most important ones on the right.
            while widths and total_width >= rect.width():
                removed = widths.pop(0)
                badges.pop(0)
                total_width -= removed
                if widths:
                    total_width -= self.BADGE_SPACING
        return self._BadgeLayout(badges=badges, widths=widths, total_width=total_width, badge_height=badge_height)

    def _draw_badges(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        metrics: QtGui.QFontMetrics,
        layout: "_QueueItemDelegate._BadgeLayout",
    ) -> None:
        rect = option.rect.marginsRemoved(self.TEXT_MARGIN)
        right = rect.right()
        text_pen = option.palette.color(QtGui.QPalette.ColorRole.HighlightedText)
        default_pen = QtGui.QColor("white")

        for badge, width in reversed(list(zip(layout.badges, layout.widths))):
            top = rect.center().y() - layout.badge_height // 2
            badge_rect = QtCore.QRect(right - width, top, width, layout.badge_height)
            right = badge_rect.left() - self.BADGE_SPACING

            color = QtGui.QColor(self._CATEGORY_COLORS.get(badge.category, self._CATEGORY_COLORS["default"]))
            if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
                highlight = option.palette.color(QtGui.QPalette.ColorRole.Highlight)
                color = self._blend_colors(color, highlight)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, 6, 6)
            painter.setPen(text_pen if option.state & QtWidgets.QStyle.StateFlag.State_Selected else default_pen)
            painter.drawText(
                badge_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                badge.text,
            )

    @staticmethod
    def _blend_colors(base: QtGui.QColor, overlay: QtGui.QColor) -> QtGui.QColor:
        ratio = 0.5
        r = int(base.red() * (1 - ratio) + overlay.red() * ratio)
        g = int(base.green() * (1 - ratio) + overlay.green() * ratio)
        b = int(base.blue() * (1 - ratio) + overlay.blue() * ratio)
        return QtGui.QColor(r, g, b)


class FileQueueDock(QDockWidget):
    paths_dropped = QtCore.pyqtSignal(list)
    inspect_requested = QtCore.pyqtSignal(str)
    preview_requested = QtCore.pyqtSignal(str)
    locate_requested = QtCore.pyqtSignal(str)
    overrides_changed = QtCore.pyqtSignal(dict)
    clear_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        plugin_resolver: Optional[Callable[[List[str]], object]] = None,
    ) -> None:
        super().__init__("File Queue", parent)
        self.setObjectName("FileQueueDock")
        self.list = QListWidget()
        self.list.setItemDelegate(_QueueItemDelegate(self, self.list))
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setUniformItemSizes(False)
        self.list.setAlternatingRowColors(False)

        header_widget = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(6, 6, 6, 6)
        header_layout.setSpacing(6)
        header_layout.addStretch(1)

        self._clear_button = QtWidgets.QToolButton(header_widget)
        self._clear_button.setText("Clear queue")
        self._clear_button.setEnabled(False)
        self._clear_button.clicked.connect(self._on_clear_clicked)
        header_layout.addWidget(self._clear_button)

        container = QtWidgets.QWidget(self)
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(header_widget)
        container_layout.addWidget(self.list)

        self.setWidget(container)
        self.setAcceptDrops(True)
        self._plugin_resolver = plugin_resolver
        self._entries: List[QueueEntry] = []
        self._raw_entries: List[QueueEntry] = []
        self._overrides: Dict[str, Dict[str, object]] = {}

    def load_overrides(self, overrides: Dict[str, Dict[str, object]]) -> None:
        """Seed override state without emitting change notifications."""
        loaded: Dict[str, Dict[str, object]] = {}
        if isinstance(overrides, dict):
            for path, values in overrides.items():
                if not isinstance(path, str):
                    continue
                if not isinstance(values, dict):
                    continue
                normalized = self._normalize_path(path)
                loaded[normalized] = dict(values)
        self._overrides = loaded
        if self._raw_entries:
            self._entries = [self._apply_overrides(entry) for entry in self._raw_entries]
            self._refresh_list_widget()

    def set_plugin_resolver(
        self, resolver: Optional[Callable[[List[str]], object]]
    ) -> None:
        self._plugin_resolver = resolver

    def set_entries(self, entries: List[QueueEntry]) -> None:
        selected_path = self._selected_path()
        self.list.clear()
        self._raw_entries = [
            replace(entry, metadata=dict(entry.metadata), overrides=dict(entry.overrides))
            for entry in entries
        ]
        self._prune_overrides()
        applied = [self._apply_overrides(entry) for entry in self._raw_entries]
        self._entries = order_queue_entries(applied)
        self._refresh_list_widget(selected_path)

    def apply_metadata(self, entries: List[QueueEntry]) -> None:
        if not self._entries:
            self.set_entries(entries)
            return
        mapping = {entry.path: entry for entry in entries}
        ordered: List[QueueEntry] = []
        seen: set[str] = set()
        for existing in self._entries:
            updated = mapping.get(existing.path, existing)
            ordered.append(updated)
            seen.add(existing.path)
        for entry in entries:
            if entry.path not in seen:
                ordered.append(entry)
        self.set_entries(ordered)

    def badges_for_entry(self, entry: Optional[QueueEntry]) -> List[QueueBadge]:
        if not isinstance(entry, QueueEntry):
            return []

        badges: List[QueueBadge] = []
        technique = entry.technique or (entry.plugin_id.upper() if entry.plugin_id else None)
        if technique:
            tooltip = f"Technique: {technique}"
            if entry.plugin_id and entry.plugin_id.lower() != technique.lower():
                tooltip += f" ({entry.plugin_id})"
            badges.append(QueueBadge(text=technique, category="technique", tooltip=tooltip))

        if entry.is_manifest:
            if entry.manifest_status == "manifest-error":
                badges.append(
                    QueueBadge(
                        text="Manifest ⚠",
                        category="manifest-error",
                        tooltip="Manifest file could not be parsed",
                    )
                )
            else:
                badges.append(
                    QueueBadge(
                        text="Manifest",
                        category="manifest-file",
                        tooltip="Manifest file",
                    )
                )
            return badges

        mode_badge = self._mode_badge(entry)
        if mode_badge:
            badges.append(mode_badge)

        role_badge = self._role_badge(entry.role)
        if role_badge:
            badges.append(role_badge)

        manifest_badge = self._manifest_badge(entry.manifest_status)
        if manifest_badge:
            badges.append(manifest_badge)

        return badges

    def _mode_badge(self, entry: QueueEntry) -> Optional[QueueBadge]:
        mode = entry.mode
        if not mode and entry.plugin_id == "uvvis":
            mode = "absorbance"
        if not mode:
            return None
        normalized = str(mode).strip().lower()
        mapping = {
            "absorbance": ("A", "Absorbance"),
            "transmittance": ("%T", "Transmittance"),
            "percent_transmittance": ("%T", "Percent transmittance"),
            "reflectance": ("%R", "Reflectance"),
        }
        text, tooltip = mapping.get(normalized, (normalized.title() or "—", normalized.title() or "Unknown mode"))
        return QueueBadge(text=text, category="mode", tooltip=f"Mode: {tooltip}")

    @staticmethod
    def _role_badge(role: Optional[str]) -> Optional[QueueBadge]:
        if role is None:
            return QueueBadge(text="—", category="role-unknown", tooltip="Role not specified")
        normalized = str(role).strip().lower()
        mapping = {
            "blank": QueueBadge(text="Blank", category="role-blank", tooltip="Blank spectrum"),
            "sample": QueueBadge(text="Sample", category="role-sample", tooltip="Sample spectrum"),
            "standard": QueueBadge(text="Standard", category="role-standard", tooltip="Standard"),
        }
        if normalized in mapping:
            return mapping[normalized]
        if not normalized:
            return QueueBadge(text="—", category="role-unknown", tooltip="Role not specified")
        return QueueBadge(text=normalized.title(), category="role-unknown", tooltip=f"Role: {normalized}")

    @staticmethod
    def _manifest_badge(status: Optional[str]) -> Optional[QueueBadge]:
        mapping = {
            "linked": QueueBadge(text="Manifest (deprecated)", category="manifest-ok", tooltip="Legacy manifest metadata applied; feature is deprecated"),
            "missing": QueueBadge(text="Manifest Missing", category="manifest-missing", tooltip="Deprecated manifest file provided but no rows matched"),
            "none": QueueBadge(text="No Manifest", category="manifest-na", tooltip="Manifests are ignored unless explicitly enabled"),
            "unsupported": QueueBadge(text="Manifest N/A", category="manifest-na", tooltip="Plugin does not expose the deprecated manifest feature"),
            "manifest-error": QueueBadge(text="Manifest ⚠", category="manifest-error", tooltip="Deprecated manifest metadata could not be read"),
        }
        if not status:
            return None
        badge = mapping.get(status)
        if badge:
            return badge
        return QueueBadge(text=status.title(), category="manifest-na", tooltip=f"Manifest status: {status}")

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

        plugin = self._resolve_plugin_for_paths(ordered_paths)
        manifest_ui_enabled = self._plugin_supports_manifest_ui(plugin)

        manifest_paths: List[Path] = []
        data_paths: List[Path] = []
        plugin = self._resolve_plugin_for_paths(ordered_paths)
        manifest_supported = bool(
            plugin and getattr(plugin, "manifest_ui_capability_enabled", False)
        )


        if manifest_ui_enabled and plugin and hasattr(plugin, "_is_manifest_file"):
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
            "Manifest ingestion is deprecated and disabled by default; include the"
            " manifests anyway when updating the queue?"
        )
        if names:
            message += f"\n\nDetected manifests:\n{names}"

        choice = QtWidgets.QMessageBox.question(
            self,
            "Include Deprecated Manifest Files?",
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

    @staticmethod
    def _plugin_supports_manifest_ui(plugin: Optional[object]) -> bool:
        if not plugin or not hasattr(plugin, "_is_manifest_file"):
            return False

        supports_manifest = getattr(plugin, "supports_manifest_ui", None)
        if callable(supports_manifest):
            try:
                return bool(supports_manifest())
            except Exception:
                return True

        expose_attr = getattr(plugin, "expose_manifest_ui", None)
        if expose_attr is None:
            return True

        try:
            return bool(expose_attr()) if callable(expose_attr) else bool(expose_attr)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Context menu actions
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:  # type: ignore[name-defined]
        menu = QtWidgets.QMenu(self)
        selected_path = self._selected_path()

        inspect_action = menu.addAction("Inspect Header…")
        preview_action = menu.addAction("Preview…")
        menu.addSeparator()
        role_menu = menu.addMenu("Set Role")
        auto_role_action = role_menu.addAction("Auto-detect")
        sample_role_action = role_menu.addAction("Sample")
        blank_role_action = role_menu.addAction("Blank…")
        standard_role_action = role_menu.addAction("Standard")
        edit_blank_id_action = menu.addAction("Edit Blank ID…")
        menu.addSeparator()
        locate_action = menu.addAction("Show in File Manager")

        enabled = bool(selected_path)
        inspect_action.setEnabled(enabled)
        preview_action.setEnabled(enabled)
        role_menu.setEnabled(enabled)
        locate_action.setEnabled(enabled)
        entry = self._entry_for_path(selected_path) if selected_path else None
        is_blank = bool(entry and (entry.role == "blank"))
        edit_blank_id_action.setEnabled(enabled and is_blank)

        action = menu.exec(event.globalPos()) if menu.actions() else None
        if not action or not selected_path:
            return
        path = selected_path
        if action is inspect_action:
            self.inspect_requested.emit(path)
        elif action is preview_action:
            self.preview_requested.emit(path)
        elif action is auto_role_action:
            self._set_role_auto(path)
        elif action is sample_role_action:
            self._set_role_override(path, "sample")
        elif action is blank_role_action:
            self._set_role_blank(path)
        elif action is standard_role_action:
            self._set_role_override(path, "standard")
        elif action is edit_blank_id_action:
            self._edit_blank_id(path)
        elif action is locate_action:
            self.locate_requested.emit(path)

    def _selected_path(self) -> Optional[str]:
        items = self.list.selectedItems()
        if not items:
            return None
        item = items[0]
        entry = item.data(QUEUE_ENTRY_ROLE)
        if isinstance(entry, QueueEntry):
            return entry.path
        data = item.data(QtCore.Qt.ItemDataRole.DisplayRole)
        if isinstance(data, str):
            return data
        return item.text()

    # ------------------------------------------------------------------
    # Override management helpers
    def _apply_overrides(self, entry: QueueEntry) -> QueueEntry:
        normalized = self._normalize_path(entry.path)
        overrides = self._overrides.get(normalized)
        if not overrides:
            return replace(entry, overrides={})

        updated = replace(entry)
        updated.metadata = dict(entry.metadata)
        updated.overrides = dict(overrides)

        role_override = overrides.get("role")
        if isinstance(role_override, str):
            normalized_role = role_override.strip().lower()
            updated.role = normalized_role or None
            if normalized_role:
                updated.metadata["role"] = normalized_role
        blank_id_override = overrides.get("blank_id")
        if isinstance(blank_id_override, str) and blank_id_override.strip():
            updated.metadata["blank_id"] = blank_id_override.strip()
        return updated

    def _entry_for_path(self, path: Optional[str]) -> Optional[QueueEntry]:
        if not path:
            return None
        for entry in self._entries:
            if entry.path == path:
                return entry
        return None

    def _normalize_path(self, path: str) -> str:
        try:
            return str(Path(path))
        except Exception:
            return str(path)

    def _update_overrides(self, path: str, values: Dict[str, object], *, clear: Iterable[str] = ()) -> None:
        normalized = self._normalize_path(path)
        current = dict(self._overrides.get(normalized, {}))
        changed = False

        for key in clear:
            if key in current:
                current.pop(key)
                changed = True

        for key, value in values.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                if key in current:
                    current.pop(key)
                    changed = True
                continue
            if current.get(key) != value:
                current[key] = value
                changed = True

        if current:
            self._overrides[normalized] = current
        elif normalized in self._overrides:
            self._overrides.pop(normalized, None)
            changed = True

        if changed:
            self._emit_overrides_changed()
        applied = [self._apply_overrides(entry) for entry in self._raw_entries]
        self._entries = order_queue_entries(applied)
        self._refresh_list_widget()

    def _refresh_list_widget(self, selected: Optional[str] = None) -> None:
        if selected is None:
            selected = self._selected_path()
        self.list.clear()
        for entry in self._entries:
            item = QtWidgets.QListWidgetItem(entry.display_name)
            item.setData(QUEUE_ENTRY_ROLE, entry)
            tooltip_lines = [entry.path]
            if entry.manifest_status:
                tooltip_lines.append(f"Manifest: {entry.manifest_status}")
            if entry.metadata:
                for key, value in entry.metadata.items():
                    tooltip_lines.append(f"{key}: {value}")
            if entry.overrides:
                for key, value in entry.overrides.items():
                    tooltip_lines.append(f"Override {key}: {value}")
            item.setToolTip("\n".join(tooltip_lines))
            self.list.addItem(item)
            if selected and entry.path == selected:
                item.setSelected(True)
        if hasattr(self, "_clear_button"):
            self._clear_button.setEnabled(bool(self._entries))

    def _on_clear_clicked(self) -> None:
        self.clear_requested.emit()

    def _set_role_auto(self, path: str) -> None:
        self._update_overrides(path, {}, clear=("role", "blank_id"))

    def _set_role_override(self, path: str, role: str) -> None:
        normalized_role = role.strip().lower()
        clears = ("blank_id",) if normalized_role != "blank" else ()
        self._update_overrides(path, {"role": normalized_role}, clear=clears)

    def _set_role_blank(self, path: str) -> None:
        entry = self._entry_for_path(path)
        current_override = self._overrides.get(self._normalize_path(path), {})
        default_blank = (
            current_override.get("blank_id")
            or (entry.metadata.get("blank_id") if entry else None)
        )
        if not default_blank:
            try:
                default_blank = Path(path).stem
            except Exception:
                default_blank = ""
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set Blank Identifier",
            "Blank identifier (name used to match samples to this blank; leave empty if matching by cuvette slot):",
            text=str(default_blank or ""),
        )
        if not ok:
            return
        value = text.strip()
        overrides = {"role": "blank"}
        clears: tuple[str, ...] = ()
        if value:
            overrides["blank_id"] = value
        else:
            clears = ("blank_id",)
        self._update_overrides(path, overrides, clear=clears)

    def _edit_blank_id(self, path: str) -> None:
        entry = self._entry_for_path(path)
        if not entry:
            return
        current = (
            self._overrides.get(self._normalize_path(path), {}).get("blank_id")
            or entry.metadata.get("blank_id")
            or ""
        )
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Edit Blank Identifier",
            "Blank identifier (name used to match samples to this blank; leave empty if matching by cuvette slot):",
            text=str(current or ""),
        )
        if not ok:
            return
        value = text.strip()
        overrides = {"role": "blank"}
        clears: tuple[str, ...] = ()
        if value:
            overrides["blank_id"] = value
        else:
            clears = ("blank_id",)
        self._update_overrides(path, overrides, clear=clears)

    def _prune_overrides(self) -> None:
        valid_paths = {self._normalize_path(entry.path) for entry in self._raw_entries}
        removed = [key for key in self._overrides if key not in valid_paths]
        if not removed:
            return
        for key in removed:
            self._overrides.pop(key, None)
        self._emit_overrides_changed()

    def _emit_overrides_changed(self) -> None:
        payload = {path: dict(values) for path, values in self._overrides.items()}
        self.overrides_changed.emit(payload)
