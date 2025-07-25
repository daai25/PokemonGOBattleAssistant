import sys, os, glob, math, json
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel
from PyQt5.QtGui import QImage, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableView, QStyledItemDelegate,
    QMessageBox, QFileDialog, QLineEdit, QVBoxLayout, QWidget, QLabel,
    QPushButton, QHBoxLayout, QShortcut, QStyle
)
from PIL import Image

NUMBER_OF_COLUMNS = 8
CELL_PADDING = 10
HIGHLIGHT_COLOR = QColor(0, 120, 215)  # Blue for selection border

# Wähle hier, welche Person du bist (1 oder 2)
PERSON = 1

def load_done_folders(path):
    """Load finished folders from JSON"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def save_done_folders(path, done_folders):
    """Save finished folders to JSON"""
    with open(path, "w") as f:
        json.dump(list(done_folders), f)


class PreviewDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        data = index.model().data(index, Qt.DisplayRole)
        if data is None:
            return

        # Draw image
        width = option.rect.width() - CELL_PADDING * 2
        height = option.rect.height() - CELL_PADDING * 2
        scaled = data.scaled(width, height, aspectRatioMode=Qt.KeepAspectRatio,
                             transformMode=Qt.SmoothTransformation)
        x = option.rect.x() + CELL_PADDING + (width - scaled.width()) / 2
        y = option.rect.y() + CELL_PADDING + (height - scaled.height()) / 2
        painter.drawImage(int(x), int(y), scaled)

        # Draw border if selected
        if option.state & QStyle.State_Selected:
            pen = QPen(HIGHLIGHT_COLOR, 4)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(2, 2, -2, -2))

    def sizeHint(self, option, index):
        return QSize(200, 150)


class PreviewModel(QAbstractTableModel):
    def __init__(self, images):
        super().__init__()
        self.previews = images

    def data(self, index, role):
        idx = index.row() * NUMBER_OF_COLUMNS + index.column()
        if idx >= len(self.previews):
            return None
        if role == Qt.DisplayRole:
            return self.previews[idx]
        return None

    def rowCount(self, parent):
        return math.ceil(len(self.previews) / NUMBER_OF_COLUMNS)

    def columnCount(self, parent):
        return NUMBER_OF_COLUMNS


class MainWindow(QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.done_file = os.path.join(self.folder, "done_folders.json")
        self.done_folders = load_done_folders(self.done_file)

        # Alle Subfolder mit >= 30 Bildern laden
        all_subfolders = []
        for f in os.scandir(folder):
            if f.is_dir() and f.path not in self.done_folders:
                num_images = len(glob.glob(os.path.join(f.path, "*.jpg"))) + \
                             len(glob.glob(os.path.join(f.path, "*.jpeg"))) + \
                             len(glob.glob(os.path.join(f.path, "*.png"))) + \
                             len(glob.glob(os.path.join(f.path, "*.gif")))
                if num_images >= 35:
                    all_subfolders.append(f.path)

        if not all_subfolders:
            QMessageBox.information(self, "Fertig!", "Keine Ordner mit ≥30 Bildern gefunden.")
            sys.exit(0)

        # 👥 Ordner aufteilen für zwei Personen
        half = len(all_subfolders) // 2
        if PERSON == 1:
            self.subfolders = all_subfolders[half:]  # Untere Hälfte
        else:
            self.subfolders = all_subfolders[:half]  # Obere Hälfte

        if not self.subfolders:
            QMessageBox.information(self, "Fertig!", "Keine zugewiesenen Ordner für diese Person.")
            sys.exit(0)

        self.current_index = 0

        # Widgets
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px;")

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter folder number and press Enter...")
        self.input_field.returnPressed.connect(self.jump_to_folder)

        self.prev_button = QPushButton("◀ Previous Folder")
        self.prev_button.clicked.connect(self.prev_folder)
        self.next_button = QPushButton("Next Folder ▶")
        self.next_button.clicked.connect(self.next_folder)
        self.crop_button = QPushButton("Crop Selected Images (Overwrite)")
        self.crop_button.clicked.connect(self.crop_selected_images)

        self.mark_done_button = QPushButton("Ordner als fertig markieren ✅")
        self.mark_done_button.clicked.connect(self.mark_folder_done)

        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectItems)
        self.table.setSelectionMode(QTableView.MultiSelection)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("QTableView { background-color: #222; }")

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.crop_button)
        button_layout.addWidget(self.mark_done_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.input_field)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Shortcuts
        QShortcut(Qt.Key_Left, self, activated=self.prev_folder)
        QShortcut(Qt.Key_Right, self, activated=self.next_folder)
        QShortcut(Qt.Key_Delete, self, activated=self.delete_selected)

        # Load first folder
        self.load_folder(0)
        self.resize(1000, 700)

    def update_title(self):
        folder_name = os.path.basename(self.subfolders[self.current_index])
        self.title_label.setText(f"Current Folder: {folder_name} ({self.current_index + 1} / {len(self.subfolders)})")

    def load_folder(self, idx):
        if 0 <= idx < len(self.subfolders):
            self.current_index = idx
            folder = self.subfolders[idx]
            files = glob.glob(os.path.join(folder, "*.png")) + \
                    glob.glob(os.path.join(folder, "*.jpg")) + \
                    glob.glob(os.path.join(folder, "*.jpeg")) + \
                    glob.glob(os.path.join(folder, "*.gif"))
            imgs = [QImage(f) for f in files]
            self.paths = files
            self.model = PreviewModel(imgs)
            self.table.setModel(self.model)
            delegate = PreviewDelegate()
            self.table.setItemDelegate(delegate)
            self.table.resizeRowsToContents()
            self.table.resizeColumnsToContents()
            self.update_title()

    def prev_folder(self):
        new_idx = (self.current_index - 1) % len(self.subfolders)
        self.load_folder(new_idx)

    def next_folder(self):
        new_idx = (self.current_index + 1) % len(self.subfolders)
        self.load_folder(new_idx)

    def jump_to_folder(self):
        try:
            idx = int(self.input_field.text()) - 1
            if 0 <= idx < len(self.subfolders):
                self.load_folder(idx)
            else:
                QMessageBox.warning(self, "Invalid", f"Number must be between 1 and {len(self.subfolders)}.")
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Please enter a valid number.")

    def delete_selected(self):
        sel = self.table.selectionModel().selectedIndexes()
        if not sel:
            return
        files = set()
        for ix in sel:
            idx = ix.row() * NUMBER_OF_COLUMNS + ix.column()
            if idx < len(self.paths):
                files.add(self.paths[idx])
        reply = QMessageBox.question(self, "Delete?", f"Delete {len(files)} selected images?")
        if reply == QMessageBox.Yes:
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    print("Error deleting:", e)
            self.load_folder(self.current_index)

    def crop_selected_images(self):
        sel = self.table.selectionModel().selectedIndexes()
        if not sel:
            QMessageBox.information(self, "No Selection", "No images selected.")
            return

        files = set()
        for ix in sel:
            idx = ix.row() * NUMBER_OF_COLUMNS + ix.column()
            if idx < len(self.paths):
                files.add(self.paths[idx])

        for path in files:
            try:
                img = Image.open(path)
                width, height = img.size
                crop_box = (
                    int(width * 0.08), int(height * 0.05),
                    int(width * 0.92), int(height * 0.45)
                )
                cropped = img.crop(crop_box)
                cropped.save(path)
            except Exception as e:
                print("Error cropping:", e)

        QMessageBox.information(self, "Done", f"Cropped and saved {len(files)} images.")
        self.load_folder(self.current_index)

    def mark_folder_done(self):
        folder = self.subfolders[self.current_index]
        self.done_folders.add(folder)
        save_done_folders(self.done_file, self.done_folders)
        QMessageBox.information(self, "Fertig", f"Ordner '{os.path.basename(folder)}' wurde als fertig markiert.")
        self.subfolders.pop(self.current_index)
        if not self.subfolders:
            QMessageBox.information(self, "Fertig!", "Alle Ordner sind fertig.")
            sys.exit(0)
        else:
            self.load_folder(self.current_index % len(self.subfolders))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    folder = QFileDialog.getExistingDirectory(None, "Select Root Folder")
    if not folder:
        sys.exit()
    w = MainWindow(folder)
    w.show()
    sys.exit(app.exec_())