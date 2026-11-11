import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from soniclab.ui.main_window import MainWindow


def configure_macos() -> None:
    """Keep settings data and Demucs weights in the macOS user library.

    The launcher sets the same variables. Setting them here covers a start
    that does not go through the launcher. An empty Demucs path is left
    unset so a developer environment can still be found.
    """
    if sys.platform != "darwin":
        return
    root = Path.home() / "Library" / "Application Support" / "SonicLab"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(root / "models"))
    demucs_root = root / "demucs-venv"
    demucs = demucs_root / "bin" / "python"
    if "SONICLAB_DEMUCS_PYTHON" not in os.environ and (demucs_root / ".soniclab-ready").is_file() and demucs.is_file():
        os.environ["SONICLAB_DEMUCS_PYTHON"] = str(demucs)
    madmom_root = root / "madmom-venv"
    madmom = madmom_root / "bin" / "python"
    worker = root / "madmom_worker.py"
    if "SONICLAB_MADMOM_PYTHON" not in os.environ and (madmom_root / ".soniclab-ready").is_file() and madmom.is_file():
        os.environ["SONICLAB_MADMOM_PYTHON"] = str(madmom)
    if "SONICLAB_MADMOM_WORKER" not in os.environ and worker.is_file():
        os.environ["SONICLAB_MADMOM_WORKER"] = str(worker)
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def main():
    configure_macos()
    app = QApplication(sys.argv)
    app.setApplicationName("SonicLab")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
