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
    infer_python = root / "venv" / "bin" / "python"
    infer_worker = root / "madmom_infer_worker.py"
    if (
        "SONICLAB_MADMOM_PYTHON" not in os.environ
        and (root / ".soniclab-infer-ready").is_file()
        and infer_python.is_file()
        and infer_worker.is_file()
    ):
        os.environ["SONICLAB_MADMOM_PYTHON"] = str(infer_python)
        os.environ.setdefault("SONICLAB_MADMOM_WORKER", str(infer_worker))
    madmom_root = root / "madmom-venv"
    madmom = madmom_root / "bin" / "python"
    worker = root / "madmom_worker.py"
    if "SONICLAB_MADMOM_PYTHON" not in os.environ and (madmom_root / ".soniclab-ready").is_file() and madmom.is_file():
        os.environ["SONICLAB_MADMOM_PYTHON"] = str(madmom)
    if "SONICLAB_MADMOM_WORKER" not in os.environ and worker.is_file():
        os.environ["SONICLAB_MADMOM_WORKER"] = str(worker)
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def configure_drag_log() -> None:
    """Write drag-and-drop decisions and any uncaught error to
    ``~/.cache/soniclab/drag.log`` (kept small), so a drag that does not do
    what it should can be traced after the fact."""
    import logging
    import traceback
    from logging.handlers import RotatingFileHandler

    folder = Path.home() / ".cache" / "soniclab"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(folder / "drag.log", maxBytes=512_000, backupCount=1, encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger("soniclab.drag")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    previous = sys.excepthook

    def log_errors(kind, value, trace):
        logger.error("uncaught error:\n%s", "".join(traceback.format_exception(kind, value, trace)))
        previous(kind, value, trace)

    sys.excepthook = log_errors


def main():
    configure_macos()
    configure_drag_log()
    app = QApplication(sys.argv)
    app.setApplicationName("SonicLab")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
