#!/usr/bin/env python3
"""Download one Demucs model into TORCH_HOME, unless it is already there.

Run with the Demucs environment's Python. Progress from the download is
printed for the installer. Weights are not bundled with SonicLab.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MIN_BYTES = 1_000_000


def signatures(model: str) -> list[str]:
    import yaml
    from demucs import pretrained

    remote = Path(pretrained.__file__).parent / "remote" / f"{model}.yaml"
    if not remote.is_file():
        raise SystemExit(f"Demucs does not include a description for {model}.")
    document = yaml.safe_load(remote.read_text())
    models = document.get("models") if isinstance(document, dict) else None
    if isinstance(models, str):
        return [models]
    if isinstance(models, list) and models:
        return [str(item) for item in models]
    raise SystemExit(f"Demucs did not list the files for {model}.")


def cache_folders(torch_home: Path) -> list[Path]:
    folders = [torch_home / "hub" / "checkpoints"]
    for key in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(key):
            folders.append(Path(os.environ[key]))
    if os.environ.get("HF_HOME"):
        folders.append(Path(os.environ["HF_HOME"]) / "hub")
    folders.append(Path.home() / ".cache" / "huggingface" / "hub")
    return folders


def matches(folder: Path, signature: str) -> list[Path]:
    found = []
    patterns = (signature + "-*.th", signature + ".safetensors", "**/" + signature + ".safetensors")
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file() and path.stat().st_size >= MIN_BYTES:
                found.append(path)
    return found


def incomplete(folder: Path, signature: str) -> None:
    for path in folder.glob(signature + "-*.th"):
        if path.is_file() and path.stat().st_size < MIN_BYTES:
            path.unlink()


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "htdemucs_6s"
    home = os.environ.get("TORCH_HOME")
    if not home:
        print("The model folder is not configured.", file=sys.stderr)
        return 1
    folders = cache_folders(Path(home))
    folders[0].mkdir(parents=True, exist_ok=True)
    try:
        names = signatures(model)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Could not read the Demucs model list: {exc}", file=sys.stderr)
        return 1

    for folder in folders:
        for name in names:
            incomplete(folder, name)
    if all(any(matches(folder, name) for folder in folders) for name in names):
        print(f"The {model} model is already downloaded. It will not be downloaded again.")
        for name in names:
            for folder in folders:
                for path in matches(folder, name):
                    print(f"Verified {path.name} ({path.stat().st_size} bytes)")
        return 0

    print(f"Downloading the {model} model. This can take several minutes.")
    try:
        from demucs.pretrained import get_model

        get_model(model)
    except Exception as exc:
        print(f"The stem-splitting model could not be downloaded: {exc}", file=sys.stderr)
        return 1

    missing = [name for name in names if not any(matches(folder, name) for folder in folders)]
    if missing:
        print(
            "The download finished, but the model files are missing or too small: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print(f"Verified the {model} model.")
    for name in names:
        for folder in folders:
            for path in matches(folder, name):
                print(f"Verified {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
