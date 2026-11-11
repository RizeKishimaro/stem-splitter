# Install SonicLab on a Mac

1. Open the `macos` folder.
2. Double-click **Install SonicLab.command**.
3. If the Mac says the file cannot be opened, right-click it, choose **Open**, then **Open** again.
4. Leave the window alone until it says **SonicLab is ready.**
5. Open **SonicLab** from Applications or the Desktop.

No Terminal commands are required. The window that opens is the installer itself.

**Update SonicLab.command** installs a newer copy and keeps projects, preferences, and downloaded models.

**Uninstall SonicLab.command** asks before it removes anything. It does not delete projects or personal audio files. Models and generated stem files in SonicLab's library folder are removed only if you say so. The Audio Unit plugin is removed only if you say so.

**Launch SonicLab.command** opens the installed application. The normal way to open SonicLab is the SonicLab icon.

Logs are in `~/Library/Logs/SonicLab/`. `install.log` is the setup log. `soniclab.log` is the application log.

## What the installer does

It checks the macOS version, Apple Silicon or Intel, free disk space, and the network. It needs macOS 12 or newer and about 8 GB free.

It installs Python 3.12 from python.org when this Mac does not already have Python 3.12. The official installer is one universal package for Apple Silicon and Intel. macOS asks for your password for that step only.

Beat analysis uses a second, private Python 3.11. If that part cannot be installed or compiled, SonicLab still opens and says advanced Madmom analysis is unavailable until it is repaired. Repair it by double-clicking **Update SonicLab.command**. The details are in `~/Library/Logs/SonicLab/install.log`.

SonicLab's libraries go in a private folder:

`~/Library/Application Support/SonicLab/`

The downloaded project folder can be moved or deleted after setup finishes. Projects you save stay wherever you saved them. Preferences stay in macOS Preferences. Stem-splitting weights stay in `Application Support/SonicLab/models` and are not downloaded again when they are already there.

SoundFile and SoundDevice are installed from packages that already contain libsndfile and PortAudio. Homebrew is not used.

## Stem splitting

Demucs is optional. The installer asks for this pair and checks it after installation:

- Demucs 4.0.1
- PyTorch 2.2.2 and torchaudio 2.2.2

A different PyTorch is not accepted. Newer PyTorch has no official Intel Mac installer, and a newer torchaudio no longer provides the audio writer Demucs 4.0.1 uses. Demucs 4.1 is not used, because it can require a compiler on Intel Macs.

The model it prepares is `htdemucs_6s` (vocals, drums, bass, guitar, piano, other). Piano is the weakest of those stems. Weights are downloaded into the Application Support models folder, with progress shown in the setup window. If the files are already present and large enough, they are not downloaded again.

The CPU is always available. PyTorch on Apple Silicon can report MPS. This setup does not measure whether MPS is faster or correct. SonicLab falls back to the CPU when an MPS run fails.

If stem splitting cannot be installed, SonicLab still opens and the window explains that.

## The application

`macos/build_app.sh` creates `SonicLab.app`. The installer runs it.

PyInstaller and Nuitka were not used. Freezing PySide6 that way often drops Qt's Mac window plugin, and neither tool can be run for a Mac from the Linux machine where this package is prepared. The bundle starts the private Python environment. It has an icon, a bundle id (`org.soniclab.app`), Retina enabled, and a microphone prompt for recording. It does not open a Terminal window.

## Plugins

The Beat Analyzer Audio Unit is optional. The installer does not build it on Linux.

On a Mac it installs a `.component` from `macos/plugins/` when one was included. If a plugin is already in `~/Library/Audio/Plug-Ins/Components/`, it asks before replacing it. It then checks the file and, when `auval` is available, runs `auval -v aufx Btan Snlb`. It does not say the plugin is ready unless that check passes. If `auval` is missing, it says the plugin was copied and not validated.

If no plugin package is included and Xcode is installed, the installer can offer to build it. That offer can be skipped. Without Xcode, it says the plugin package has to be supplied separately.

## What is in the download, and what is fetched later

The zip contains SonicLab's source and these installer files. It does not contain Python, Qt, PyTorch, Demucs weights, or a built Audio Unit.

Fetched on the Mac when needed:

| Piece | Licence | Notes |
|---|---|---|
| Python 3.12.10 | PSF Licence | Official universal2 installer. Left on the Mac if SonicLab is removed. |
| PySide6 / Qt 6 | LGPL (or GPL) | Shared libraries inside SonicLab's private folder, not a static build. |
| NumPy, SciPy | BSD | |
| SoundFile, libsndfile | BSD | Brought in by the SoundFile package. |
| SoundDevice, PortAudio | MIT | Brought in by the SoundDevice package. |
| PyQtGraph | MIT | Declared by the project. The editor does not import it today. |
| aubio | GPLv3 | In-process beat tracking. If no ready-made package exists for this Mac, setup continues and that tracker stays off. Installing aubio means a copy of SonicLab that you give to other people is covered by the GPL. Using it on your own Mac is fine. |
| Demucs 4.0.1 | MIT | Code only. |
| PyTorch 2.2.2 | BSD | CPU build used here. MPS is not certified by this installer. |
| `htdemucs_6s` weights | Not bundled | Downloaded by Demucs into Application Support. The Demucs project says the weight licence is not clear. Do not ship those files inside a product until that is settled. |
| Madmom 0.16.1 | Code metadata says BSD. The model files in the package are CC BY-NC-SA 4.0 | Downloaded into a private Python 3.11 folder. Not in the zip. Not linked by the Audio Unit plugins. The model licence does not allow commercial use without permission from the model authors. Do not ship the models, or a product that depends on them, as a closed commercial application. |
| Python 3.11.9 | PSF Licence | Used only for Madmom. NumPy 1.23.5 has no Python 3.12 build, and Madmom calls NumPy names that NumPy 2 removed. |

JUCE is AGPL or a commercial licence and is involved only if you choose to compile the Audio Unit. That plugin is not part of the zip. ONNX Runtime is not installed by this setup.

## Beat analysis

Madmom does not run inside SonicLab's main Python. The app's NumPy is 2.x, and importing Madmom there fails because Madmom still uses `np.int` and `np.float`. Those names are absent from NumPy 1.24 onward.

The working set, checked by running audio through the models, is:

- Python 3.11
- Madmom 0.16.1
- NumPy 1.23.5
- SciPy 1.10.1
- Cython 0.29.37
- mido 1.3.3
- setuptools older than 81, because Madmom imports `pkg_resources`

PyPI has no Madmom wheels, only a source package. NumPy and SciPy have separate Apple Silicon and Intel wheels for Python 3.11, so those are not compiled. Madmom's own audio extensions are compiled on the Mac. If Apple's command-line compiler is missing, setup explains that and opens Apple's installer. Full Xcode is not required. If the compiler is not ready yet, SonicLab still opens and the status line says advanced Madmom analysis is unavailable. Double-click **Update SonicLab.command** after the compiler finishes.

The eight `downbeats_blstm_*.pkl` files ship inside the Madmom package. They are checked for presence and size, then a 16-second click is tracked. An environment that already passes that check is not reinstalled.

On a development machine the same worker is `.venv-madmom/bin/python` plus `tools/madmom_worker.py`. The installed app looks in `~/Library/Application Support/SonicLab/madmom-venv` instead, so deleting the downloaded project folder does not remove beat analysis.

## What has not been tested

The Madmom audio checks in this document were run on this Linux machine. The Mac installer itself was not run on Apple Silicon or Intel hardware. NumPy and SciPy wheels for both Mac processors are published; Madmom still has to compile there. Do not treat this as proof of Mac audio playback, recording, Demucs, MPS, or Audio Unit validation.
