#!/usr/bin/env bash
# Build, install and validate the SonicLab Beat Analyzer AUv2 on macOS.
#
#   native/scripts/build_au_macos.sh            build + ad-hoc sign
#   native/scripts/build_au_macos.sh --install  also copy to ~/Library/Audio/Plug-Ins/Components,
#                                               refresh the AU cache and run auval
#
# Needs full Xcode (not only Command Line Tools), CMake 3.25+ and git.
# Ad-hoc signing ("-") is enough to load the plugin on this Mac; signing for
# other Macs needs a Developer ID and notarization (native/LICENSING.md).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Audio Units build only on macOS. On Linux use the linux-juce preset (Standalone)." >&2
    exit 1
fi

INSTALL=0
for arg in "$@"; do
    case "${arg}" in
        --install) INSTALL=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "error: unknown option ${arg}" >&2; exit 2 ;;
    esac
done

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="Release"
COMPONENT_NAME="SonicLab Beat Analyzer.component"
ARTEFACT="${NATIVE_DIR}/build/macos-xcode/plugins/beat_analyzer/juce/SonicLabBeatAnalyzer_artefacts/${CONFIG}/AU/${COMPONENT_NAME}"
DEST_DIR="${HOME}/Library/Audio/Plug-Ins/Components"

if ! xcodebuild -version >/dev/null 2>&1; then
    echo "error: xcodebuild not usable. Install Xcode, open it once, then run:" >&2
    echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
fi

"${NATIVE_DIR}/scripts/get_juce.sh"

cd "${NATIVE_DIR}"
cmake --preset macos-xcode
cmake --build --preset macos-xcode --target SonicLabBeatAnalyzer_AU

if [[ ! -d "${ARTEFACT}" ]]; then
    echo "error: build finished but ${ARTEFACT} is missing" >&2
    exit 1
fi

# Ad-hoc signature so Apple Silicon and auval accept the bundle on this Mac.
codesign --force --deep --sign - "${ARTEFACT}"
codesign --verify --verbose=2 "${ARTEFACT}"
echo "Built: ${ARTEFACT}"
lipo -info "${ARTEFACT}/Contents/MacOS/SonicLab Beat Analyzer" || true

if [[ "${INSTALL}" -eq 1 ]]; then
    mkdir -p "${DEST_DIR}"
    rm -rf "${DEST_DIR:?}/${COMPONENT_NAME}"
    cp -R "${ARTEFACT}" "${DEST_DIR}/"
    echo "Installed: ${DEST_DIR}/${COMPONENT_NAME}"
    # Make the system re-read installed components.
    killall -9 AudioComponentRegistrar 2>/dev/null || true
    sleep 2
    auval -a | grep -i "Snlb" || echo "warning: auval -a does not list Snlb yet; log out and in, then retry" >&2
    auval -v aufx Btan Snlb
fi
