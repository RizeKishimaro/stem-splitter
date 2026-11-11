#!/bin/bash
# Build SonicLab.app.
#
# PyInstaller and Nuitka were considered and not used. Both freeze PySide6
# into one binary, and both regularly drop Qt's cocoa plugin or the audio
# libraries on macOS. The reliable package is a normal application bundle
# whose program starts the private Python environment installed under
# ~/Library/Application Support/SonicLab. That environment does not need
# the downloaded project folder to remain on disk.
#
# This script only assembles the bundle. Install SonicLab.command installs
# the libraries first, then calls this script.

set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MACOS_DIR}/.." && pwd)"
DEST="${1:-${HOME}/Library/Application Support/SonicLab/SonicLab.app}"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "${PROJECT_ROOT}/pyproject.toml" | head -1)"
VERSION="${VERSION:-0.1.0}"

rm -rf "${DEST}"
mkdir -p "${DEST}/Contents/MacOS" "${DEST}/Contents/Resources"

cat >"${DEST}/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>SonicLab</string>
    <key>CFBundleIconFile</key>
    <string>SonicLab</string>
    <key>CFBundleIdentifier</key>
    <string>org.soniclab.app</string>
    <key>CFBundleDisplayName</key>
    <string>SonicLab</string>
    <key>CFBundleName</key>
    <string>SonicLab</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.music</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>SonicLab uses the microphone when you record audio.</string>
</dict>
</plist>
EOF

cat >"${DEST}/Contents/MacOS/SonicLab" <<'EOF'
#!/bin/bash
ROOT="${HOME}/Library/Application Support/SonicLab"
LOG_DIR="${HOME}/Library/Logs/SonicLab"
LOG="${LOG_DIR}/soniclab.log"
PYTHON="${ROOT}/venv/bin/python"
DEMUCS="${ROOT}/demucs-venv/bin/python"
mkdir -p "${LOG_DIR}" "${ROOT}"
{
    printf '---- %s ----\n' "$(date)"
} >>"${LOG}"

explain() {
    osascript - "$1" <<'APPLESCRIPT' >/dev/null 2>&1 || true
on run argv
    display dialog (item 1 of argv) with title "SonicLab" buttons {"OK"} default button "OK" with icon caution
end run
APPLESCRIPT
}

if [[ ! -x "${PYTHON}" ]]; then
    explain "SonicLab is not installed completely. Double-click Install SonicLab.command and leave the window open until it says SonicLab is ready.

Details: ${LOG}"
    printf 'missing %s\n' "${PYTHON}" >>"${LOG}"
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export QT_MAC_WANTS_LAYER=1
export QT_ENABLE_HIGHDPI_SCALING=1
export TORCH_HOME="${ROOT}/models"
if [[ -x "${DEMUCS}" && -f "${ROOT}/demucs-venv/.soniclab-ready" ]]; then
    export SONICLAB_DEMUCS_PYTHON="${DEMUCS}"
fi
if [[ -x "${ROOT}/madmom-venv/bin/python" && -f "${ROOT}/madmom-venv/.soniclab-ready" ]]; then
    export SONICLAB_MADMOM_PYTHON="${ROOT}/madmom-venv/bin/python"
fi
if [[ -f "${ROOT}/madmom_worker.py" ]]; then
    export SONICLAB_MADMOM_WORKER="${ROOT}/madmom_worker.py"
fi
cd "${HOME}"
"${PYTHON}" -m soniclab.main >>"${LOG}" 2>&1
status=$?
if [[ "${status}" -ne 0 ]]; then
    explain "SonicLab closed because of a problem. Details were saved to:

${LOG}"
fi
exit "${status}"
EOF
chmod 755 "${DEST}/Contents/MacOS/SonicLab"

make_icon() {
    local png="${MACOS_DIR}/SonicLab.png"
    local iconset
    [[ -f "${png}" ]] || return 0
    command -v sips >/dev/null 2>&1 || return 0
    command -v iconutil >/dev/null 2>&1 || return 0
    iconset="$(mktemp -d)/SonicLab.iconset"
    mkdir -p "${iconset}"
    sips -z 16 16 "${png}" --out "${iconset}/icon_16x16.png" >/dev/null
    sips -z 32 32 "${png}" --out "${iconset}/icon_16x16@2x.png" >/dev/null
    sips -z 32 32 "${png}" --out "${iconset}/icon_32x32.png" >/dev/null
    sips -z 64 64 "${png}" --out "${iconset}/icon_32x32@2x.png" >/dev/null
    sips -z 128 128 "${png}" --out "${iconset}/icon_128x128.png" >/dev/null
    sips -z 256 256 "${png}" --out "${iconset}/icon_128x128@2x.png" >/dev/null
    sips -z 256 256 "${png}" --out "${iconset}/icon_256x256.png" >/dev/null
    sips -z 512 512 "${png}" --out "${iconset}/icon_256x256@2x.png" >/dev/null
    sips -z 512 512 "${png}" --out "${iconset}/icon_512x512.png" >/dev/null
    sips -z 1024 1024 "${png}" --out "${iconset}/icon_512x512@2x.png" >/dev/null
    iconutil -c icns "${iconset}" -o "${DEST}/Contents/Resources/SonicLab.icns"
}
make_icon || printf '%s\n' "SonicLab will open without a custom icon."

printf '%s\n' "Built ${DEST}"
