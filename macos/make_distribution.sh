#!/bin/bash
# Build dist/SonicLab-macOS.zip from this checkout.
# The archive contains source and the installer. It does not contain model
# weights, virtual environments, or build output.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
DEST="${ROOT}/dist"
mkdir -p "${DEST}"
chmod 755 \
    "${ROOT}/macos/Install SonicLab.command" \
    "${ROOT}/macos/Launch SonicLab.command" \
    "${ROOT}/macos/Update SonicLab.command" \
    "${ROOT}/macos/Uninstall SonicLab.command" \
    "${ROOT}/macos/build_app.sh" \
    "${ROOT}/macos/make_distribution.sh" \
    "${ROOT}/macos/lib/make_icon.py" \
    "${ROOT}/macos/lib/fetch_model.py" \
    "${ROOT}/macos/lib/verify_app.py" \
    "${ROOT}/macos/lib/verify_madmom.py" \
    "${ROOT}/macos/lib/verify_infer.py" \
    "${ROOT}/macos/lib/verify_decode.py"

rsync -a \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.venv-*/' \
    --exclude '.venv*/' \
    --exclude '**/__pycache__/' \
    --exclude '**/*.pyc' \
    --exclude 'native/build/' \
    --exclude 'native/third_party/' \
    --exclude 'dist/' \
    --exclude '.cursor/' \
    --exclude '**/*.egg-info/' \
    "${ROOT}/" "${STAGE}/SonicLab/"

python3 -m py_compile "${ROOT}/macos/lib/fetch_model.py" "${ROOT}/macos/lib/verify_app.py" "${ROOT}/macos/lib/verify_madmom.py" "${ROOT}/macos/lib/verify_infer.py" "${ROOT}/macos/lib/verify_decode.py"
find "${STAGE}" -type d -name '__pycache__' -exec rm -rf {} + || true
(
    cd "${STAGE}"
    rm -f "${DEST}/SonicLab-macOS.zip"
    zip -r -X -y "${DEST}/SonicLab-macOS.zip" "SonicLab" >/dev/null
)
rm -rf "${STAGE}"
printf '%s\n' "Wrote ${DEST}/SonicLab-macOS.zip"
