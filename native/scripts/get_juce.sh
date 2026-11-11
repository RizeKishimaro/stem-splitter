#!/usr/bin/env bash
# Clone the pinned JUCE release into native/third_party/JUCE (gitignored).
#
# JUCE is dual-licensed AGPLv3 / commercial: read native/LICENSING.md before
# distributing anything built with it. Running this script downloads JUCE
# from GitHub; nothing is bundled in the SonicLab repository.
set -euo pipefail

TAG="8.0.4"
COMMIT="51d11a2be6d5c97ccf12b4e5e827006e19f0555a"
NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${NATIVE_DIR}/third_party/JUCE"

if [[ -d "${DEST}/.git" ]]; then
    echo "JUCE already present at ${DEST}"
else
    mkdir -p "${NATIVE_DIR}/third_party"
    git clone --depth 1 --branch "${TAG}" https://github.com/juce-framework/JUCE.git "${DEST}"
fi

actual="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${actual}" != "${COMMIT}" ]]; then
    echo "error: ${DEST} is at ${actual}, expected JUCE ${TAG} (${COMMIT})." >&2
    echo "Remove ${DEST} and run this script again." >&2
    exit 1
fi
echo "JUCE ${TAG} ready at ${DEST} (${COMMIT})"
