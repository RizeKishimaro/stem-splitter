#!/bin/bash
cd "$(dirname "$0")" || exit 1
MACOS_DIR="$(pwd)"
PROJECT_ROOT="$(cd "${MACOS_DIR}/.." && pwd)"
export MACOS_DIR PROJECT_ROOT
# shellcheck source=lib/common.sh
source "${MACOS_DIR}/lib/common.sh"
soniclab_uninstall
