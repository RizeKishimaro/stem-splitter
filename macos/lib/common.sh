# Shared helpers for the SonicLab macOS installer, launcher, updater, and uninstaller.
# User-facing text stays in plain language. Technical detail goes to the log.

set -euo pipefail

SUPPORT="${HOME}/Library/Application Support/SonicLab"
LOG_DIR="${HOME}/Library/Logs/SonicLab"
VENV="${SUPPORT}/venv"
DEMUCS_VENV="${SUPPORT}/demucs-venv"
MADMOM_VENV="${SUPPORT}/madmom-venv"
MODELS="${SUPPORT}/models"
APP_ID="org.soniclab.app"
PYTHON_URL="https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg"
PYTHON_MD5="8f4989592c9412e51fcad70273442c22"
FRAMEWORK_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
PYTHON311_URL="https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg"
PYTHON311_MD5="fa29f456feb6b5c4f52456a8b8ba347b"
FRAMEWORK_PYTHON_311="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
MIN_FREE_GB=8
CURL_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
WARNINGS=()

log() {
    mkdir -p "${LOG_DIR}"
    printf '%s\n' "$*" >>"${LOG_DIR}/install.log"
}

step() {
    printf '\n%s\n' "$*"
    log "$*"
}

note() {
    printf '%s\n' "$*"
    log "$*"
}

warn() {
    WARNINGS+=("$*")
    printf 'Note: %s\n' "$*"
    log "WARNING: $*"
}

dialog() {
    local message="$1"
    local icon="${2:-note}"
    osascript - "$message" "$icon" <<'EOF' >/dev/null 2>&1 || true
on run argv
    set iconName to item 2 of argv
    if iconName is "caution" then
        display dialog (item 1 of argv) with title "SonicLab" buttons {"OK"} default button "OK" with icon caution
    else
        display dialog (item 1 of argv) with title "SonicLab" buttons {"OK"} default button "OK" with icon note
    end if
end run
EOF
}

ask() {
    local prompt="$1"
    local left="$2"
    local right="$3"
    local default="$4"
    local answer
    answer="$(osascript - "$prompt" "$left" "$right" "$default" <<'EOF' 2>/dev/null || true
on run argv
    set choice to button returned of (display dialog (item 1 of argv) buttons {(item 2 of argv), (item 3 of argv)} default button (item 4 of argv) with title "SonicLab" with icon note)
    return choice
end run
EOF
)"
    printf '%s' "${answer}"
}

die() {
    log "ERROR: $*"
    printf '\n%s\n' "$*"
    printf 'A detailed log was saved to:\n%s\n' "${LOG_DIR}/install.log"
    dialog "$*

A detailed log was saved to:
${LOG_DIR}/install.log" "caution"
    exit 1
}

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        printf '%s\n' "This installer only runs on a Mac. It was started on $(uname -s)."
        printf '%s\n' "Copy the SonicLab folder to the Mac, extract it, and double-click Install SonicLab.command there."
        exit 1
    fi
}

mac_version() {
    sw_vers -productVersion
}

mac_arch_name() {
    case "$(uname -m)" in
        arm64) printf 'Apple Silicon' ;;
        x86_64) printf 'Intel' ;;
        *) printf '%s' "$(uname -m)" ;;
    esac
}

check_mac() {
    step "Checking your Mac..."
    local version major arch free_kb free_gb
    version="$(mac_version)"
    major="${version%%.*}"
    arch="$(uname -m)"
    note "macOS ${version}, ${arch} ($(mac_arch_name))."
    if [[ "${major}" -lt 12 ]]; then
        die "SonicLab needs macOS 12 or newer. This Mac is running macOS ${version}."
    fi
    if [[ "${arch}" != "arm64" && "${arch}" != "x86_64" ]]; then
        die "This Mac's processor (${arch}) is not one SonicLab can set up."
    fi
    free_kb="$(df -k "${HOME}" | awk 'NR==2 {print $4}')"
    free_gb="$((free_kb / 1024 / 1024))"
    note "About ${free_gb} GB free on this Mac."
    if [[ "${free_gb}" -lt "${MIN_FREE_GB}" ]]; then
        die "SonicLab needs about ${MIN_FREE_GB} GB free. This Mac has about ${free_gb} GB. Free some space and run the installer again."
    fi
    if ! curl -fsI --max-time 20 -A "${CURL_AGENT}" https://pypi.org >/dev/null 2>&1; then
        die "This Mac does not appear to be online. Connect to the internet and run the installer again."
    fi
    note "Internet connection looks available."
}

python_is_312() {
    local bin="$1"
    [[ -x "${bin}" ]] || return 1
    "${bin}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
}

ensure_python() {
    step "Setting up Python..."
    if python_is_312 "${FRAMEWORK_PYTHON}"; then
        PYTHON_BIN="${FRAMEWORK_PYTHON}"
        note "Python 3.12 is already installed."
        return
    fi
    if command -v python3.12 >/dev/null 2>&1 && python_is_312 "$(command -v python3.12)"; then
        PYTHON_BIN="$(command -v python3.12)"
        note "Python 3.12 is already installed."
        return
    fi
    note "Python 3.12 is not on this Mac yet. SonicLab will download the official installer."
    note "macOS will ask for the password you use to unlock this Mac."
    mkdir -p "${SUPPORT}/cache"
    local pkg="${SUPPORT}/cache/python-3.12.10-macos11.pkg"
    if ! curl -fL --retry 3 --retry-delay 2 -A "${CURL_AGENT}" -o "${pkg}" "${PYTHON_URL}"; then
        die "The Python download did not finish. Check the internet connection and run the installer again."
    fi
    local got
    got="$(md5 -q "${pkg}")"
    if [[ "${got}" != "${PYTHON_MD5}" ]]; then
        rm -f "${pkg}"
        die "The Python download was damaged, so it was not installed. Run the installer again."
    fi
    if ! osascript - "${pkg}" <<'EOF'
on run argv
    do shell script "installer -pkg " & quoted form of (item 1 of argv) & " -target /" with administrator privileges
end run
EOF
    then
        die "Python was not installed. Run the installer again and enter your Mac password when macOS asks."
    fi
    local certificates="/Applications/Python 3.12/Install Certificates.command"
    if [[ -x "${certificates}" ]]; then
        "${certificates}" >>"${LOG_DIR}/install.log" 2>&1 || true
    fi
    if ! python_is_312 "${FRAMEWORK_PYTHON}"; then
        die "Python was downloaded, but SonicLab cannot find Python 3.12. The log has the details."
    fi
    PYTHON_BIN="${FRAMEWORK_PYTHON}"
    note "Python 3.12 is installed."
}

ensure_venv() {
    local directory="$1"
    if [[ -x "${directory}/bin/python" ]] && python_is_312 "${directory}/bin/python"; then
        return
    fi
    rm -rf "${directory}"
    if ! "${PYTHON_BIN}" -m venv "${directory}"; then
        die "SonicLab could not prepare its private folder. The log has the details."
    fi
}

pip_install() {
    local python="$1"
    shift
    "${python}" -m pip install --disable-pip-version-check "$@" >>"${LOG_DIR}/install.log" 2>&1
}

install_application() {
    step "Preparing SonicLab..."
    mkdir -p "${SUPPORT}" "${MODELS}" "${LOG_DIR}"
    ensure_venv "${VENV}"
    if ! "${VENV}/bin/python" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel >>"${LOG_DIR}/install.log" 2>&1; then
        die "SonicLab could not prepare its installer tools. Check the internet connection and run the installer again."
    fi

    step "Installing audio components..."
    # The pip wheels for these packages already include PortAudio and libsndfile.
    # Homebrew is not required.
    if ! pip_install "${VENV}/bin/python" "numpy>=1.26" "scipy>=1.12" "soundfile>=0.12" "sounddevice>=0.4.6"; then
        die "The audio libraries could not be installed. Check the internet connection and run the installer again."
    fi

    step "Installing SonicLab..."
    if ! pip_install "${VENV}/bin/python" "PySide6>=6.7" "pyqtgraph>=0.13.7"; then
        die "SonicLab's window system could not be installed. Check the internet connection and run the installer again."
    fi
    if ! pip_install "${VENV}/bin/python" "${PROJECT_ROOT}" --no-deps; then
        die "SonicLab itself could not be installed. Run the installer again. If it fails a second time, send the log to the person who gave you SonicLab."
    fi
    if ! pip_install "${VENV}/bin/python" "aubio>=0.4.9"; then
        warn "Beat tracking could not be installed automatically. The editor will still open. The log has the details."
    fi
}

install_demucs() {
    step "Preparing AI features..."
    note "Stem splitting uses Demucs 4.0.1 with PyTorch 2.2.2. CPU is always available."
    note "Apple Silicon may report a graphics option named MPS. This installer does not test whether that option is faster or correct."
    ensure_venv "${DEMUCS_VENV}"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    if ! pip_install "${DEMUCS_VENV}/bin/python" --upgrade pip setuptools wheel; then
        warn "The stem-splitting feature could not be prepared. SonicLab itself is still being installed."
        return
    fi
    # Installed before Demucs, with --no-deps on Demucs, so pip cannot swap in
    # another PyTorch. Intel Macs have no newer official PyTorch wheel, and
    # Demucs 4.0.1 writes audio with the torchaudio version paired to 2.2.2.
    if ! pip_install "${DEMUCS_VENV}/bin/python" -c "${MACOS_DIR}/lib/demucs-constraints.txt" --only-binary=:all: "numpy<2" "torch==2.2.2" "torchaudio==2.2.2"; then
        warn "PyTorch 2.2.2 has no ready-made installer for this Mac, so stem splitting was not installed. SonicLab itself will still open."
        return
    fi
    if ! pip_install "${DEMUCS_VENV}/bin/python" --no-deps "demucs==4.0.1"; then
        warn "Demucs could not be installed. Stem splitting will be unavailable. SonicLab itself will still open."
        return
    fi
    if ! pip_install "${DEMUCS_VENV}/bin/python" -c "${MACOS_DIR}/lib/demucs-constraints.txt" "dora-search" "einops" "julius>=0.2.3" "lameenc>=1.2" "openunmix" "pyyaml" "tqdm" "soundfile>=0.12"; then
        warn "A small piece of the stem splitter could not be installed. Stem splitting may be unavailable. SonicLab itself will still open."
        return
    fi
    if ! "${DEMUCS_VENV}/bin/python" - <<'EOF' >>"${LOG_DIR}/install.log" 2>&1
import demucs, torch, torchaudio
assert demucs.__version__ == "4.0.1", demucs.__version__
assert torch.__version__.split("+")[0] == "2.2.2", torch.__version__
assert torchaudio.__version__.split("+")[0] == "2.2.2", torchaudio.__version__
mps = getattr(torch.backends, "mps", None)
print("mps_reported", bool(mps is not None and mps.is_available()))
EOF
    then
        warn "The stem splitter installed, but the versions were not the tested pair. It will not be used until the installer is run again."
        rm -rf "${DEMUCS_VENV}"
        return
    fi
    log "--- demucs audio decode ---"
    if ! "${DEMUCS_VENV}/bin/python" "${MACOS_DIR}/lib/verify_decode.py" >>"${LOG_DIR}/install.log" 2>&1; then
        warn "Stem splitting was installed, but it could not read a test recording. It will not be used until that is repaired. SonicLab will still open."
        return
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        note "FFmpeg is not installed. WAV, FLAC, AIFF, and MP3 can still be split. Other formats need to be converted to WAV first."
    fi
    touch "${DEMUCS_VENV}/.soniclab-ready"

    step "Downloading Demucs model..."
    export TORCH_HOME="${MODELS}"
    if ! "${DEMUCS_VENV}/bin/python" "${MACOS_DIR}/lib/fetch_model.py" htdemucs_6s 2>&1 | tee -a "${LOG_DIR}/install.log"; then
        warn "The stem-splitting model was not downloaded. SonicLab will try again the first time you split a song, if this Mac is online."
    fi
}

is_soniclab_app() {
    local bundle="$1"
    [[ -d "${bundle}" ]] || return 1
    [[ -f "${bundle}/Contents/Info.plist" ]] || return 1
    grep -q "${APP_ID}" "${bundle}/Contents/Info.plist"
}

place_application() {
    local built="$1"
    local target="/Applications/SonicLab.app"
    if [[ -e "${target}" && ! -L "${target}" ]] && ! is_soniclab_app "${target}"; then
        target="${HOME}/Applications/SonicLab.app"
    fi
    mkdir -p "$(dirname "${target}")"
    if [[ -d "${target}" ]] && is_soniclab_app "${target}"; then
        rm -rf "${target}"
    fi
    if [[ -w "$(dirname "${target}")" ]]; then
        ditto "${built}" "${target}"
    else
        note "macOS will ask for your password to put SonicLab in Applications."
        if ! osascript - "${built}" "${target}" <<'EOF'
on run argv
    do shell script "rm -rf " & quoted form of (item 2 of argv) & " && ditto " & quoted form of (item 1 of argv) & " " & quoted form of (item 2 of argv) with administrator privileges
end run
EOF
        then
            target="${HOME}/Applications/SonicLab.app"
            mkdir -p "${HOME}/Applications"
            rm -rf "${target}"
            ditto "${built}" "${target}"
            note "SonicLab was put in your personal Applications folder because the main Applications folder was not available."
        fi
    fi
    if ! is_soniclab_app "${target}"; then
        die "SonicLab was installed, but the application icon could not be created."
    fi
    INSTALLED_APP="${target}"
    local desktop="${HOME}/Desktop/SonicLab.app"
    if [[ -L "${desktop}" || ! -e "${desktop}" ]]; then
        ln -sfn "${target}" "${desktop}"
        note "A SonicLab icon was added to the Desktop."
    elif is_soniclab_app "${desktop}"; then
        rm -rf "${desktop}"
        ln -sfn "${target}" "${desktop}"
    else
        warn "The Desktop already has something named SonicLab.app, so a new shortcut was not added."
    fi
}

verify_installed_plugin() {
    local dest="${HOME}/Library/Audio/Plug-Ins/Components/SonicLab Beat Analyzer.component"
    if [[ ! -d "${dest}" ]]; then
        warn "The plugin build finished, but the plugin is not in the Audio Units folder."
        return 1
    fi
    if ! codesign --verify --verbose=2 "${dest}" >>"${LOG_DIR}/install.log" 2>&1; then
        warn "The plugin file is in the Audio Units folder, but macOS did not accept its signature. Logic may refuse it."
        return 1
    fi
    if ! command -v auval >/dev/null 2>&1; then
        note "The plugin was copied. The auval checker is not on this Mac, so the plugin was not validated."
        return 0
    fi
    if auval -v aufx Btan Snlb >>"${LOG_DIR}/install.log" 2>&1; then
        note "The Beat Analyzer plugin is installed, and auval accepted it."
        return 0
    fi
    warn "The plugin was copied, but auval did not accept it. Logic may not load it. See the log."
    return 1
}

install_one_plugin() {
    local source="$1"
    local dest="${HOME}/Library/Audio/Plug-Ins/Components/SonicLab Beat Analyzer.component"
    mkdir -p "${HOME}/Library/Audio/Plug-Ins/Components"
    if [[ -d "${dest}" ]]; then
        local choice
        choice="$(ask "A Beat Analyzer plugin is already installed. Replace it?" "Keep existing" "Replace" "Keep existing")"
        if [[ "${choice}" != "Replace" ]]; then
            note "The existing plugin was left in place."
            return 0
        fi
        rm -rf "${dest}"
    fi
    ditto "${source}" "${dest}"
    if [[ ! -d "${dest}" ]]; then
        warn "The plugin could not be copied into the Audio Units folder."
        return 1
    fi
    if ! codesign --verify --verbose=2 "${dest}" >>"${LOG_DIR}/install.log" 2>&1; then
        warn "The plugin file is in the Audio Units folder, but macOS did not accept its signature. Logic may refuse it."
        return 1
    fi
    if ! command -v auval >/dev/null 2>&1; then
        note "The plugin was copied. The auval checker is not on this Mac, so the plugin was not validated."
        return 0
    fi
    if auval -v aufx Btan Snlb >>"${LOG_DIR}/install.log" 2>&1; then
        note "The Beat Analyzer plugin is installed, and auval accepted it."
        return 0
    fi
    warn "The plugin was copied, but auval did not accept it. Logic may not load it. See the log."
    return 1
}

offer_plugins() {
    step "Checking audio plugins..."
    local found=()
    local candidate
    shopt -s nullglob
    for candidate in "${MACOS_DIR}/plugins/"*.component; do
        found+=("${candidate}")
    done
    shopt -u nullglob
    candidate="${PROJECT_ROOT}/native/build/macos-xcode/plugins/beat_analyzer/juce/SonicLabBeatAnalyzer_artefacts/Release/AU/SonicLab Beat Analyzer.component"
    if [[ -d "${candidate}" ]]; then
        found+=("${candidate}")
    fi
    if [[ ${#found[@]} -gt 0 ]]; then
        install_one_plugin "${found[0]}" || true
        return
    fi
    if xcodebuild -version >/dev/null 2>&1; then
        local choice
        choice="$(ask "No plugin package was included. Xcode is installed, so the Beat Analyzer plugin can be built now. This is optional and can take a long time." "Not now" "Build plugin" "Not now")"
        if [[ "${choice}" == "Build plugin" ]]; then
        if "${PROJECT_ROOT}/native/scripts/build_au_macos.sh" --install >>"${LOG_DIR}/install.log" 2>&1; then
            verify_installed_plugin || true
        else
                warn "The plugin build did not finish. SonicLab itself is still installed. The log has the details."
            fi
        else
            note "The plugin package was not included. SonicLab does not need it to open. A plugin package can be added later."
        fi
        return
    fi
    note "The Beat Analyzer plugin is not in this download, and Xcode is not installed, so it was not built. SonicLab itself does not need the plugin."
}

verify_installation() {
    step "Verifying installation..."
    local report
    if ! report="$("${VENV}/bin/python" "${MACOS_DIR}/lib/verify_app.py" 2>>"${LOG_DIR}/install.log")"; then
        log "${report}"
        die "SonicLab was copied, but it could not start its libraries. Run the installer again. If it fails a second time, send the log to the person who gave you SonicLab."
    fi
    log "${report}"
    if [[ ! -x "${INSTALLED_APP}/Contents/MacOS/SonicLab" ]]; then
        die "The SonicLab application is missing its startup program."
    fi
    local devices
    devices="$(printf '%s\n' "${report}" | sed -n 's/.*"audio_devices": \([0-9][0-9]*\).*/\1/p')"
    if [[ -n "${devices}" ]]; then
        note "Audio devices reported by this Mac: ${devices}."
    else
        warn "SonicLab is installed, but no audio device list was returned. Check System Settings, Sound, if recording or playback does not appear."
    fi
    if [[ "${report}" != *'"aubio": true'* ]]; then
        warn "Beat tracking is not available yet. The rest of SonicLab is installed."
    fi
    if [[ -f "${DEMUCS_VENV}/.soniclab-ready" ]]; then
        note "Stem splitting is installed. It runs on the CPU. MPS was not measured by this installer."
    fi
    {
        printf 'version=0.1.0\n'
        printf 'date=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'macos=%s\n' "$(mac_version)"
        printf 'arch=%s\n' "$(uname -m)"
        printf 'app=%s\n' "${INSTALLED_APP}"
    } >"${SUPPORT}/install-receipt.txt"
}

python_is_311() {
    local bin="$1"
    [[ -x "${bin}" ]] || return 1
    "${bin}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
}

ensure_python_311() {
    if python_is_311 "${FRAMEWORK_PYTHON_311}"; then
        PYTHON311_BIN="${FRAMEWORK_PYTHON_311}"
        return 0
    fi
    if command -v python3.11 >/dev/null 2>&1 && python_is_311 "$(command -v python3.11)"; then
        PYTHON311_BIN="$(command -v python3.11)"
        return 0
    fi
    note "Beat analysis needs Python 3.11. SonicLab's own window uses Python 3.12, and the two cannot share libraries."
    note "macOS will ask for the password you use to unlock this Mac."
    mkdir -p "${SUPPORT}/cache"
    local pkg="${SUPPORT}/cache/python-3.11.9-macos11.pkg"
    if ! curl -fL --retry 3 --retry-delay 2 -A "${CURL_AGENT}" -o "${pkg}" "${PYTHON311_URL}"; then
        return 1
    fi
    local got
    got="$(md5 -q "${pkg}")"
    if [[ "${got}" != "${PYTHON311_MD5}" ]]; then
        rm -f "${pkg}"
        return 1
    fi
    if ! osascript - "${pkg}" <<'EOF'
on run argv
    do shell script "installer -pkg " & quoted form of (item 1 of argv) & " -target /" with administrator privileges
end run
EOF
    then
        return 1
    fi
    local certificates="/Applications/Python 3.11/Install Certificates.command"
    if [[ -x "${certificates}" ]]; then
        "${certificates}" >>"${LOG_DIR}/install.log" 2>&1 || true
    fi
    if ! python_is_311 "${FRAMEWORK_PYTHON_311}"; then
        return 1
    fi
    PYTHON311_BIN="${FRAMEWORK_PYTHON_311}"
    return 0
}

install_beat_analysis() {
    step "Preparing beat analysis..."
    log "--- beat analysis ---"
    note "Beat analysis uses madmom-infer. Its pretrained weights are non-commercial and are downloaded into your cache, not shipped with SonicLab."
    if [[ -f "${PROJECT_ROOT}/tools/madmom_infer_worker.py" ]]; then
        cp "${PROJECT_ROOT}/tools/madmom_infer_worker.py" "${SUPPORT}/madmom_infer_worker.py"
    fi
    if [[ "${SONICLAB_MADMOM_LEGACY:-}" == "1" ]]; then
        note "SONICLAB_MADMOM_LEGACY=1, so the older Madmom 0.16 setup is used."
        install_madmom
        return
    fi
    if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c "import madmom_infer" >/dev/null 2>&1; then
        if "${VENV}/bin/python" "${MACOS_DIR}/lib/verify_infer.py" >>"${LOG_DIR}/install.log" 2>&1; then
            touch "${SUPPORT}/.soniclab-infer-ready"
            note "Beat analysis is already working, so its libraries were left in place."
            return
        fi
        note "The existing beat-analysis check failed. SonicLab will reinstall madmom-infer."
        rm -f "${SUPPORT}/.soniclab-infer-ready"
    fi
    if ! pip_install "${VENV}/bin/python" "madmom-infer @ git+https://github.com/openmirlab/madmom-infer.git@bffead9be61857fd44b2e0d3a2510d9d94c8f2d2"; then
        warn "madmom-infer could not be installed. SonicLab will try the older Madmom setup. SonicLab will still open."
        install_madmom
        return
    fi
    if ! "${VENV}/bin/python" "${MACOS_DIR}/lib/verify_infer.py" >>"${LOG_DIR}/install.log" 2>&1; then
        rm -f "${SUPPORT}/.soniclab-infer-ready"
        warn "madmom-infer was installed, but it could not track a test recording. SonicLab will try the older Madmom setup."
        install_madmom
        return
    fi
    touch "${SUPPORT}/.soniclab-infer-ready"
    note "Beat analysis is ready. A test click was tracked at about 120 BPM."
}

install_madmom() {
    step "Preparing beat analysis..."
    log "--- madmom ---"
    note "Madmom stays in its own Python 3.11 folder. If this step fails, SonicLab still opens."
    if [[ -f "${PROJECT_ROOT}/tools/madmom_worker.py" ]]; then
        cp "${PROJECT_ROOT}/tools/madmom_worker.py" "${SUPPORT}/madmom_worker.py"
    fi
    if [[ -x "${MADMOM_VENV}/bin/python" && -f "${MADMOM_VENV}/.soniclab-ready" ]]; then
        if "${MADMOM_VENV}/bin/python" "${MACOS_DIR}/lib/verify_madmom.py" >>"${LOG_DIR}/install.log" 2>&1; then
            note "Beat analysis is already working, so its libraries were left in place."
            return
        fi
        note "The existing beat-analysis setup failed its check. SonicLab will repair it."
        rm -f "${MADMOM_VENV}/.soniclab-ready"
    fi
    if ! ensure_python_311; then
        warn "Advanced Madmom analysis is unavailable until Python 3.11 can be installed. SonicLab will still open."
        return
    fi
    if ! xcrun --find clang >/dev/null 2>&1; then
        note "Madmom publishes no Mac package. Its audio code has to be compiled on this Mac."
        note "That uses Apple's command-line tools. The full Xcode application is not required."
        note "If a window opens, click Install. When it finishes, double-click Update SonicLab.command."
        xcode-select --install >/dev/null 2>&1 || true
        local attempt
        for attempt in 1 2 3 4 5 6; do
            if xcrun --find clang >/dev/null 2>&1; then
                break
            fi
            sleep 10
        done
    fi
    if ! xcrun --find clang >/dev/null 2>&1; then
        warn "Advanced Madmom analysis is unavailable because the command-line compiler is not ready. SonicLab will still open. After Apple's installer finishes, double-click Update SonicLab.command."
        return
    fi
    if [[ ! -x "${MADMOM_VENV}/bin/python" ]] || ! python_is_311 "${MADMOM_VENV}/bin/python"; then
        rm -rf "${MADMOM_VENV}"
        if ! "${PYTHON311_BIN}" -m venv "${MADMOM_VENV}"; then
            warn "Advanced Madmom analysis is unavailable because its private folder could not be created. SonicLab will still open."
            return
        fi
    fi
    if ! "${MADMOM_VENV}/bin/python" -m pip install --disable-pip-version-check --upgrade "pip<25" "setuptools>=68,<81" wheel >>"${LOG_DIR}/install.log" 2>&1; then
        warn "Advanced Madmom analysis is unavailable because its installer tools could not be prepared. SonicLab will still open."
        return
    fi
    if ! pip_install "${MADMOM_VENV}/bin/python" -c "${MACOS_DIR}/lib/madmom-constraints.txt" --only-binary=:all: "setuptools>=68,<81" "cython==0.29.37" "numpy==1.23.5" "scipy==1.10.1" "mido==1.3.3"; then
        warn "Advanced Madmom analysis is unavailable because a compatible NumPy could not be installed for this processor. SonicLab will still open."
        return
    fi
    if ! pip_install "${MADMOM_VENV}/bin/python" -c "${MACOS_DIR}/lib/madmom-constraints.txt" --no-build-isolation "madmom==0.16.1"; then
        warn "Advanced Madmom analysis is unavailable because Madmom could not be compiled. SonicLab will still open. The log has the compiler output."
        return
    fi
    if ! "${MADMOM_VENV}/bin/python" "${MACOS_DIR}/lib/verify_madmom.py" >>"${LOG_DIR}/install.log" 2>&1; then
        rm -f "${MADMOM_VENV}/.soniclab-ready"
        warn "Madmom was installed, but it could not track a test recording. Advanced beat analysis is unavailable until that is repaired. SonicLab will still open."
        return
    fi
    touch "${MADMOM_VENV}/.soniclab-ready"
    note "Beat analysis is ready. A test click was tracked at about 120 BPM."
}

soniclab_install() {
    require_macos
    : "${MACOS_DIR:?}"
    : "${PROJECT_ROOT:?}"
    mkdir -p "${LOG_DIR}" "${SUPPORT}"
    log "---- SonicLab install $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
    printf '%s\n' "SonicLab setup"
    printf '%s\n' "Leave this window open. You do not need to type any commands."
    if [[ -f "${SUPPORT}/install-receipt.txt" ]]; then
        note "SonicLab is already installed. This will update it and keep your projects, preferences, and downloaded models."
    fi
    check_mac
    ensure_python
    install_application
    install_demucs
    install_beat_analysis
    if ! bash "${MACOS_DIR}/build_app.sh" "${SUPPORT}/SonicLab.app" >>"${LOG_DIR}/install.log" 2>&1; then
        die "The SonicLab application could not be created. The log has the details."
    fi
    place_application "${SUPPORT}/SonicLab.app"
    verify_installation
    offer_plugins
    step "SonicLab is ready!"
    local extra=""
    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        extra="

One part needs attention:
${WARNINGS[0]}"
    fi
    note "Open SonicLab from Applications or the Desktop. You can close this window."
    dialog "SonicLab is ready.

Open it from Applications or the Desktop.${extra}" "note"
}

soniclab_launch() {
    require_macos
    mkdir -p "${LOG_DIR}"
    local app=""
    local candidate
    for candidate in "/Applications/SonicLab.app" "${HOME}/Applications/SonicLab.app" "${HOME}/Desktop/SonicLab.app" "${SUPPORT}/SonicLab.app"; do
        if is_soniclab_app "${candidate}"; then
            app="${candidate}"
            break
        fi
    done
    if [[ -z "${app}" ]]; then
        dialog "SonicLab is not installed yet. Double-click Install SonicLab.command and leave the window open until it finishes." "caution"
        printf '%s\n' "SonicLab is not installed yet. Double-click Install SonicLab.command first."
        exit 1
    fi
    if ! open "${app}"; then
        dialog "SonicLab did not open. Details were saved to ${LOG_DIR}/soniclab.log" "caution"
        printf '%s\n' "SonicLab did not open. Details were saved to ${LOG_DIR}/soniclab.log"
        exit 1
    fi
    printf '%s\n' "SonicLab is opening. You can close this window."
    printf '%s\n' "If it does not appear, see ${LOG_DIR}/soniclab.log"
}

remove_bundle() {
    local bundle="$1"
    if [[ -L "${bundle}" ]]; then
        local target
        target="$(readlink "${bundle}")"
        case "${target}" in
            *SonicLab.app) rm -f "${bundle}"; note "Removed ${bundle}" ;;
        esac
        return
    fi
    if ! is_soniclab_app "${bundle}"; then
        return
    fi
    if [[ -w "$(dirname "${bundle}")" ]]; then
        rm -rf "${bundle}"
    else
        osascript - "${bundle}" <<'EOF' || die "SonicLab could not be removed from Applications. The log has the details."
on run argv
    do shell script "rm -rf " & quoted form of (item 1 of argv) with administrator privileges
end run
EOF
    fi
    note "Removed ${bundle}"
}

soniclab_uninstall() {
    require_macos
    mkdir -p "${LOG_DIR}"
    local choice
    choice="$(ask "This removes SonicLab from Applications and the Desktop.

Your projects and your own audio files are not deleted.

The private beat-analysis library is removed with the program. Its model files are not commercial weights and are not kept separately.

Downloaded stem-splitting models are kept unless you say so on the next screen.

Preferences are kept, so a later install remembers your settings.

Remove SonicLab?" "Cancel" "Remove" "Cancel")"
    if [[ "${choice}" != "Remove" ]]; then
        note "Uninstall cancelled. Nothing was removed."
        exit 0
    fi
    remove_bundle "/Applications/SonicLab.app"
    remove_bundle "${HOME}/Applications/SonicLab.app"
    remove_bundle "${HOME}/Desktop/SonicLab.app"
    remove_bundle "${SUPPORT}/SonicLab.app"
    rm -rf "${VENV}" "${DEMUCS_VENV}" "${MADMOM_VENV}" "${SUPPORT}/madmom_worker.py" "${SUPPORT}/madmom_infer_worker.py" "${SUPPORT}/.soniclab-infer-ready" "${SUPPORT}/cache" "${HOME}/.cache/madmom_infer"
    if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        rm -rf "${XDG_CACHE_HOME}/madmom_infer"
    fi
    note "Removed SonicLab's installed program files."

    choice="$(ask "Also remove downloaded AI models and stem files that SonicLab stored in its own library folder?

Your projects stay where you saved them. This does not search your Music folder or other disks." "Keep" "Remove models" "Keep")"
    if [[ "${choice}" == "Remove models" ]]; then
        rm -rf "${MODELS}" "${SUPPORT}/stems"
        note "Removed SonicLab's downloaded models and stored stem files."
    else
        note "Downloaded models and stored stem files were kept."
    fi

    local plugin="${HOME}/Library/Audio/Plug-Ins/Components/SonicLab Beat Analyzer.component"
    if [[ -d "${plugin}" ]]; then
        choice="$(ask "Also remove the Beat Analyzer Audio Unit plugin?" "Keep" "Remove plugin" "Keep")"
        if [[ "${choice}" == "Remove plugin" ]]; then
            rm -rf "${plugin}"
            note "Removed the Beat Analyzer plugin."
        else
            note "The plugin was kept."
        fi
    fi
    rmdir "${SUPPORT}" 2>/dev/null || true
    note "Python, if this Mac had it installed for SonicLab, was left in place."
    note "Uninstall finished."
    dialog "SonicLab was removed.

Your projects were not deleted." "note"
}
