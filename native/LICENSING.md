# Licensing notes for SonicLab native work

This is an engineering summary, not legal advice. License terms change;
check the current text from each vendor before any release, and get a
lawyer's view before selling anything. **No configuration below should be
assumed to allow commercial or closed-source distribution until that check
is done.**

## Summary

| Component | License options | Matters when |
|---|---|---|
| JUCE 8 | AGPLv3 **or** JUCE commercial licence (EULA tiers) | Any binary that links JUCE is distributed |
| VST3 SDK (bundled in JUCE) | Depends on the SDK version JUCE bundles; see below. "VST" name/logo are Steinberg trademarks | A VST3 plugin or VST3 host is distributed |
| Apple AU | No SDK licence fee; needs Xcode, Apple Developer Program for signing/notarization | An AU is given to anyone else's Mac |
| aubio (Python dep, already used) | GPLv3 or later | SonicLab itself is distributed |
| Madmom (worker venv, already used) | Code BSD; **pretrained models CC BY-NC-SA** | Any commercial use of Madmom-based results or a port of its models |
| PySide6 | LGPLv3 / GPLv2 / GPLv3 | SonicLab is distributed (LGPL: keep Qt dynamically linked and replaceable) |

Nothing in `native/` bundles JUCE or any SDK. JUCE is obtained only when
you run `scripts/get_juce.sh`, set `SONICLAB_JUCE_DIR`, or opt in to
`SONICLAB_FETCH_JUCE`. `.gitignore` excludes `build/` and `third_party/`,
so a fetched or checked-out JUCE is not committed by accident.

**Pinned version (verified Phase 4):** JUCE tag `8.0.4`, commit
`51d11a2be6d5c97ccf12b4e5e827006e19f0555a` (2024-11-18). Its `LICENSE.md`
states the modules are dual-licensed: AGPLv3, or the JUCE 8 licence/EULA.
On the development machine a copy is cloned in `native/third_party/JUCE`
for local builds under AGPLv3. It is not committed and not distributed.

## JUCE

JUCE 8 is dual-licensed:

* **AGPLv3.** Free. Every program you *distribute* that links JUCE must be
  released under AGPLv3-compatible terms with complete corresponding source.
  AGPL also covers users interacting with the program over a network.
  For SonicLab this means:
  * A native extension module linked with JUCE and loaded by the SonicLab
    Python app makes the distributed whole a combined work, so SonicLab
    would have to be distributed as AGPLv3 (its Python code included).
  * The Beat Analyzer plugin binary would have to ship with its source
    under AGPLv3.
  * Using JUCE privately (builds you never give to anyone) carries no
    distribution obligations.
* **Commercial licence (JUCE 8 EULA).** Paid tiers (and historically a
  free "Starter" tier below a revenue cap) allow closed-source
  distribution. Tier names, revenue caps and per-seat terms change; read
  the current EULA at juce.com before relying on one.

Practical guidance:
* Internal development and testing: AGPLv3 is fine.
* Shipping the plugin or a JUCE-linked SonicLab to customers without
  releasing source: needs a commercial JUCE licence for every developer
  seat the EULA requires.
* App Store distribution of AGPL/GPL software is widely considered
  incompatible with App Store terms. Plan on a commercial JUCE licence if
  an AUv3 is ever shipped through the Mac App Store.
* The AU build itself pulls in no third-party SDK. The AU wrapper uses only
  Apple's system frameworks.
* Dropped-file analysis decodes audio with JUCE's `juce_audio_formats`.
  That compiles in JUCE's bundled **FLAC** and **Ogg Vorbis** codecs, both
  BSD-licensed (see `modules/juce_audio_formats/codecs/` and JUCE's
  `LICENSE.md`). BSD only asks that their copyright notices ship with
  binaries. On macOS, MP3/M4A decoding comes from Apple's Core Audio, not
  bundled code. JUCE's own MP3 decoder (`JUCE_USE_MP3AUDIOFORMAT`) stays
  off.
* `JUCE_DISPLAY_SPLASH_SCREEN=0` is set in the plugin target. Whether a
  splash screen is required depends on the JUCE version and licence in
  use; check the EULA for the pinned tag before release.

The pinned tag is `SONICLAB_JUCE_TAG` (default `8.0.4`) in
`native/CMakeLists.txt`. Bump it deliberately and re-read the licence text
in that tag's `LICENSE.md` when you do.

## Steinberg VST3 SDK

JUCE carries a copy of the VST3 SDK and uses it whenever a VST3 plugin is
built (`SONICLAB_BUILD_VST3=ON`) or VST3 hosting is enabled.

* **VST3 SDK up to 3.7.x** is dual-licensed: **GPLv3**, or Steinberg's
  **proprietary VST3 licence agreement** (free, signed with Steinberg).
  Closed-source VST3 plugins or hosts need the signed agreement.
* **VST3 SDK 3.8 and later** was relicensed by Steinberg under the **MIT
  licence** (announced October 2025).
* **What we actually use:** JUCE 8.0.4 bundles **VST3 SDK 3.7.12**
  (`modules/juce_audio_processors/format_types/VST3_SDK`, checked in
  Phase 4). That copy carries `LICENSE.txt` (dual GPLv3 / Steinberg
  agreement), `VST3_License_Agreement.pdf` and `VST3_Usage_Guidelines.pdf`.
  The 3.7.12 dual licence therefore applies to any VST3 built from this
  pin, until JUCE is bumped to a release that bundles SDK 3.8+.
* **Trademark.** Independently of the code licence, the "VST" name and the
  VST logo are Steinberg trademarks. Using them on a product, website or
  installer requires following Steinberg's VST usage guidelines / trademark
  terms (logo placement, wording, attribution). The MIT relicensing does
  not grant trademark rights.
* VST2 is not used anywhere (`JUCE_VST3_CAN_REPLACE_VST2=0`); the VST2 SDK
  is no longer licensed to new developers.

Because of this, **VST3 is off by default** on every platform. Turn it on
for local testing freely; check the above before distributing a VST3.

## Apple Audio Units

There is no separate AU SDK licence: the AudioUnit/AudioToolbox frameworks
come with Xcode. Distribution has practical requirements instead:

* **Xcode** (full Xcode, not only Command Line Tools) for the Xcode
  generator, codesigning and notarization tools.
* **Apple Developer Program** membership for a **Developer ID Application**
  certificate (code) and a **Developer ID Installer** certificate (`.pkg`).
* **Codesigning** with the hardened runtime:
  `codesign --force --options runtime --timestamp --sign "Developer ID Application: <Team>" "SonicLab Beat Analyzer.component"`.
* **Notarization**: `xcrun notarytool submit <zip or pkg> --keychain-profile <profile> --wait`,
  then `xcrun stapler staple` on the `.pkg` (or the component).
  Without this, Gatekeeper blocks the plugin on other Macs, and Logic may
  refuse to load it.
* **Validation**: `auval -v aufx Btan Snlb` must pass. Logic Pro runs the
  same validation in its Plug-in Manager and disables plugins that fail.
* **AUv2 vs AUv3.**
  * AUv2 (`.component`, built here) installs into
    `~/Library/Audio/Plug-Ins/Components` or `/Library/Audio/Plug-Ins/Components`,
    runs in the host's process, and is what Logic Pro, Ableton Live and most
    macOS hosts load. Distribute it as a signed, notarized `.pkg`.
  * AUv3 is an **app extension** (`.appex`) inside a containing app. It must
    be sandboxed, is registered when the containing app is first launched,
    can run out-of-process, and is required for iOS/iPadOS and for Mac App
    Store distribution. JUCE builds it with `FORMATS AUv3` and the Xcode
    generator; it needs an App Group and provisioning profile. Not set up
    in Phase 1.

## Already in SonicLab today (outside `native/`)

Two existing dependencies constrain distribution regardless of the native
work, so they belong in the same conversation:

* **aubio is GPLv3+.** `soniclab/core/beats.py` imports it in-process, so a
  distributed SonicLab is already GPLv3-bound unless aubio is removed or
  replaced. (GPLv3 and AGPLv3 are compatible, so choosing JUCE AGPLv3 does
  not add a conflict; it adds the network clause.)
* **Madmom models are CC BY-NC-SA (non-commercial).** The code is BSD,
  but the pretrained beat/downbeat networks used by the worker are not for
  commercial use. Porting those models into the native Beat Analyzer would
  carry the same restriction. A commercial detector needs its own models
  or an algorithm without them.
