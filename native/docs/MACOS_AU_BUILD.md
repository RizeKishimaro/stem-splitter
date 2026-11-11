# Building and testing the SonicLab Beat Analyzer AU on macOS

This guide is for the person with the Mac and Logic Pro. **Nobody has
built this AU yet.** The plugin code was compiled and tested on Linux,
where Audio Units cannot be built, so the steps below are the first real
macOS build. Please send back the outputs listed under
[Report back](#report-back), whether they succeed or fail.

| | |
|---|---|
| Product | SonicLab Beat Analyzer |
| Format | Audio Unit v2 effect (`aufx`) |
| Manufacturer / subtype | `Snlb` / `Btan` |
| Manufacturer name in Logic | SonicLab |
| Version | 0.2.0 |
| Bundle ID | `org.soniclab.beatanalyzer` |
| Artifact | `SonicLab Beat Analyzer.component` |
| Install location | `~/Library/Audio/Plug-Ins/Components/` |
| Validate with | `auval -v aufx Btan Snlb` |

## What the plugin does (0.2.0)

* Audio passes through **unchanged**, on mono or stereo tracks.
* The editor shows the input level (peak meter, -60 to +6 dBFS, with
  peak hold) and an **Enabled** switch.
* **Enabled** is an automatable parameter that is saved with the project.
  Turning it off stops metering and analysis. Audio still passes through.
* **Beat tracking.** While audio plays, the display shows the tempo in
  BPM and the beats per bar, each with a five-step confidence bar. The
  "Live beats" lane shows the last 8 seconds of beats, numbered within
  the bar, with downbeats in amber. Beats appear about a quarter of a
  second after they sound, and the tempo needs about 4 seconds to lock.
* **Settings**, all automatable and saved with the project: *Tempo* (the
  slowest and fastest tempo to look for, 60 and 200 BPM by default) and
  *Beats / bar* (Auto chooses 3 or 4; 2 to 7 fix the bar length).
  **Reset** forgets the live analysis.
* Dropping an audio file (WAV, AIFF, FLAC, Ogg, and on macOS also MP3,
  M4A and other Core Audio formats) onto the plugin window analyses it in
  the background and shows its waveform with beats and downbeats, its
  tempo, beats per bar, length and peak. **Analyse again** re-runs it with
  the current settings. The track's audio is not affected, and the file
  is not saved with the project.
* The tracker is SonicLab's own (not Madmom, which the DAW uses). On ten
  test songs it matched Madmom's tempo on eight; downbeats are less
  reliable than beats. If the tempo reads half or double, narrow the
  Tempo range.

## 1. Prerequisites (once)

* macOS 11 or newer, on Apple Silicon or Intel.
* **Full Xcode** from the App Store. The Command Line Tools alone are not
  enough. Open Xcode once to accept the licence and install its
  components, then run:

  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  xcodebuild -version
  ```

* **CMake 3.25 or newer** and **git**, for example `brew install cmake git`.
  Check with `cmake --version`.
* Network access for the first build. It clones JUCE 8.0.4 from GitHub,
  about 100 MB.

## 2. Build, install and validate: one command

From the SonicLab repository:

```bash
cd native
scripts/build_au_macos.sh --install
```

The script does the following, and stops at the first error:

1. `scripts/get_juce.sh` clones JUCE 8.0.4 into `native/third_party/JUCE`
   and verifies commit `51d11a2b…`.
2. `cmake --preset macos-xcode` generates
   `native/build/macos-xcode/SonicLabNative.xcodeproj`. It builds
   universal (arm64 + x86_64) for macOS 11+.
3. `cmake --build --preset macos-xcode --target SonicLabBeatAnalyzer_AU`
   builds the Release configuration.
4. `codesign --force --deep --sign -` ad-hoc signs the component. That is
   enough to load it on this Mac.
5. It copies the component to `~/Library/Audio/Plug-Ins/Components/`.
6. It restarts `AudioComponentRegistrar` so macOS notices the new
   component.
7. It runs `auval -a | grep Snlb`, then `auval -v aufx Btan Snlb`.

Without `--install`, the script stops after step 4 and leaves the build in
`native/build/macos-xcode/plugins/beat_analyzer/juce/SonicLabBeatAnalyzer_artefacts/Release/AU/`.

## 3. The same steps by hand

```bash
cd native
scripts/get_juce.sh
cmake --preset macos-xcode
cmake --build --preset macos-xcode --target SonicLabBeatAnalyzer_AU
```
```bash
AU="build/macos-xcode/plugins/beat_analyzer/juce/SonicLabBeatAnalyzer_artefacts/Release/AU/SonicLab Beat Analyzer.component"
codesign --force --deep --sign - "$AU"
lipo -info "$AU/Contents/MacOS/SonicLab Beat Analyzer"   # expect: x86_64 arm64
mkdir -p ~/Library/Audio/Plug-Ins/Components
cp -R "$AU" ~/Library/Audio/Plug-Ins/Components/
killall -9 AudioComponentRegistrar
auval -a | grep Snlb
auval -v aufx Btan Snlb
```

You can also open `build/macos-xcode/SonicLabNative.xcodeproj` in Xcode,
choose the `SonicLabBeatAnalyzer_AU` scheme and build there.

Optional extra checks:

* `ctest --preset macos-xcode` runs the native unit tests, including the
  20 plugin tests, natively on the Mac. Build `ALL_BUILD` first.
* `cmake --build --preset macos-xcode --target SonicLabBeatAnalyzer_Standalone`
  builds a standalone app. It is useful for a quick look at the editor
  without Logic.

## 4. AU validation

`auval -a | grep Snlb` should list one line like:

```
aufx Btan Snlb  -  SonicLab: SonicLab Beat Analyzer
```

`auval -v aufx Btan Snlb` runs Apple's validation, which is the same one
Logic uses. It must end with:

```
AU VALIDATION SUCCEEDED.
```

On Apple Silicon you can also validate the Intel slice:
`arch -x86_64 auval -v aufx Btan Snlb`.

## 5. Logic Pro scanning and test

1. **Quit Logic** if it is open. Start Logic Pro. On launch it scans new
   Audio Units.
2. Open **Logic Pro → Settings → Plug-in Manager**. Find **SonicLab →
   SonicLab Beat Analyzer**. The *Compatibility* column must say
   *successfully validated*.
   * If it says *failed validation* or is missing: select it, click
     **Reset & Rescan Selection**, and copy the result text.
   * Logic caches components by version. After rebuilding the same
     version, use *Reset & Rescan Selection* so Logic reloads it.
3. Create an **audio track (stereo)**. Click an Audio FX slot → **Audio
   Units → SonicLab → SonicLab Beat Analyzer → Stereo**.
4. Play some audio, then check:
   * The sound is unchanged. For a strict check, duplicate the track, put
     the plugin on one copy, invert polarity on the other (Gain plugin →
     Invert), and the sum should be silent.
   * The meter moves with the input and the "listened" time counts up.
     It must never say "dropped".
   * After a few seconds the BPM appears and matches the project tempo
     (try a drum loop or a full song at a known tempo), and the Live beats
     lane shows lines on the beat with amber downbeats.
   * Untick **Enabled**: the sound is unchanged, and the meter, tempo and
     counter stop.
5. Repeat on a **mono** audio track and choose the **Mono** version of
   the plugin.
6. **State:** turn Enabled off, set Tempo to 90–150 and Beats / bar to 3,
   save the project, close it, reopen it. The plugin should open with the
   same four settings. A project saved with version 0.1.x should open
   with Enabled as saved and the new settings at their defaults.
7. **Automation (optional):** show automation for the track. *Enabled*,
   *Slowest tempo*, *Fastest tempo* and *Beats per bar* should be listed
   under SonicLab Beat Analyzer and be drawable.
8. **Dropped file (optional):** drag a WAV or MP3 from **Finder** onto the
   plugin window. It should show "Analysing...", then the waveform with
   beat lines, and under it the length, sample rate, peak, BPM and beats
   per bar. Playback must not glitch meanwhile. Also
   try a drag from Logic's own browser or a region, and say whether the
   plugin or Logic took it.

## 6. Troubleshooting

| Symptom | Try |
|---|---|
| `cmake --preset macos-xcode`: "Cannot use disabled preset" | You are not on macOS, or CMake is older than 3.25 |
| "No CMAKE_CXX_COMPILER" or Xcode errors | `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, then open Xcode once |
| "JUCE was not found" | Run `scripts/get_juce.sh`. Check `native/third_party/JUCE/CMakeLists.txt` exists |
| `auval -a` does not list Snlb | `killall -9 AudioComponentRegistrar`; check the component is in `~/Library/Audio/Plug-Ins/Components/`; log out and in |
| auval: "Cannot open component" or a code-signing error | Re-run the `codesign --force --deep --sign -` line on the **installed** copy |
| The component was copied from another Mac | `xattr -dr com.apple.quarantine ~/Library/Audio/Plug-Ins/Components/"SonicLab Beat Analyzer.component"` |
| Logic shows the old build | Plug-in Manager → Reset & Rescan Selection, or bump `VERSION` in `native/plugins/beat_analyzer/juce/CMakeLists.txt` |
| Logic: "failed validation" | Run `auval -v aufx Btan Snlb` in Terminal and send the full output |

## Report back

Please send:

1. Output of `sw_vers`, `xcodebuild -version`, `cmake --version`, and
   whether the Mac is Apple Silicon or Intel.
2. The full output of `scripts/build_au_macos.sh --install`, or the first
   error.
3. The full output of `auval -v aufx Btan Snlb`.
4. The Plug-in Manager status line for the plugin.
5. A screenshot of the editor while audio plays, plus whether the
   null test was silent.
6. Whether dropping a file onto the plugin window worked (step 5.8).

## Distribution (not yet)

Ad-hoc signing only works on the Mac that built the plugin. Giving the
component to anyone else requires all of the following (see
`native/LICENSING.md`):

* a Developer ID Application certificate and the hardened runtime
* notarization (`xcrun notarytool`)
* a signed `.pkg`
* **settling the JUCE licence question** (AGPLv3 or commercial)
