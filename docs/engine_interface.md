# Audio engine interface

This is the boundary between the UI and the audio engine. The UI
(`soniclab/ui/`) may use only what is listed here. A future native engine
(see `native/docs/ARCHITECTURE.md`) must provide the same surface. Then a
Python class wrapping it can replace `AudioEngine` in `MainWindow.__init__`
and nothing else in the UI changes.

`tests/test_engine_interface.py` covers the methods added for this
boundary. It also fails if UI code reads an `engine._private` field again.

## Units and identity

* **Clip index equals project track index.** `add_clip` appends and
  returns the new index. `clear_clips` removes every clip. There is no
  per-clip removal or reorder. The UI rebuilds all clips with
  `clear_clips` and `add_clip` instead.
* **Seconds** are used for positions, clip starts, regions, fades and the
  loop. **Frames** at `OUTPUT_RATE` (44 100 Hz) are used only by
  `render(start_frame, frames, include)`.
* **Audio** passed to `add_clip` is float32 at `OUTPUT_RATE`, mono or
  stereo. Callers resample first with `AudioEngine.resample`.

## Surface

| Group | Members | Notes |
|---|---|---|
| Clips | `add_clip(audio, start_seconds) -> int`, `clear_clips()`, `clip_count` | `clip_count` is read-only |
| Per-clip mix | `set_clip_start`, `set_clip_gain`, `set_clip_pan`, `set_clip_muted`, `set_clip_solo`, `set_clip_region(index, start, offset, duration, fade_in, fade_out, clip_gain)` | An out-of-range index is ignored |
| Master | `set_output_gain(gain)` (clamped 0..1.5), `output_gain`; `set_master_muted(bool)`, `master_muted` | The properties return plain `float` and `bool` |
| Transport | `play()`, `pause()`, `stop()` (also rewinds to 0), `seek(seconds)`, `position` (seconds), `playing` | |
| Loop | `set_loop(start, end) -> bool`, `clear_loop()`, `loop` (`(start, end)` or `None`) | The end is exclusive |
| Offline mix | `render(start_frame, frames, include=None)` | Same mix as playback; used by bar export |
| Meters and scope | `take_peaks() -> (master, [tracks])` (reads and resets), `set_capture(bool)`, `read_master(count)`, `read_track(index, count)` | Scope reads return N×2 float32 |
| Lifecycle | `close()` | Stops and releases the output stream |
| Utility | `AudioEngine.resample(audio, rate)` (static), `OUTPUT_RATE`, `BLOCK_SIZE` | |

Threading: every member may be called from the Qt main thread or a worker
thread. None may be called from inside the audio callback. In the Python
engine they take the engine lock. In a native engine, setters become
queued commands (`native/common/.../engine_command.h`) and getters read
atomics, so values a setter just wrote may show up one audio block later.
The tests read values straight back after setting them. That works with
the Python engine. A native wrapper should keep a copy of what was last
set on the Python side, so that reading straight back still works.

## Not part of the boundary

These stay implementation details. A native engine does not have to
reproduce them:

* `_callback` and every underscore attribute (`_clips`, `_output_gain`,
  `_master_muted`, `_playing`, `_position`, `_capture_enabled`, ...).
* The clip dictionary format. `replace_clips(clips)` remains for backward
  compatibility, but it takes that internal format. The UI now uses only
  `clear_clips()`. New code should not call `replace_clips` with a
  non-empty list.
* `render_clips(clips, ...)`, the module-level offline twin of the
  callback. It works on clip dictionaries, so it belongs to the Python
  engine. Callers go through `engine.render`.

## Remaining private-field use (tests only)

These tests drive the callback directly or inspect internal clip state.
They test the Python engine's internals, not the boundary. A native engine
needs its own equivalents for them:

* `tests/test_phase1.py`
  * sets `_playing` and calls `_callback`
  * reads and writes `_clips[i]["start"]` and `["muted"]`, and `len(_clips)`
* `tests/test_phase2.py`
  * sets `_playing` and calls `_callback`
  * reads `_output_gain`
* `tests/test_downbeat_loop.py`
  * sets `_playing`, calls `_callback` and reads `_position`
* `tests/test_bar_export.py`
  * sets `_capture_enabled`, `_playing` and `_position`, and calls `_callback`
* `tests/test_arrow_keys.py`
  * reads `_output_gain`

The `_output_gain` reads and the `len(_clips)` read have public
equivalents (`output_gain` and `clip_count`). They were left as they are
to keep this change small.
