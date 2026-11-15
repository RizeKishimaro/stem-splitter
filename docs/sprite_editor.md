# Sprite Animation Editor

The editor is a separate window for cutting a sprite sheet into frames and
timing those frames to a beat. It opens from **View → Sprite Animation Editor**.
Nothing extra has to be installed. It uses the same PySide6 app as the rest
of SonicLab.

The first project is the heart dancer: `soniclab/assets/heart_dancer.sprite.json`
and `soniclab/assets/heart_dance_smooth_sheet.png`. The sheet is 12 columns by
8 rows of 384×384 frames (96 frames). Frame names follow the choreography
counts (`1 e & a`) at 125 BPM. The editor is not limited to that character.
Import any PNG and set the cell size.

The dancer dock starts on the heart dancer. **View → Heart Dancer**, or **Heart** on the dancer, brings that character back. The green dancer is a second sheet, measured rather than guessed: 8×10 cells of 110×128. Its body dance is one pose per beat, and kick, snare, hat, bass and percussion show the matching pose. **File → Open Green Dancer** loads it. **File → Open Heart Dancer** loads the heart project. **View → Verify Mapping** draws each source rectangle, the tight content box, and the frame number. The X Y W H fields correct a rectangle, and Category changes a pose label. The map, the two sequences, and the reading notes are in `docs/green_dancer_mapping_report.md`.

## Layout

- **Left:** the sheet, with a transparency checker and cell grid, and every frame as a selectable thumbnail. Drag thumbnails to reorder them. Ctrl-click the sheet to add to the selection.
- **Center:** the preview, a metronome lamp, and buttons that fire downbeat, beat, kick, snare, hat, bass and percussion.
- **Bottom:** the current animation’s sequence. Drag frames in from the browser, or drag them along the strip to reorder.
- **Right:** the animation, its layer, and the beat settings.
- **Far right:** the project JSON. It updates as you edit. Turn on **Edit JSON** to change it, then **Apply** or **Validate**.

The panels are splitters, so the window can be resized.

## Timing

BPM is the clock for the base animation (the one marked **Base dance**).
Beats per cycle, the grid (beat, eighth, sixteenth) and each frame’s length
in beats are measured on that clock. A speed multiplier belongs to one
animation. Changing the hat animation’s multiplier does not move the base
dance. Playback speed on the toolbar only affects the preview.

Drum buttons and detected hits start their own animations for the length of
that sequence. They do not restart the base frame. A cooldown stops a second
hit of the same kind from firing again too soon. Sensitivity is 0 to 1;
higher values accept quieter hits.

Layers are body, left arm, right arm, left leg, right leg, beat accents and
instrument effects. A full-frame drawing can be assigned to a layer. The
sheet is not sliced into limbs. **Combined** plays the base dance and lays
triggers on top. The layer menu previews one layer on its own.

Loop modes are loop, once, reverse and ping-pong.

## Song preview

**Follow song** reads the open arrangement: playback position, BPM, the
dancer’s clock frame, and kick, snare, hat, bass and percussion hits the
dancer already detected. Detection still runs on the UI timer, not in the
audio callback.

**Send to Dancer** gives the project to the dancer dock. The dancer keeps
its own clock. The base sequence picks which cell that clock shows, and a
kick, snare, hat or percussion hit can show that animation’s frame until the
hit fades. Bass stays a sway, so a held note does not swap the picture.

## JSON

Files are `soniclab.sprite` version 1.

```json
{
  "format": "soniclab.sprite",
  "version": 1,
  "name": "heart_dancer",
  "bpm": 125,
  "spritesheet": {
    "file": "heart_dance_smooth_sheet.png",
    "frame_width": 384,
    "frame_height": 384,
    "columns": 12,
    "rows": 8,
    "spacing_x": 0,
    "spacing_y": 0
  },
  "frames": [{"id": 0, "name": "Beat 1 1", "source": 0}],
  "animations": {
    "body_bounce": {
      "layer": "body",
      "frames": [0, 1, 2, 3],
      "durations": [0.083, 0.083, 0.083, 0.083],
      "fps": 25,
      "loop": "loop",
      "base": true,
      "speed_multiplier": 1.0,
      "sync": {
        "type": "beat",
        "grid": "sixteenth",
        "beats_per_cycle": 8,
        "speed_multiplier": 1.0,
        "intensity": 1.0,
        "sensitivity": 0.5,
        "cooldown": 0.0
      }
    },
    "snare_clap": {
      "layer": "left_arm",
      "frames": [48, 49, 50],
      "trigger": "snare",
      "loop": "once",
      "speed_multiplier": 1.0,
      "sync": {"type": "snare", "beats_per_cycle": 0.25, "speed_multiplier": 1.0, "cooldown": 0.08}
    }
  }
}
```

`frames` are the browser order. `source` is the cell in the sheet, so
reordering or duplicating a frame does not move its pixels. `durations` are
in beats, one per frame. `loop` may be `loop`, `once`, `reverse`, `pingpong`,
or a boolean (`true` means loop). `trigger` is `downbeat`, `beat`, `kick`,
`snare`, `hat`, `bass` or `perc`. An animation with no trigger can be the
base dance. `instruments` in the file repeats the trigger map. Paths are
relative to the project file.

A shorter file is accepted too: a spritesheet with only `file`,
`frame_width` and `frame_height`, and animations that list frame numbers.
**Validate** reports a bad frame id, a non-positive speed, two base dances,
or a grid that runs off the image.

## Shortcuts

| Action | Shortcut |
|---|---|
| Undo / redo | Ctrl+Z, and the platform redo shortcut |
| Duplicate, copy, paste | Ctrl+D, Ctrl+C, Ctrl+V |
| Delete | Delete |
| Rename | F2 |
| Save | Ctrl+S |
| Copy JSON | Ctrl+Shift+C |
| Send to dancer | Ctrl+Enter |

Play and Stop are on the toolbar. Space is left to the text fields.
