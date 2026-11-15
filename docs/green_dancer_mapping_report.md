# Green dancer sprite map

The source image is 880×1280. It was not modified. The copy used by the editor is `soniclab/assets/green_dancer/sprites.png`.

## Grid

The layout was measured from the pixels, not assumed.

| | |
|---|---|
| Frames | 80 |
| Columns × rows | 8 × 10 |
| Cell | 110 × 128 |
| Spacing | 0, 0 |
| Margin | 0, 0 |
| Origin | 0, 0 |
| Connected sprites | 80 |
| Sprites crossing a cell edge | 0 |
| Empty cells | 0 |
| Grid confidence | high |

880 ÷ 110 = 8 and 1280 ÷ 128 = 10, with no remainder. Labeling every non-transparent pixel gives one sprite per cell. Each sprite's pixels stay inside its cell. Opaque black outlines are part of the drawing. The background is transparent, so it looks black when composited.

The source rectangle for every frame is the full cell. Content often comes within 2 pixels of the bottom edge, and a raised hand or the sprout can come within a few pixels of the top or side. A tighter crop would cut feet, hair, or hands. The tight pixel bounds are stored as `content_rect` and are not the crop.

No two frames are byte-identical.

## Content inset

Left, right, top, and bottom empty pixels inside each cell. Bottom insets of 2 mean the feet are almost on the cell edge.

| Frame | L | R | T | B |
|---|---:|---:|---:|---:|
| 000 | 30 | 30 | 22 | 2 |
| 001 | 30 | 30 | 22 | 2 |
| 002 | 29 | 30 | 22 | 2 |
| 003 | 31 | 31 | 24 | 2 |
| 004 | 30 | 30 | 22 | 2 |
| 005 | 30 | 30 | 22 | 2 |
| 006 | 30 | 30 | 22 | 2 |
| 007 | 30 | 30 | 22 | 2 |
| 008 | 30 | 30 | 23 | 2 |
| 009 | 30 | 30 | 24 | 2 |
| 010 | 31 | 32 | 22 | 3 |
| 011 | 30 | 31 | 21 | 3 |
| 012 | 20 | 39 | 21 | 3 |
| 013 | 20 | 40 | 21 | 2 |
| 014 | 22 | 41 | 21 | 3 |
| 015 | 21 | 40 | 22 | 3 |
| 016 | 27 | 28 | 31 | 2 |
| 017 | 34 | 35 | 14 | 19 |
| 018 | 24 | 31 | 3 | 32 |
| 019 | 23 | 24 | 7 | 20 |
| 020 | 30 | 30 | 23 | 2 |
| 021 | 27 | 28 | 31 | 2 |
| 022 | 30 | 30 | 23 | 2 |
| 023 | 30 | 30 | 23 | 2 |
| 024 | 31 | 30 | 24 | 2 |
| 025 | 19 | 32 | 25 | 2 |
| 026 | 31 | 30 | 24 | 2 |
| 027 | 30 | 30 | 24 | 2 |
| 028 | 29 | 32 | 23 | 2 |
| 029 | 32 | 20 | 25 | 2 |
| 030 | 29 | 32 | 23 | 2 |
| 031 | 30 | 30 | 24 | 2 |
| 032 | 6 | 28 | 40 | 2 |
| 033 | 13 | 27 | 34 | 3 |
| 034 | 28 | 29 | 21 | 3 |
| 035 | 26 | 14 | 29 | 3 |
| 036 | 27 | 8 | 39 | 2 |
| 037 | 26 | 14 | 30 | 3 |
| 038 | 28 | 29 | 21 | 3 |
| 039 | 13 | 27 | 32 | 3 |
| 040 | 20 | 33 | 20 | 2 |
| 041 | 18 | 32 | 20 | 2 |
| 042 | 20 | 33 | 20 | 2 |
| 043 | 30 | 30 | 22 | 3 |
| 044 | 31 | 22 | 21 | 2 |
| 045 | 30 | 19 | 21 | 2 |
| 046 | 31 | 22 | 20 | 2 |
| 047 | 30 | 30 | 22 | 3 |
| 048 | 22 | 10 | 59 | 2 |
| 049 | 33 | 14 | 67 | 2 |
| 050 | 23 | 23 | 61 | 2 |
| 051 | 25 | 21 | 41 | 3 |
| 052 | 25 | 23 | 49 | 3 |
| 053 | 19 | 18 | 48 | 2 |
| 054 | 18 | 9 | 58 | 2 |
| 055 | 19 | 19 | 48 | 2 |
| 056 | 26 | 24 | 27 | 2 |
| 057 | 16 | 29 | 32 | 2 |
| 058 | 11 | 23 | 29 | 2 |
| 059 | 14 | 29 | 24 | 3 |
| 060 | 25 | 28 | 20 | 2 |
| 061 | 27 | 26 | 23 | 2 |
| 062 | 24 | 17 | 27 | 2 |
| 063 | 24 | 32 | 26 | 2 |
| 064 | 26 | 26 | 26 | 3 |
| 065 | 31 | 30 | 28 | 3 |
| 066 | 28 | 24 | 29 | 3 |
| 067 | 31 | 17 | 29 | 3 |
| 068 | 30 | 22 | 28 | 3 |
| 069 | 30 | 33 | 26 | 3 |
| 070 | 26 | 26 | 23 | 3 |
| 071 | 23 | 24 | 24 | 2 |
| 072 | 37 | 37 | 13 | 4 |
| 073 | 37 | 36 | 13 | 5 |
| 074 | 35 | 35 | 13 | 8 |
| 075 | 34 | 33 | 13 | 10 |
| 076 | 33 | 32 | 13 | 11 |
| 077 | 34 | 33 | 13 | 10 |
| 078 | 35 | 35 | 13 | 8 |
| 079 | 37 | 36 | 13 | 5 |

## Categories

These are pose families, not confirmed move names. Confidence is high, medium, or low. Low and medium frames can be changed in the editor.

| Category | Frames |
|---|---:|
| idle | 19 |
| dance | 0 |
| arms | 28 |
| kick | 5 |
| lean | 8 |
| spin | 8 |
| turn | 8 |
| unknown | 4 |

Confidence counts: high 29, medium 48, low 3.

Nothing was labeled dance. The rows that look like dancing are standing, arms, leans, kicks, a roll, or a turn, and the unclear cells stay unknown.

## Playback

The heart dancer stays the character in the dancer dock. View → Heart Dancer brings it back. The green dancer is a separate project. Its dance follows the beat clock. A drum hit shows that instrument's pose and does not speed up the dance.

| Clip | Trigger | Frames |
|---|---|---|
| body | beat clock | 0, 72, 74, 20, then 64, 66, 68, 71. One pose per beat. |
| kick | kick | 56, 57, 62 |
| snare | snare | 18, 60, 24 |
| hat | hat | 9, 40, 41 |
| bass | bass | 32, 34, 36, 38 |
| perc | percussion | 48, 50, 52, 55 |

Tumble is frames 48–55 in sheet order. Turn is frames 64–71. Those clips have no instrument. The second half of the body dance uses the turn.

The other rows are groups, not animations:

- **standing** — Same standing family. Frames 4, 5 and 6 are nearly the same picture. Not one animation.
- **wings** — Wing or coat changes. Not exported as an animation.
- **mixed_row_2** — Standing frames mixed with a raised arm and two unnamed poses. Not one movement.
- **arms_front** — Arms held in front. A pose family, not a confirmed clip.
- **leans** — Lean family in sheet order. It may be a pose set rather than one move, so it is not an animation.
- **arms_row** — Arm and wing positions. Not a confirmed clip.
- **kick_row** — Kicks are mixed with arm poses, so the row is not one animation.
- **front_standing** — Front standing views. Neighbor frames differ. Not one animation.

The sheet is not one 80-frame animation. The body dance is eight poses.

## Uncertain frames

High-confidence labels are the standing frames 0 and 4–6, the raised arm on 18, the leans on 32 and 34–38, the roll on 48–54, the kicks on 56, 57 and 62, and the turn on 64–71.

Every other frame is medium or low. The low ones are 16, 21 and 23. Their drawings differ from the standing pose and do not match a named move.

| Frame | Category | Confidence | Note |
|---|---|---|---|
| 001 | idle | medium | Standing. The pixels differ from frame 0. The gesture is not named. |
| 002 | idle | medium | Standing. The pixels differ from frame 0. The gesture is not named. |
| 003 | idle | medium | Standing. The pixels differ from frame 0. The gesture is not named. |
| 007 | idle | medium | Standing. The pixels differ from frame 0. The gesture is not named. |
| 008 | idle | medium | Close to the standing frames. A small wing change is possible and not named. |
| 009 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 010 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 011 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 012 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 013 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 014 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 015 | arms | medium | Coat or wing position changes along the row. Not confirmed as one clip. |
| 016 | unknown | low | The drawing starts lower in the cell than frame 0 and still reaches the bottom. Pose not named. |
| 017 | unknown | medium | The drawing sits higher in the cell than the standing frames and does not reach their foot line. The move is not named. |
| 019 | arms | medium | An arm is extended and the head is tilted. The body is high in the cell. |
| 020 | idle | medium | Standing, close to frame 0. |
| 021 | unknown | low | The drawing starts lower in the cell than frame 0 and still reaches the bottom. Pose not named. |
| 022 | idle | medium | Standing, close to frame 0. |
| 023 | unknown | low | Differs from the standing frames. A face change is possible and not confirmed. |
| 024 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 025 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 026 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 027 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 028 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 029 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 030 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 031 | arms | medium | Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip. |
| 033 | lean | medium | A lean, with a leg forward. A kick is not confirmed. |
| 039 | lean | medium | A lean, with a leg forward. A kick is not confirmed. |
| 040 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 041 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 042 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 043 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 044 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 045 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 046 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 047 | arms | medium | Arm and wing positions. Not confirmed as one clip. |
| 055 | spin | medium | The roll is finishing and the body is more upright. |
| 058 | kick | medium | A leg is lifted. The pose is less clear than 56 and 57. |
| 059 | arms | medium | Hands are near the waist. The supporting leg is down. |
| 060 | arms | medium | Hands are together in front. Not a confirmed kick. |
| 061 | kick | medium | A leg is lifted. The pose is less clear than 56 and 57. |
| 063 | arms | medium | A hand is near the head. |
| 072 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 073 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 074 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 075 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 076 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 077 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 078 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |
| 079 | idle | medium | Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named. |

## Files

- `soniclab/assets/green_dancer/sprites.png` — copy of the source
- `soniclab/assets/green_dancer/green_dancer_sprites.json` — frame ids and rectangles
- `soniclab/assets/green_dancer/green_dancer_animations.json` — body dance and the instrument poses
- `soniclab/assets/green_dancer/green_dancer_contact_sheet.png` — the sheet at 2× with frame numbers
- `soniclab/assets/green_dancer/green_dancer.sprite.json` — the editor project

In the Sprite Animation Editor, **File → Open Green Dancer** loads the 80 frames. Each thumbnail shows the frame id and the source position. **View → Verify Mapping** draws every rectangle, the tight content box, and the frame number. The X Y W H fields correct a rectangle. Category can be changed; a manual change is stored as confidence `manual`. The preview scales the selected cell by a whole number of pixels.

File → Open Heart Dancer loads the heart project. The dancer dock starts on the heart dancer. Heart on the dancer, or View → Heart Dancer, clears a sent project.
