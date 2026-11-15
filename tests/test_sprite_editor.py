"""The sprite editor keeps the dance on the BPM and the drums on their own layers."""

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from soniclab.core.dancer import pose_at
from soniclab.core.sprite_project import (
    Animation,
    EditorSession,
    SheetSpec,
    SpriteProject,
    Sync,
    base_frame_id,
    display_source,
    frame_id_at,
    guess_frame_size,
    heart_dancer_project,
    overlay_frame,
    shown_frame_id,
    validate,
)
from soniclab.ui.dancer_view import DancerView
from soniclab.ui.sprite_editor import SpriteEditorWindow

ASSETS = Path(__file__).resolve().parents[1] / "soniclab" / "assets"
EXAMPLE = ASSETS / "heart_dancer.sprite.json"
DOWNS = tuple(float(bar) * 2.0 for bar in range(8))


def _clip(*frames, loop="loop", beats=None, trigger="", multiplier=1.0, cooldown=0.0, sensitivity=0.5):
    frames = list(frames)
    step = 1.0 if beats is None else beats / len(frames)
    return Animation(
        name="clip",
        frames=frames,
        durations=[step] * len(frames),
        loop=loop,
        trigger=trigger,
        sync=Sync(trigger or "beat", "beat", step * len(frames), multiplier, 1, sensitivity, cooldown),
    )


class ProjectTests(unittest.TestCase):
    def test_the_heart_sheet_is_cut_into_the_choreography_frames(self):
        project = heart_dancer_project()
        self.assertEqual(project.bpm, 125)
        self.assertEqual(project.sheet.frame_width, 384)
        self.assertEqual(project.sheet.columns, 12)
        self.assertEqual(project.sheet.rows, 8)
        self.assertEqual(len(project.frames), 96)
        self.assertEqual(project.frames[0].name, "Beat 1 1")
        self.assertEqual(project.frames[48].name, "Beat 5 1")
        self.assertEqual(project.base_animation().name, "body_bounce")
        self.assertEqual(len(project.animations["body_pixel"].frames), 32)
        self.assertEqual(project.animations["snare_clap"].trigger, "snare")
        self.assertEqual(project.animations["snare_clap"].frames, [48, 49, 50])
        self.assertEqual(validate(project.to_dict()), [])
        self.assertEqual(guess_frame_size(4608, 3072), (384, 384, 12, 8))

    def test_spacing_is_part_of_the_slice(self):
        sheet = SheetSpec("sheet.png", 10, 8, 3, 2, spacing_x=1, spacing_y=2, margin_x=4, margin_y=5)
        self.assertEqual(sheet.rect(4), (15, 15, 10, 8))

    def test_the_same_bpm_keeps_the_same_base_dance(self):
        project = heart_dancer_project()
        bpm = 125.0
        times = [index * 0.02 for index in range(80)]
        plain = [base_frame_id(project, time, bpm) for time in times]
        self.assertEqual([base_frame_id(project, time, bpm) for time in times], plain)
        project.animations["hat_gesture"].sync.speed_multiplier = 4
        self.assertEqual([base_frame_id(project, time, bpm) for time in times], plain)
        moment = 0.5
        self.assertEqual(base_frame_id(project, moment, bpm), shown_frame_id(project, moment, [], bpm))
        self.assertNotEqual(
            shown_frame_id(project, moment, [{"kind": "hat", "time": moment, "confidence": 1.0}], bpm),
            base_frame_id(project, moment, bpm),
        )
        self.assertNotEqual(
            shown_frame_id(project, moment, [{"kind": "kick", "time": moment, "confidence": 1.0}], bpm),
            base_frame_id(project, moment, bpm),
        )
        # The clock frame the dancer already chose is not rewritten by a snare.
        self.assertEqual(display_source(project, 10, {}), 10)
        self.assertEqual(display_source(project, 10, {"snare": 1.0}), 48)

    def test_loop_modes_and_the_trigger_cooldown(self):
        forward = _clip(0, 1, 2, 3)
        self.assertEqual(frame_id_at(forward, 0, 60), 0)
        self.assertEqual(frame_id_at(forward, 4, 60), 0)
        once = _clip(0, 1, 2, 3, loop="once")
        self.assertEqual(frame_id_at(once, 10, 60), 3)
        reverse = _clip(0, 1, 2, 3, loop="reverse")
        self.assertEqual(frame_id_at(reverse, 0, 60), 3)
        self.assertEqual(frame_id_at(reverse, 1, 60), 2)
        ping = _clip(0, 1, 2, 3, loop="pingpong")
        self.assertEqual([frame_id_at(ping, second, 60) for second in range(7)], [0, 1, 2, 3, 2, 1, 0])
        project = SpriteProject(bpm=125, frames=[])
        project.animations["kick_drop"] = _clip(0, 1, 2, trigger="kick", cooldown=0.07, sensitivity=0.5)
        project.frames = []
        # Frame ids are created for the sequence.
        project.normalize()
        events = [
            {"kind": "kick", "time": 0.0, "confidence": 1.0},
            {"kind": "kick", "time": 0.02, "confidence": 1.0},
        ]
        self.assertEqual(shown_frame_id(project, 0.01, events, 125), 0)
        quiet = [{"kind": "kick", "time": 0.0, "confidence": 0.2}]
        self.assertIsNone(overlay_frame(project, 0.01, quiet, 125))

    def test_json_roundtrip_and_the_example_file(self):
        project = heart_dancer_project()
        loaded = SpriteProject.from_dict(project.to_dict(), root=project.root)
        self.assertEqual(loaded.to_dict(), project.to_dict())
        example = {
            "name": "heart_dancer",
            "spritesheet": {"file": "heart_dance.png", "frame_width": 384, "frame_height": 384},
            "animations": {
                "body_bounce": {
                    "layer": "body",
                    "frames": [0, 1, 2, 3],
                    "loop": True,
                    "sync": {"type": "beat", "beats_per_cycle": 1, "speed_multiplier": 1.0},
                },
                "snare_clap": {
                    "layer": "arms",
                    "frames": [16, 17, 18],
                    "trigger": "snare",
                    "speed_multiplier": 2.0,
                },
            },
        }
        self.assertEqual(validate(example), [])
        short = SpriteProject.from_dict(example)
        self.assertEqual(short.base_animation().frames, [0, 1, 2, 3])
        self.assertEqual(short.animation_for_trigger("snare").layer, "arms")
        self.assertEqual(short.animation_for_trigger("snare").sync.speed_multiplier, 2.0)
        self.assertEqual(base_frame_id(short, 0.0, 120), 0)
        on_disk = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(validate(on_disk), [])
        stored = SpriteProject.load(EXAMPLE)
        self.assertEqual(stored.name, "heart_dancer")
        self.assertTrue(stored.resolve_sheet().is_file())

    def test_frames_can_be_reordered_copied_and_undone(self):
        session = EditorSession(heart_dancer_project())
        session.change(lambda project: project.move_frames([0], 2))
        self.assertEqual([item.id for item in session.project.frames[:4]], [1, 2, 0, 3])
        session.change(lambda project: project.duplicate_frames([0]))
        copy = session.project.frames[3]
        self.assertEqual(copy.source, 0)
        self.assertIn("copy", copy.name)
        session.clipboard = [copy.id]
        session.change(lambda project: project.rename_frame(copy.id, "Jack"))
        self.assertEqual(session.project.frame(copy.id).name, "Jack")
        session.change(lambda project: project.delete_frames([copy.id]))
        self.assertIsNone(session.project.frame(copy.id))
        self.assertNotIn(copy.id, session.project.animations["body_bounce"].frames)
        self.assertTrue(session.undo())
        self.assertEqual(session.project.frame(copy.id).name, "Jack")
        session.change(lambda project: project.variation("body_bounce", "pixel_keys", 3))
        self.assertEqual(len(session.project.animations["pixel_keys"].frames), 32)


class EditorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_the_window_edits_the_heart_project_and_can_send_it(self):
        window = SpriteEditorWindow()
        window.resize(1100, 700)
        window.show()
        self.app.processEvents()
        self.assertEqual(window.frame_list.count(), 96)
        self.assertEqual(window.timeline.count(), 96)
        self.assertIn("body_bounce", window.json_edit.toPlainText())
        self.assertIn("base 0", window.caption.text())
        window.frame_list.item(0).setSelected(True)
        window.duplicate_selected()
        self.assertEqual(window.frame_list.count(), 97)
        window.undo()
        self.assertEqual(window.frame_list.count(), 96)
        window.reorder_frames([0], 3)
        self.assertEqual([item.id for item in window.session.project.frames[:4]], [1, 2, 0, 3])
        window.drop_on_timeline([4], 0, "frames")
        self.assertEqual(window.session.project.animations["body_bounce"].frames[0], 4)
        base = window.base_frame()
        window.test_trigger("snare")
        self.assertEqual(window.base_frame(), base)
        self.assertEqual(window.current_frame_id(), 48)
        sent = []
        window.send_requested.connect(sent.append)
        window._send()
        self.assertEqual(sent[0]["name"], "heart_dancer")
        self.assertEqual(validate({key: value for key, value in sent[0].items() if key != "_root"}), [])
        window.preview.repaint()
        window.sheet_view.repaint()
        window.close()

    def test_the_dancer_keeps_its_clock_when_a_project_is_sent(self):
        project = heart_dancer_project()
        data = project.to_dict()
        data["_root"] = project.root
        view = DancerView()
        view.set_clock(lambda: (0.075, 120.0, DOWNS, None, False))
        view.apply_sprite_project(data)
        self.assertEqual(view.pose.frame, pose_at(0.075, 120.0, DOWNS).frame)
        self.assertEqual(display_source(project, view.pose.frame, {}), view.pose.frame)
        view.resize(240, 260)
        view.show()
        self.app.processEvents()
        picture = QImage(view.stage.size(), QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(picture)
        view.paint_stage(painter, picture.width(), picture.height())
        painter.end()
        view._timer.stop()
        view.close()

    def test_the_arrangement_opens_the_editor(self):
        from soniclab.ui.main_window import MainWindow

        window = MainWindow()
        titles = []
        for action in window.menuBar().actions():
            menu = action.menu()
            if menu is None:
                continue
            titles.extend(item.text() for item in menu.actions())
        self.assertTrue(any("Sprite Animation" in title for title in titles))
        window._open_sprite_editor()
        self.app.processEvents()
        self.assertTrue(window._sprite_editor.isVisible())
        self.assertEqual(window._sprite_editor.frame_list.count(), 96)
        clock = window._sprite_song_clock()
        self.assertIn("clock_frame", clock)
        self.assertEqual(clock["bpm"], 120.0)
        window._sprite_editor.close()
        window.close()


if __name__ == "__main__":
    unittest.main()
