"""The green dancer sheet is a measured 8×10 map, not one invented dance."""

import hashlib
import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from soniclab.core.sprite_catalog import (
    BASS_FRAMES,
    BODY_POSES,
    CELL_H,
    CELL_W,
    COLUMNS,
    FRAME_COUNT,
    HAT_FRAMES,
    HOLDS,
    IMAGE_H,
    IMAGE_W,
    KICK_FRAMES,
    ORIGINAL,
    PERC_FRAMES,
    ROWS,
    SNARE_FRAMES,
    green_dancer_project,
    measure,
)
from soniclab.core.sprite_project import POSE_CATEGORIES, SpriteProject, display_source, heart_dancer_project, validate
from soniclab.ui.dancer_view import DancerView
from soniclab.ui.sprite_editor import SpriteEditorWindow

ASSET = Path(__file__).resolve().parents[1] / "soniclab" / "assets" / "green_dancer"
SPRITES = ASSET / "green_dancer_sprites.json"
ANIMATIONS = ASSET / "green_dancer_animations.json"
CONTACT = ASSET / "green_dancer_contact_sheet.png"
SHEET = ASSET / "sprites.png"


class GreenDancerMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.measured = measure(SHEET)
        cls.frame_map = json.loads(SPRITES.read_text(encoding="utf-8"))
        cls.animations = json.loads(ANIMATIONS.read_text(encoding="utf-8"))

    def test_the_copy_matches_the_original_and_the_grid_is_measured(self):
        if ORIGINAL.is_file():
            self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), hashlib.sha256(SHEET.read_bytes()).hexdigest())
        self.assertEqual((self.measured["width"], self.measured["height"]), (IMAGE_W, IMAGE_H))
        self.assertEqual(self.measured["components"], FRAME_COUNT)
        self.assertEqual(len(self.frame_map["frames"]), 80)
        self.assertEqual(self.frame_map["grid"]["columns"], COLUMNS)
        self.assertEqual(self.frame_map["grid"]["rows"], ROWS)
        self.assertEqual(self.frame_map["grid"]["cell_width"], CELL_W)
        self.assertEqual(self.frame_map["grid"]["cell_height"], CELL_H)
        self.assertEqual(self.frame_map["grid"]["spacing_x"], 0)
        self.assertEqual(self.frame_map["grid"]["margin_x"], 0)
        self.assertEqual(self.frame_map["animations"], {})

    def test_every_rect_holds_its_sprite_and_tiles_the_sheet(self):
        covered = set()
        for index, item in enumerate(self.measured["frames"]):
            key = f"frame_{index:03d}"
            rect = self.frame_map["frames"][key]["rect"]
            content = self.frame_map["frames"][key]["content_rect"]
            self.assertEqual(rect, [ (index % 8) * CELL_W, (index // 8) * CELL_H, CELL_W, CELL_H ])
            self.assertEqual(item["rect"], rect)
            x, y, w, h = content
            self.assertGreaterEqual(x, rect[0])
            self.assertGreaterEqual(y, rect[1])
            self.assertLessEqual(x + w, rect[0] + rect[2])
            self.assertLessEqual(y + h, rect[1] + rect[3])
            self.assertIn(self.frame_map["frames"][key]["category"], POSE_CATEGORIES)
            for px in range(rect[0], rect[0] + rect[2]):
                for py in range(rect[1], rect[1] + rect[3]):
                    covered.add((px, py))
        self.assertEqual(len(covered), IMAGE_W * IMAGE_H)
        self.assertEqual(validate(self.frame_map), [])

    def test_poses_play_on_the_matching_instrument(self):
        stored = self.animations["animations"]
        self.assertEqual(stored["body"]["frames"], [f"frame_{index:03d}" for index in BODY_POSES])
        self.assertEqual(stored["body"]["holds"], HOLDS)
        self.assertEqual(stored["body"]["trigger"], "")
        self.assertEqual(stored["kick"]["trigger"], "kick")
        self.assertEqual(stored["snare"]["trigger"], "snare")
        self.assertEqual(stored["hat"]["trigger"], "hat")
        self.assertEqual(stored["bass"]["trigger"], "bass")
        self.assertEqual(stored["perc"]["trigger"], "perc")
        for group in self.animations["groups"].values():
            self.assertFalse(group["sequence"])
        project = green_dancer_project()
        self.assertEqual(project.base_animation().name, "body")
        self.assertEqual(project.base_animation().frames, [pose for pose in BODY_POSES for _ in range(HOLDS)])
        self.assertEqual(project.animations["kick"].frames, KICK_FRAMES)
        self.assertEqual(project.animations["snare"].frames, SNARE_FRAMES)
        self.assertEqual(project.animations["hat"].frames, HAT_FRAMES)
        self.assertEqual(project.animations["bass"].frames, BASS_FRAMES)
        self.assertEqual(project.animations["bass"].paint, "frame")
        self.assertEqual(project.animations["perc"].frames, PERC_FRAMES)
        self.assertFalse(project.animations["tumble"].base)
        self.assertEqual(display_source(project, 0, {}), BODY_POSES[0])
        self.assertEqual(display_source(project, 12, {}), BODY_POSES[1])
        self.assertEqual(display_source(project, 48, {}), BODY_POSES[4])
        self.assertEqual(display_source(project, 12, {"kick": 1.0}), KICK_FRAMES[0])
        self.assertEqual(display_source(project, 12, {"snare": 1.0}), SNARE_FRAMES[0])
        self.assertEqual(display_source(project, 12, {"hat": 1.0}), HAT_FRAMES[0])
        self.assertEqual(display_source(project, 12, {"bass": 1.0}), BASS_FRAMES[1])
        self.assertEqual(display_source(project, 12, {"perc": 1.0}), PERC_FRAMES[0])
        self.assertEqual(display_source(project, 12, {"kick": 1.0, "snare": 1.0}), KICK_FRAMES[0])
        heart = heart_dancer_project()
        self.assertEqual(display_source(heart, 10, {}), 10)
        self.assertEqual(display_source(heart, 10, {"bass": 1.0}), 10)

    def test_frame_map_import_and_export_keep_the_rectangles(self):
        loaded = SpriteProject.load(SPRITES)
        exported = loaded.to_frame_map()
        self.assertEqual(validate(exported), [])
        again = SpriteProject.from_dict(exported, root=str(ASSET))
        for index in range(80):
            key = f"frame_{index:03d}"
            self.assertEqual(exported["frames"][key]["rect"], self.frame_map["frames"][key]["rect"])
            self.assertEqual(exported["frames"][key]["category"], self.frame_map["frames"][key]["category"])
            self.assertEqual(again.rect_of(index), tuple(self.frame_map["frames"][key]["rect"]))
        image = QImage(str(CONTACT))
        self.assertEqual(image.width(), IMAGE_W * 2)
        self.assertEqual(image.height(), IMAGE_H * 2)
        self.assertNotEqual(hashlib.sha256(CONTACT.read_bytes()).hexdigest(), hashlib.sha256(SHEET.read_bytes()).hexdigest())


class GreenDancerEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_the_editor_lists_every_frame_and_can_correct_a_mapping(self):
        window = SpriteEditorWindow(green_dancer_project())
        window.resize(1100, 700)
        window.show()
        self.app.processEvents()
        self.assertEqual(window.frame_list.count(), 80)
        self.assertIn("000", window.frame_list.item(0).text())
        self.assertIn("0,0", window.frame_list.item(0).text())
        self.assertIn("110×128", window.frame_list.item(0).toolTip())
        window.frame_list.item(18).setSelected(True)
        self.app.processEvents()
        self.assertEqual(window.category_box.currentData(), "arms")
        window.category_box.setCurrentIndex(window.category_box.findData("unknown"))
        self.app.processEvents()
        self.assertEqual(window.session.project.frame(18).category, "unknown")
        self.assertEqual(window.session.project.frame(18).confidence, "manual")
        window.undo()
        self.assertEqual(window.session.project.frame(18).category, "arms")
        window.rect_x.setValue(4)
        window._rect_changed()
        self.assertEqual(window.session.project.rect_of(18)[0], 4)
        window._reset_rect()
        self.assertEqual(window.session.project.rect_of(18), (220, 256, 110, 128))
        view = DancerView()
        view.apply_sprite_project(window._send_payload())
        self.assertEqual(view._sprite_project.name, "green_dancer")
        view.restore_heart()
        self.assertIsNone(view._sprite_project)
        window._open_heart_dancer()
        self.assertEqual(window.session.project.name, "heart_dancer")
        window.show_verify = True
        window.sheet_view.repaint()
        window.preview.repaint()
        self.assertIsNotNone(window.image())
        window.close()
