import json
from pathlib import Path

class Project:
    def __init__(self):
        self.name = "Untitled"
        self.bpm = 120.0
        self.time_signature = [4, 4]
        # Bar-line times in seconds. Empty means bars follow the drum track,
        # or the tempo when there are no drum downbeats.
        self.bars = []
        self.tracks = []
        # The session's musical timeline (SessionTimeline.to_dict()), the
        # authority for bars and tempo. ``bars`` above is kept beside it as
        # bar-line seconds for older SonicLab builds; None in a project saved
        # before the timeline existed, which is migrated when it is opened.
        self.timeline = None
        # The Track Registry and action history (see track_registry): stable
        # ids for songs, stems and segments. None before it existed.
        self.registry = None

    def to_dict(self):
        return {
            "version": 1,
            "name": self.name,
            "bpm": self.bpm,
            "time_signature": self.time_signature,
            "bars": self.bars,
            "tracks": self.tracks,
            **({"timeline": self.timeline} if self.timeline else {}),
            **({"registry": self.registry} if self.registry else {}),
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        project = cls()
        project.name = data.get("name", "Untitled")
        project.bpm = float(data.get("bpm", 120.0))
        project.time_signature = data.get("time_signature", [4, 4])
        project.bars = [float(value) for value in data.get("bars", []) or []]
        project.tracks = data.get("tracks", [])
        project.timeline = data.get("timeline") or None
        project.registry = data.get("registry") or None
        return project
