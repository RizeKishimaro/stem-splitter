"""The arrangement's data, as a small database with stable IDs.

* ``TrackRegistry``: every original song, stem and segment as a node with a
  stable id and explicit relations (parent, root, source asset). A node
  refers to its source audio by asset and sample frames; audio is never
  copied per node. Nodes are deactivated, not deleted, so undo can bring
  back the very same id.
* ``ActionHistory``: the project's one undo history, saved with it. Every
  split, move, deactivation and restore is a recorded action: the nodes it
  read, made and deactivated, the exact split frames, and each touched
  node's data before and after. Every edit is an ``edit`` action holding the
  project as it was before and after. Actions of one gesture share a
  ``group`` and undo and redo together. Undo writes the "before" data back
  and deactivates what the actions made; redo writes the "after" data, so
  the same ids come back.
* ``SelectionRegistry``: the selected node ids. Selection only; it is not
  playback state and not identity.
* ``PlaybackRegistry``: which segment ids the arrangement transport is
  playing or holding paused. It does not make sound; the existing mixer
  does, filtered to these segments, on its one clock.

``Transaction`` updates all four together and puts every one back if any
step fails, so no registry is left pointing at a node another dropped.

Positions: source ranges are sample frames of the source asset (end
exclusive); arrangement positions are session ticks (see
``soniclab.core.session_timeline``).
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

TRACK_ORIGINAL = "original"  # an imported song or file
TRACK_STEM = "stem"  # one stem of a split original
TRACK_SEGMENT = "segment"  # a stretch of an original or stem (a bar, a split half, a moved piece)

PLAYING = "playing"
PAUSED = "paused"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# A block that was never split or moved has no node: its identity is the
# node it is a stretch of and its source frames, so it is the same every
# time it is computed, and no node is made for every bar.
BLOCK_PREFIX = "blk:"


def block_id(node_id: str, start_frame: int, end_frame: int) -> str:
    return f"{BLOCK_PREFIX}{node_id}:{int(start_frame)}:{int(end_frame)}"


def is_block_id(identity: str) -> bool:
    return str(identity).startswith(BLOCK_PREFIX)


def block_node(identity: str) -> str:
    return str(identity)[len(BLOCK_PREFIX):].rsplit(":", 2)[0]


def block_frames(identity: str) -> tuple[int, int]:
    _node, first, last = str(identity)[len(BLOCK_PREFIX):].rsplit(":", 2)
    return int(first), int(last)


@dataclass
class SourceAsset:
    """One audio file. ``downbeats``/``beats`` are detected events in its
    sample frames (analysis data; it never changes session bars)."""

    id: str
    path: str = ""
    rate: int = 44100
    frames: int = 0
    beats: list[int] = field(default_factory=list)
    downbeats: list[int] = field(default_factory=list)


@dataclass
class TrackNode:
    id: str
    parent_id: str | None
    root_id: str
    track_type: str
    source_asset_id: str | None
    stem_type: str | None
    source_start_frame: int
    source_end_frame: int
    arrangement_start_tick: int
    arrangement_end_tick: int
    created_by_action_id: str | None = None
    active: bool = True

    def data(self) -> dict:
        return asdict(self)


_NODE_FIELDS = {item.name for item in fields(TrackNode)}


def _node_from(data: dict) -> TrackNode:
    return TrackNode(**{key: value for key, value in data.items() if key in _NODE_FIELDS})


class TrackRegistry:
    def __init__(self):
        self.nodes: dict[str, TrackNode] = {}
        self.assets: dict[str, SourceAsset] = {}

    # ----------------------------------------------------------------- assets

    def asset_for(self, path: str, rate: int = 44100, frames: int = 0) -> SourceAsset:
        """The asset of ``path``; one per file, however many nodes use it."""
        for asset in self.assets.values():
            if path and asset.path == path:
                if frames and not asset.frames:
                    asset.frames = int(frames)
                return asset
        asset = SourceAsset(new_id("asset"), path or "", int(rate), int(frames))
        self.assets[asset.id] = asset
        return asset

    # ------------------------------------------------------------------ nodes

    def create(
        self,
        track_type: str,
        source_asset_id: str | None,
        source_start_frame: int,
        source_end_frame: int,
        arrangement_start_tick: int,
        arrangement_end_tick: int,
        parent_id: str | None = None,
        stem_type: str | None = None,
        action_id: str | None = None,
        node_id: str | None = None,
    ) -> TrackNode:
        node_id = node_id or new_id({TRACK_ORIGINAL: "song", TRACK_STEM: "stem"}.get(track_type, "seg"))
        if node_id in self.nodes and self.nodes[node_id].active:
            raise ValueError(f"duplicate active node id {node_id}")
        root = self.nodes[parent_id].root_id if parent_id in self.nodes else node_id
        node = TrackNode(
            node_id, parent_id, root, track_type, source_asset_id, stem_type,
            int(source_start_frame), int(source_end_frame),
            int(arrangement_start_tick), int(arrangement_end_tick), action_id, True,
        )
        self.nodes[node_id] = node
        return node

    def get(self, node_id: str | None) -> TrackNode | None:
        return self.nodes.get(node_id) if node_id else None

    def children(self, node_id: str, active_only: bool = True) -> list[TrackNode]:
        return [
            node for node in self.nodes.values()
            if node.parent_id == node_id and (node.active or not active_only)
        ]

    def descendants(self, node_id: str, active_only: bool = True) -> list[TrackNode]:
        """Every node below ``node_id``. The walk goes through inactive
        nodes (a split's original), so their active children are found."""
        found = []
        for child in self.children(node_id, active_only=False):
            if child.active or not active_only:
                found.append(child)
            found.extend(self.descendants(child.id, active_only))
        return found

    def query(self, track_type=None, root_id=None, parent_id=None, stem_type=None, active=True) -> list[TrackNode]:
        return [
            node for node in self.nodes.values()
            if (active is None or node.active == active)
            and (track_type is None or node.track_type == track_type)
            and (root_id is None or node.root_id == root_id)
            and (parent_id is None or node.parent_id == parent_id)
            and (stem_type is None or node.stem_type == stem_type)
        ]

    def segment_for(self, parent_id: str, start_frame: int, end_frame: int, start_tick: int, end_tick: int) -> TrackNode:
        """The active segment of ``parent_id`` over exactly these source
        frames, made the first time it is needed (a bar that is selected or
        played). One per range, so a block never has two ids."""
        # Any active segment below the parent over this exact range: a half a
        # split action made keeps the id it was given.
        for node in self.descendants(parent_id):
            if node.track_type == TRACK_SEGMENT and node.source_start_frame == int(start_frame) and node.source_end_frame == int(end_frame):
                node.arrangement_start_tick, node.arrangement_end_tick = int(start_tick), int(end_tick)
                return node
        parent = self.nodes[parent_id]
        return self.create(
            TRACK_SEGMENT, parent.source_asset_id, start_frame, end_frame, start_tick, end_tick,
            parent_id=parent_id, stem_type=parent.stem_type,
        )

    # ---------------------------------------------------------------- persist

    def to_dict(self) -> dict:
        return {
            "nodes": [node.data() for node in self.nodes.values()],
            "assets": [asdict(asset) for asset in self.assets.values()],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> TrackRegistry:
        registry = cls()
        for raw in (data or {}).get("assets") or ():
            asset = SourceAsset(
                str(raw["id"]), str(raw.get("path") or ""), int(raw.get("rate") or 44100), int(raw.get("frames") or 0),
                [int(f) for f in raw.get("beats") or ()], [int(f) for f in raw.get("downbeats") or ()],
            )
            registry.assets[asset.id] = asset
        for raw in (data or {}).get("nodes") or ():
            node = _node_from(raw)
            registry.nodes[node.id] = node
        return registry


@dataclass
class Action:
    """One recorded change. ``before``/``after`` hold every touched node's
    data (None in ``before`` for a node the action made)."""

    id: str
    kind: str
    timestamp: float
    parent_action_id: str | None = None
    originals: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    deactivated: list[str] = field(default_factory=list)
    split_frames: list[int] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    group: str | None = None  # one gesture: undone and redone together
    snapshot_before: dict | None = None  # an "edit": the project before it
    snapshot_after: dict | None = None  # and after it


# Older actions are dropped past this many, a whole gesture at a time.
HISTORY_LIMIT = 400


class ActionHistory:
    def __init__(self):
        self.actions: list[Action] = []
        self.done = 0  # actions[:done] are applied; the rest can be redone
        self.group: str | None = None  # the gesture being recorded, if any

    def _record(self, action: Action) -> Action:
        del self.actions[self.done:]  # a new action drops the redo branch
        if action.group is None:
            action.group = self.group
        self.actions.append(action)
        self.done = len(self.actions)
        while len(self.actions) > HISTORY_LIMIT:
            first = self.actions[0].group
            count = 1
            while count < len(self.actions) and first is not None and self.actions[count].group == first:
                count += 1
            del self.actions[:count]
            self.done -= count
        return action

    def begin(self) -> str:
        """Start a gesture: the actions recorded until ``end`` are one undo step."""
        self.group = new_id("gesture")
        return self.group

    def end(self) -> None:
        self.group = None

    def record_edit(self, snapshot_before: dict, snapshot_after: dict) -> Action:
        """The project before and after one edit (in the current gesture)."""
        return self._record(Action(
            new_id("act"), "edit", time.time(),
            snapshot_before=copy.deepcopy(snapshot_before), snapshot_after=copy.deepcopy(snapshot_after),
        ))

    def recorded_since(self, done: int) -> int:
        return self.done - int(done)

    @staticmethod
    def _apply(registry: TrackRegistry, states: dict) -> None:
        for node_id, data in states.items():
            if data is None:
                node = registry.nodes.get(node_id)
                if node is not None:
                    node.active = False
            else:
                registry.nodes[node_id] = _node_from(data)

    @property
    def can_undo(self) -> bool:
        return self.done > 0

    @property
    def can_redo(self) -> bool:
        return self.done < len(self.actions)

    def undo(self, registry: TrackRegistry) -> list[Action]:
        """Undo the last gesture (all its actions, last first); the actions
        undone, for the caller to restore the project from."""
        undone = []
        while self.can_undo:
            action = self.actions[self.done - 1]
            if undone and (action.group is None or action.group != undone[0].group):
                break
            self.done -= 1
            self._apply(registry, action.before)
            undone.append(action)
        return undone

    def redo(self, registry: TrackRegistry) -> list[Action]:
        """Redo the next gesture with the same ids and data."""
        redone = []
        while self.can_redo:
            action = self.actions[self.done]
            if redone and (action.group is None or action.group != redone[0].group):
                break
            self.done += 1
            self._apply(registry, action.after)
            redone.append(action)
        return redone

    def managed(self) -> set[str]:
        """Ids that applied actions made or replaced: their state is the
        history's, not derived from anything else."""
        found = set()
        for action in self.actions[: self.done]:
            found.update(action.originals)
            found.update(action.created)
        return found

    def last(self) -> Action | None:
        return self.actions[self.done - 1] if self.done else None

    # ------------------------------------------------------- the operations

    def split(self, registry: TrackRegistry, node_id: str, frames, frame_to_tick, parent_action_id=None) -> Action:
        """Split ``node_id`` at source ``frames``: it is deactivated and one
        child segment per stretch replaces it, on the same arrangement
        place (``frame_to_tick`` maps the node's source frames to ticks)."""
        node = registry.nodes[node_id]
        cuts = sorted({int(f) for f in frames if node.source_start_frame < int(f) < node.source_end_frame})
        if not cuts:
            raise ValueError("no split frame inside the node")
        action = Action(new_id("act"), "split", time.time(), parent_action_id, [node_id], split_frames=cuts)
        action.before[node_id] = node.data()
        edges = [node.source_start_frame, *cuts, node.source_end_frame]
        for first, last in zip(edges, edges[1:]):
            child = registry.create(
                TRACK_SEGMENT, node.source_asset_id, first, last,
                int(frame_to_tick(first)), int(frame_to_tick(last)),
                parent_id=node_id, stem_type=node.stem_type, action_id=action.id,
            )
            action.created.append(child.id)
            action.before[child.id] = None
            action.after[child.id] = child.data()
        node.active = False
        action.deactivated.append(node_id)
        action.after[node_id] = node.data()
        return self._record(action)

    def move(self, registry: TrackRegistry, node_ids, delta_ticks: int) -> Action:
        """Move nodes by ``delta_ticks`` together, keeping their offsets.
        Only their arrangement place changes."""
        action = Action(new_id("act"), "move", time.time(), None, list(node_ids))
        for node_id in node_ids:
            node = registry.nodes[node_id]
            action.before[node_id] = node.data()
            node.arrangement_start_tick += int(delta_ticks)
            node.arrangement_end_tick += int(delta_ticks)
            action.after[node_id] = node.data()
        return self._record(action)

    def set_active(self, registry: TrackRegistry, node_ids, active: bool) -> Action:
        """Deactivate (or restore) nodes and their descendants."""
        kind = "restore" if active else "deactivate"
        action = Action(new_id("act"), kind, time.time(), None, list(node_ids))
        touched = []
        for node_id in node_ids:
            touched.append(node_id)
            if not active:
                touched.extend(child.id for child in registry.descendants(node_id))
        for node_id in touched:
            node = registry.nodes[node_id]
            action.before[node_id] = node.data()
            node.active = bool(active)
            action.after[node_id] = node.data()
            if not active:
                action.deactivated.append(node_id)
        return self._record(action)

    # ---------------------------------------------------------------- persist

    def to_dict(self) -> dict:
        return {"done": self.done, "actions": [asdict(action) for action in self.actions]}

    @classmethod
    def from_dict(cls, data: dict | None) -> ActionHistory:
        history = cls()
        known = {item.name for item in fields(Action)}
        for raw in (data or {}).get("actions") or ():
            history.actions.append(Action(**{key: value for key, value in raw.items() if key in known}))
        history.done = min(len(history.actions), int((data or {}).get("done", len(history.actions))))
        return history


class SelectionRegistry:
    """The selected node ids, in the order they were selected."""

    def __init__(self, ids=()):
        self.ids: list[str] = list(dict.fromkeys(ids))

    def __contains__(self, node_id) -> bool:
        return node_id in self.ids

    def select(self, node_id: str) -> None:
        if node_id not in self.ids:
            self.ids.append(node_id)

    def unselect(self, node_id: str) -> None:
        if node_id in self.ids:
            self.ids.remove(node_id)

    def toggle(self, node_id: str) -> bool:
        if node_id in self.ids:
            self.ids.remove(node_id)
            return False
        self.ids.append(node_id)
        return True

    def clear(self) -> None:
        self.ids.clear()

    def select_children(self, registry: TrackRegistry, parent_id: str) -> None:
        """A parent's active segments, each selected on its own id."""
        for child in registry.descendants(parent_id):
            if child.track_type == TRACK_SEGMENT:
                self.select(child.id)

    def replace(self, ids) -> None:
        self.ids = list(dict.fromkeys(ids))

    def reconcile(self, registry: TrackRegistry) -> None:
        """Forget ids whose node is gone or inactive. A logical block id
        (see ``block_id``) lives while the node it is a stretch of does."""
        def alive(identity: str) -> bool:
            node = registry.get(block_node(identity) if is_block_id(identity) else identity)
            return node is not None and node.active

        self.ids = [identity for identity in self.ids if alive(identity)]


class PlaybackRegistry:
    """Segment ids the arrangement transport plays (or holds paused) as a
    filtered selection. A record, not a voice: the mixer plays them."""

    def __init__(self):
        self.states: dict[str, str] = {}

    def playing(self) -> list[str]:
        return [node_id for node_id, state in self.states.items() if state == PLAYING]

    def start(self, ids) -> None:
        self.states = {node_id: PLAYING for node_id in dict.fromkeys(ids)}

    def pause(self) -> None:
        self.states = {node_id: PAUSED for node_id in self.states}

    def resume(self) -> None:
        self.states = {node_id: PLAYING for node_id in self.states}

    def remove(self, node_id: str) -> None:
        self.states.pop(node_id, None)

    def stop(self) -> None:
        self.states.clear()

    def reconcile(self, registry: TrackRegistry, selection: SelectionRegistry) -> None:
        """A segment that is inactive or no longer selected stops playing.
        A logical block lives while the node it is a stretch of does."""
        def alive(identity: str) -> bool:
            node = registry.get(block_node(identity) if is_block_id(identity) else identity)
            return node is not None and node.active

        self.states = {
            identity: state for identity, state in self.states.items()
            if alive(identity) and identity in selection
        }

    def reconcile_selection(self, selection: SelectionRegistry) -> None:
        """What left the selection leaves playback (its record; the mixer
        filter follows from the same selection)."""
        self.states = {identity: state for identity, state in self.states.items() if identity in selection}


class Transaction:
    """Change the track, selection and playback registries and the history
    together; on an exception all four are put back as they were."""

    def __init__(self, registry: TrackRegistry, selection: SelectionRegistry, playback: PlaybackRegistry, history: ActionHistory):
        self.parts = (registry, selection, playback, history)
        self._saved = None

    def __enter__(self):
        self._saved = [copy.deepcopy(part.__dict__) for part in self.parts]
        return self

    def __exit__(self, kind, _value, _traceback):
        if kind is not None:
            for part, saved in zip(self.parts, self._saved):
                part.__dict__.clear()
                part.__dict__.update(saved)
            return False
        registry, selection, playback, _history = self.parts
        selection.reconcile(registry)
        playback.reconcile(registry, selection)
        return False
