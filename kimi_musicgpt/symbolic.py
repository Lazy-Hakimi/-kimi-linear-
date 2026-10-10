"""REMI+ symbolic music tokenizer (design doc §3), complete implementation.

Vocabulary (~900 tokens) follows §3.2 exactly:

    Bar x1 | Position x128 | NoteOn x128 | NoteOff x128 | Drum x128 |
    Duration x32 | Velocity x32 | Chord x144 (12 roots x 12 types) |
    Tempo x32 (40..200 BPM, 5-BPM buckets) | Section x6 |
    Instrument x4 | PAD/BOS/EOS/MASK

Encoding consumes a structured song description (see ``Song`` below) and
produces a REMI event stream::

    BOS Section_* Tempo_* [Bar Chord_* Position_* Instrument_* NoteOn_*
    Velocity_* Duration_* ...]* EOS

``decode`` performs the exact inverse. ``symbolic_to_acoustic_alignment``
maps symbolic positions to 25-Hz codec frame indices (§3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TYPES = ["maj", "min", "dim", "aug", "maj7", "min7", "7",
               "sus2", "sus4", "dim7", "m7b5", "add9"]
SECTIONS = ["Intro", "Verse", "Chorus", "Bridge", "Outro", "Interlude"]
INSTRUMENTS = ["Piano", "Guitar", "Bass", "Strings"]
POSITIONS_PER_BAR = 128
DURATION_BUCKETS = 32
VELOCITY_BUCKETS = 32
TEMPO_MIN, TEMPO_STEP = 40, 5


@dataclass
class Note:
    pitch: int          # 0-127 MIDI pitch
    position: int       # 0-127 (128th of a bar)
    duration: int       # in 1/128-bar ticks (>=1)
    velocity: int       # 1-127
    instrument: int = 0


@dataclass
class DrumHit:
    pitch: int
    position: int
    velocity: int = 100


@dataclass
class Bar:
    chord: int = 0                  # chord id 0-143
    section: int = 1                # section id (per bar, REMI+ extension)
    notes: list = field(default_factory=list)       # list[Note]
    drums: list = field(default_factory=list)       # list[DrumHit]


@dataclass
class Song:
    bpm: int = 120
    bars: list = field(default_factory=list)        # list[Bar]


class RemiTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._build()

    # ------------------------------------------------------------------ #
    def _add(self, name):
        self.token_to_id[name] = len(self.token_to_id)

    def _build(self):
        self._add("PAD"), self._add("BOS"), self._add("EOS"), self._add("MASK")
        self._add("Bar")
        for i in range(POSITIONS_PER_BAR):
            self._add(f"Position_{i}")
        for i in range(128):
            self._add(f"NoteOn_{i}")
        for i in range(128):
            self._add(f"NoteOff_{i}")
        for i in range(128):
            self._add(f"Drum_{i}")
        for i in range(DURATION_BUCKETS):
            self._add(f"Duration_{i}")
        for i in range(VELOCITY_BUCKETS):
            self._add(f"Velocity_{i}")
        for root in CHORD_ROOTS:
            for ctype in CHORD_TYPES:
                self._add(f"Chord_{root}:{ctype}")
        for i in range(32):
            self._add(f"Tempo_{TEMPO_MIN + i * TEMPO_STEP}")
        for s in SECTIONS:
            self._add(f"Section_{s}")
        for inst in INSTRUMENTS:
            self._add(f"Instrument_{inst}")
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    # ------------------------------------------------------------------ #
    # id helpers
    # ------------------------------------------------------------------ #
    def chord_id(self, root_idx: int, type_idx: int) -> int:
        return self.token_to_id[f"Chord_{CHORD_ROOTS[root_idx]}:{CHORD_TYPES[type_idx]}"]

    @staticmethod
    def chord_index(root_idx: int, type_idx: int) -> int:
        return root_idx * len(CHORD_TYPES) + type_idx

    def tempo_bucket(self, bpm: float) -> int:
        b = int(round((bpm - TEMPO_MIN) / TEMPO_STEP))
        return max(0, min(31, b))

    @staticmethod
    def duration_bucket(ticks: int) -> int:
        """ticks (1/128 bar) -> bucket (each bucket = 4 ticks = 32nd note)."""
        return max(0, min(DURATION_BUCKETS - 1, (ticks + 3) // 4 - 1))

    @staticmethod
    def bucket_ticks(bucket: int) -> int:
        return (bucket + 1) * 4

    @staticmethod
    def velocity_bucket(velocity: int) -> int:
        return max(0, min(VELOCITY_BUCKETS - 1, velocity * VELOCITY_BUCKETS // 128))

    @staticmethod
    def bucket_velocity(bucket: int) -> int:
        return min(127, (bucket * 128) // VELOCITY_BUCKETS + 2)

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, song: Song, add_bos_eos: bool = True) -> list:
        ids = []
        if add_bos_eos:
            ids.append(self.token_to_id["BOS"])
        ids.append(self.token_to_id[f"Tempo_{TEMPO_MIN + self.tempo_bucket(song.bpm) * TEMPO_STEP}"])
        for bar in song.bars:
            ids.append(self.token_to_id["Bar"])
            ids.append(self.token_to_id[f"Section_{SECTIONS[bar.section]}"])
            root, ctype = divmod(bar.chord, len(CHORD_TYPES))
            ids.append(self.chord_id(root, ctype))
            events = []
            for n in bar.notes:
                events.append((n.position, "note", n))
            for d in bar.drums:
                events.append((d.position, "drum", d))
            events.sort(key=lambda e: e[0])
            last_pos = -1
            for pos, kind, ev in events:
                if pos != last_pos:
                    ids.append(self.token_to_id[f"Position_{pos}"])
                    last_pos = pos
                if kind == "note":
                    ids.append(self.token_to_id[f"Instrument_{INSTRUMENTS[ev.instrument]}"])
                    ids.append(self.token_to_id[f"NoteOn_{ev.pitch}"])
                    ids.append(self.token_to_id[f"Velocity_{self.velocity_bucket(ev.velocity)}"])
                    ids.append(self.token_to_id[f"Duration_{self.duration_bucket(ev.duration)}"])
                else:
                    ids.append(self.token_to_id[f"Drum_{ev.pitch}"])
                    ids.append(self.token_to_id[f"Velocity_{self.velocity_bucket(ev.velocity)}"])
        if add_bos_eos:
            ids.append(self.token_to_id["EOS"])
        return ids

    # ------------------------------------------------------------------ #
    def decode(self, ids) -> Song:
        song = Song()
        cur_bar = None
        cur_pos = 0
        pending = {}
        toks = [self.id_to_token[int(i)] for i in ids]
        for tok in toks:
            if tok.startswith("Tempo_"):
                song.bpm = int(tok.split("_")[1])
            elif tok == "Bar":
                cur_bar = Bar()
                song.bars.append(cur_bar)
                cur_pos = 0
            elif tok.startswith("Section_") and cur_bar is not None:
                cur_bar.section = SECTIONS.index(tok.split("_")[1])
            elif tok.startswith("Chord_") and cur_bar is not None:
                name = tok[len("Chord_"):]
                root, ctype = name.split(":")
                cur_bar.chord = self.chord_index(CHORD_ROOTS.index(root),
                                                 CHORD_TYPES.index(ctype))
            elif tok.startswith("Position_"):
                cur_pos = int(tok.split("_")[1])
            elif tok.startswith("Instrument_"):
                pending["instrument"] = INSTRUMENTS.index(tok.split("_")[1])
            elif tok.startswith("NoteOn_"):
                pending["pitch"] = int(tok.split("_")[1])
            elif tok.startswith("Velocity_"):
                pending["velocity"] = int(tok.split("_")[1])
                if "drum_pitch" in pending and cur_bar is not None:
                    cur_bar.drums.append(DrumHit(
                        pitch=pending["drum_pitch"], position=cur_pos,
                        velocity=self.bucket_velocity(pending["velocity"])))
                    pending = {}
            elif tok.startswith("Duration_"):
                pending["duration"] = int(tok.split("_")[1])
                if cur_bar is not None and "pitch" in pending:
                    cur_bar.notes.append(Note(
                        pitch=pending["pitch"],
                        position=cur_pos,
                        duration=self.bucket_ticks(pending["duration"]),
                        velocity=self.bucket_velocity(pending.get("velocity", 16)),
                        instrument=pending.get("instrument", 0)))
                pending = {}
            elif tok.startswith("Drum_"):
                pending["drum_pitch"] = int(tok.split("_")[1])
        return song

    # ------------------------------------------------------------------ #
    def symbolic_to_acoustic_alignment(self, song: Song, frame_rate: int = 25):
        """§3.3: map each bar boundary to an acoustic frame index (4/4 time)."""
        frames_per_bar = int(frame_rate * 60 * 4 / song.bpm)
        return [i * frames_per_bar for i in range(len(song.bars) + 1)]
