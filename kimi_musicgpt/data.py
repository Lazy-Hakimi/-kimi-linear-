"""Training data pipeline (design doc §9.1.2): a real music synthesizer +
dataset producing aligned (waveform, structure, symbolic, text) tuples.

``MusicSynthesizer`` renders genuinely structured songs -- key, chord
progression, section form (Intro/Verse/Chorus/...), melody over chord
tones, bass line, synthesized drums (kick sine-sweep, snare noise+tone,
hi-hat filtered noise) with ADSR envelopes -- so every stage of the stack
(codec training, GPT training, symbolic tokenization, text conditioning)
trains on real audio with real musical structure, not noise.

``SynthMusicDataset`` yields dicts with the waveform and all metadata the
training scripts need; codec tokenization happens in the training loop
(batched, on-device).
"""

from __future__ import annotations

import random

import torch

from .symbolic import (Song, Bar, Note, DrumHit, RemiTokenizer,
                       CHORD_ROOTS, CHORD_TYPES, SECTIONS)

# chord type -> semitone offsets from root
_CHORD_INTERVALS = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "7": [0, 4, 7, 10],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10], "add9": [0, 4, 7, 14],
}
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# common progressions as scale degrees (major) / (natural minor)
_PROGRESSIONS = {
    "pop": ([0, 4, 5, 3], "maj"),          # I V vi IV
    "ballad": ([0, 5, 3, 4], "maj"),       # I vi IV V
    "rock": ([0, 3, 4, 4], "maj"),         # I IV V V
    "folk": ([0, 4, 0, 5], "maj"),         # I V I vi
    "blues": ([0, 0, 3, 4], "maj"),        # I I IV V
    "sad": ([5, 3, 0, 4], "min"),          # i VI III VII (minor)
}
STYLES = ["pop", "ballad", "rock", "folk", "blues", "sad"]

# General-MIDI-ish drum pitches
KICK, SNARE, HIHAT = 36, 38, 42


def midi_to_freq(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def adsr(n_samples: int, sample_rate: int, attack=0.01, decay=0.08,
         sustain=0.6, release=0.1):
    """Classic ADSR envelope, length n_samples."""
    a = max(1, int(attack * sample_rate))
    d = max(1, int(decay * sample_rate))
    r = max(1, int(release * sample_rate))
    s = max(0, n_samples - a - d - r)
    env = torch.cat([
        torch.linspace(0, 1, a),
        torch.linspace(1, sustain, d),
        torch.full((s,), sustain),
        torch.linspace(sustain, 0, r),
    ])[:n_samples]
    if env.numel() < n_samples:
        env = torch.cat([env, torch.zeros(n_samples - env.numel())])
    return env


def oscillator(kind: str, freq: torch.Tensor, t: torch.Tensor):
    phase = 2 * torch.pi * freq * t
    if kind == "sine":
        return torch.sin(phase)
    if kind == "saw":
        return 2.0 * ((freq * t) % 1.0) - 1.0
    if kind == "square":
        return torch.sign(torch.sin(phase))
    raise ValueError(kind)


class MusicSynthesizer:
    """Render structured songs to waveform + aligned metadata."""

    def __init__(self, sample_rate: int = 16000, frame_rate: int = 25):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.remi = RemiTokenizer()

    # ------------------------------------------------------------------ #
    # instrument voices
    # ------------------------------------------------------------------ #
    def _voice(self, instrument: int, freq: float, dur: float, velocity: float):
        sr = self.sample_rate
        n = int(dur * sr)
        t = torch.arange(n, dtype=torch.float32) / sr
        amp = (velocity / 127.0) ** 1.5
        name = ["Piano", "Guitar", "Bass", "Strings"][instrument]
        if name == "Piano":
            env = adsr(n, sr, 0.004, 0.25, 0.25, 0.15)
            sig = (torch.sin(2 * torch.pi * freq * t)
                   + 0.5 * torch.sin(2 * torch.pi * 2 * freq * t)
                   + 0.25 * torch.sin(2 * torch.pi * 3 * freq * t))
        elif name == "Guitar":
            env = adsr(n, sr, 0.005, 0.12, 0.5, 0.1)
            sig = oscillator("saw", freq, t) * 0.7 \
                + 0.3 * torch.sin(2 * torch.pi * freq * t)
        elif name == "Bass":
            env = adsr(n, sr, 0.008, 0.1, 0.7, 0.08)
            sig = 0.8 * torch.sin(2 * torch.pi * freq * t) \
                + 0.2 * oscillator("square", freq, t)
        else:  # Strings
            env = adsr(n, sr, 0.08, 0.1, 0.8, 0.25)
            sig = (torch.sin(2 * torch.pi * freq * t)
                   + 0.6 * torch.sin(2 * torch.pi * (freq * 1.003) * t)
                   + 0.4 * torch.sin(2 * torch.pi * 0.5 * freq * t))
        return amp * env * sig / 1.5

    def _drum(self, kind: int, dur: float, velocity: float):
        sr = self.sample_rate
        n = max(1, int(dur * sr))
        t = torch.arange(n, dtype=torch.float32) / sr
        amp = (velocity / 127.0) ** 1.5
        if kind == KICK:
            f0, f1 = 120.0, 45.0
            sweep = f1 + (f0 - f1) * torch.exp(-t * 30)
            phase = 2 * torch.pi * torch.cumsum(sweep, 0) / sr
            env = torch.exp(-t * 12)
            return amp * env * torch.sin(phase)
        if kind == SNARE:
            noise = torch.randn(n)
            env = torch.exp(-t * 25)
            tone = torch.sin(2 * torch.pi * 180 * t) * torch.exp(-t * 40)
            return amp * (0.7 * noise * env + 0.3 * tone)
        # hi-hat: high-frequency modulated noise, very short
        noise = torch.randn(n)
        env = torch.exp(-t * 90)
        carrier = torch.sign(torch.sin(2 * torch.pi * 8000 * t))
        return amp * 0.4 * noise * carrier * env

    # ------------------------------------------------------------------ #
    # song generation
    # ------------------------------------------------------------------ #
    def compose(self, rng: random.Random, n_bars: int = 8):
        """-> Song with coherent key/progression/sections/melody/drums."""
        style = rng.choice(STYLES)
        degrees, mode = _PROGRESSIONS[style]
        key = rng.randint(0, 11)
        scale = _MAJOR_SCALE if mode == "maj" else _MINOR_SCALE
        bpm = rng.randint(80, 140)
        bars_per_section = 4

        song = Song(bpm=bpm)
        sections = []
        for i in range(n_bars):
            sec_group = (i // bars_per_section)
            if n_bars >= 8:
                sec = [0, 1, 2, 1, 2, 4][min(sec_group, 5)] \
                    if i < n_bars - bars_per_section else 4
            else:
                sec = 1
            sections.append(sec)
            degree = degrees[i % len(degrees)]
            root = (key + scale[degree % 7]) % 12
            ctype = "min" if (mode == "maj" and degree in (1, 2, 5)) or \
                    (mode == "min" and degree in (0, 3, 4)) else "maj"
            chord_id = RemiTokenizer.chord_index(CHORD_ROOTS.index(CHORD_ROOTS[root]),
                                                 CHORD_TYPES.index(ctype))
            bar = Bar(chord=chord_id, section=sec)
            chord_root_midi = 48 + root
            intervals = _CHORD_INTERVALS[ctype]

            # harmony pad: chord tones, whole bar
            for iv in intervals:
                bar.notes.append(Note(pitch=chord_root_midi + iv, position=0,
                                      duration=128, velocity=52,
                                      instrument=3))
            # bass: root on each beat
            for beat in range(4):
                bar.notes.append(Note(pitch=chord_root_midi - 12, position=beat * 32,
                                      duration=28, velocity=92, instrument=2))
            # melody: chord/scale tones on an 8th/16th grid, denser in chorus
            density = 0.75 if sec == 2 else 0.55
            pos = 0
            while pos < 128:
                if rng.random() < density:
                    tone_pool = [chord_root_midi + 12 + iv for iv in intervals] + \
                                [chord_root_midi + 12 + s for s in scale]
                    pitch = rng.choice(tone_pool) + 12
                    dur = rng.choice([8, 8, 16, 16, 32])
                    inst = 0 if rng.random() < 0.7 else 1
                    bar.notes.append(Note(pitch=pitch, position=pos,
                                          duration=dur,
                                          velocity=rng.randint(70, 110),
                                          instrument=inst))
                pos += 8
            # drums
            for beat in range(4):
                bar.drums.append(DrumHit(KICK, beat * 32, 110 if beat % 2 == 0 else 90))
                bar.drums.append(DrumHit(SNARE, beat * 32 + 16, 100) if beat % 2 == 1
                                 else DrumHit(HIHAT, beat * 32 + 16, 70))
                bar.drums.append(DrumHit(HIHAT, beat * 32 + 8, 55))
                bar.drums.append(DrumHit(HIHAT, beat * 32 + 24, 55))
            song.bars.append(bar)

        desc = (f"a {bpm} bpm {style} song in {CHORD_ROOTS[key]} "
                f"{'major' if mode == 'maj' else 'minor'} with "
                f"{' '.join(sorted(set(SECTIONS[s] for s in sections), reverse=True))} "
                f"sections")
        return song, {"style": style, "style_id": STYLES.index(style),
                      "key": key, "mode": mode, "bpm": bpm,
                      "description": desc}

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(self, song: Song):
        """Song -> waveform [1, T] float32 at self.sample_rate."""
        sr = self.sample_rate
        sec_per_beat = 60.0 / song.bpm
        sec_per_bar = sec_per_beat * 4
        total = sec_per_bar * len(song.bars) + 0.3
        mix = torch.zeros(int(total * sr))
        tick = sec_per_bar / 128.0
        for i, bar in enumerate(song.bars):
            bar_start = i * sec_per_bar
            for n in bar.notes:
                start = bar_start + n.position * tick
                dur = max(n.duration * tick, 0.05)
                sig = self._voice(n.instrument, midi_to_freq(n.pitch), dur,
                                  n.velocity)
                i0 = int(start * sr)
                if i0 + sig.numel() > mix.numel():
                    sig = sig[:mix.numel() - i0]
                if i0 < mix.numel() and sig.numel() > 0:
                    mix[i0:i0 + sig.numel()] += sig * 0.5
            for d in bar.drums:
                start = bar_start + d.position * tick
                sig = self._drum(d.pitch, 0.25, d.velocity)
                i0 = int(start * sr)
                if i0 + sig.numel() > mix.numel():
                    sig = sig[:mix.numel() - i0]
                if i0 < mix.numel() and sig.numel() > 0:
                    mix[i0:i0 + sig.numel()] += sig
        peak = mix.abs().max().clamp(min=1e-6)
        return (mix / peak * 0.9).unsqueeze(0)

    # ------------------------------------------------------------------ #
    def structure_labels(self, song: Song):
        """Per-bar chords/sections lists for create_structure_tokens."""
        return ([b.chord for b in song.bars], [b.section for b in song.bars])

    def frames_per_bar(self, bpm: int) -> int:
        return max(1, round(self.frame_rate * 60 * 4 / bpm))


class SynthMusicDataset(torch.utils.data.Dataset):
    """Deterministic synthetic corpus: item i is seeded by (seed, i).

    ``prerender=True`` renders the whole corpus once up-front (rendering is
    CPU-bound; caching it keeps the training loop GPU-bound)."""

    def __init__(self, size: int, sample_rate: int = 16000, frame_rate: int = 25,
                 n_bars=(4, 8), seed: int = 1234, prerender: bool = True):
        self.size = size
        self.synth = MusicSynthesizer(sample_rate, frame_rate)
        self.n_bars = n_bars
        self.seed = seed
        self._cache = None
        if prerender:
            self._cache = [self._make(i) for i in range(size)]

    def __len__(self):
        return self.size

    def _make(self, idx):
        rng = random.Random(self.seed * 1000003 + idx)
        n_bars = rng.randint(*self.n_bars)
        song, meta = self.synth.compose(rng, n_bars)
        wav = self.synth.render(song)
        chords, sections = self.synth.structure_labels(song)
        remi_ids = self.synth.remi.encode(song)
        return {"waveform": wav, "song": song, "meta": meta,
                "chords": chords, "sections": sections,
                "remi_ids": remi_ids, "text": meta["description"],
                "style_id": meta["style_id"], "bpm": meta["bpm"]}

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._make(idx)


# --------------------------------------------------------------------------- #
# Production dataset: real WAV files on disk
# --------------------------------------------------------------------------- #
class WavMusicDataset(torch.utils.data.Dataset):
    """Directory of WAV files -> the same item schema as SynthMusicDataset.

    Layout: any ``*.wav`` under ``root`` (recursively).  Optional sidecar
    ``<name>.txt`` next to a WAV supplies the text description (one line);
    optional ``<name>.json`` may set ``bpm`` / ``style_id`` / ``chords`` /
    ``sections`` (lists per bar).  Missing metadata falls back to honest
    defaults (bpm=120, style 0, neutral chords/sections) so unlabelled
    corpora still train; conditioning accuracy improves as sidecars are added.

    Clips longer than ``clip_seconds`` are randomly cropped per epoch
    (``random_crop=True``) -- the standard streaming-friendly augmentation.
    """

    def __init__(self, root: str, sample_rate: int = 16000, frame_rate: int = 25,
                 clip_seconds: float = 0.0, random_crop: bool = True):
        import glob
        import json as _json
        self.root = root
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.clip_samples = int(clip_seconds * sample_rate) if clip_seconds > 0 else 0
        self.random_crop = random_crop
        self._json = _json
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.wav"),
                                      recursive=True))
        if not self.files:
            raise FileNotFoundError(f"no .wav files under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from .audio_io import read_wav, resample_linear, normalize_loudness
        path = self.files[idx]
        wav, sr = read_wav(path)                            # [1, T]
        if sr != self.sample_rate:
            wav = resample_linear(wav, sr, self.sample_rate)
        wav = normalize_loudness(wav)

        meta = {"bpm": 120, "style_id": 0, "chords": None, "sections": None}
        side_json = os.path.splitext(path)[0] + ".json"
        side_txt = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(side_json):
            with open(side_json, "r", encoding="utf-8") as f:
                meta.update(self._json.load(f))
        if os.path.exists(side_txt):
            with open(side_txt, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = f"a {meta['bpm']} bpm song (style {meta['style_id']})"

        if self.clip_samples > 0 and wav.shape[-1] > self.clip_samples:
            if self.random_crop:
                start = random.randint(0, wav.shape[-1] - self.clip_samples)
            else:
                start = 0
            wav = wav[..., start:start + self.clip_samples]

        return {"waveform": wav, "song": None, "meta": meta,
                "chords": meta["chords"], "sections": meta["sections"],
                "remi_ids": None, "text": text,
                "style_id": meta["style_id"], "bpm": meta["bpm"],
                "path": path}
