"""Music structure encoding (design doc §6.2.3 / §7.2).

* ``create_structure_tokens`` -- per-frame (25 Hz) structure labels:
  section / bar / beat / chord / tempo, generated from BPM + time
  signature with an Intro-Verse-Chorus-Bridge cyclic prior (§7.2).
* ``StructureConditionEncoder`` -- additive embeddings injected into every
  input position of ``MusicEmbedding`` (broadcast over the frame's Q
  codebook slots).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# section ids align with symbolic.SECTIONS (0=Intro, 1=Verse, 2=Chorus,
# 3=Bridge, 4=Outro, 5=Interlude)
SECTION_CYCLE = [0, 1, 2, 1, 2, 3, 2, 4]   # 8-bar-group cyclic prior


def create_structure_tokens(seq_frames, bpm=120, time_signature=(4, 4),
                            frame_rate=25, chords=None, sections=None,
                            device=None):
    """Generate structure labels for ``seq_frames`` audio frames.

    ``chords``   -- optional list of per-bar chord ids (0-143); zeros if None.
    ``sections`` -- optional list of per-bar section ids; cyclic prior if None.
    Returns dict of LongTensors [seq_frames].
    """
    frames_per_beat = max(1, round(frame_rate * 60 / bpm))
    frames_per_bar = frames_per_beat * time_signature[0]

    labels = {"section": [], "bar": [], "beat": [], "chord": [], "tempo": []}
    tempo_bucket = max(0, min(31, int(round((bpm - 40) / 5))))
    for f in range(seq_frames):
        bar = f // frames_per_bar
        beat = (f % frames_per_bar) // frames_per_beat
        labels["section"].append(
            sections[bar] if sections is not None and bar < len(sections)
            else SECTION_CYCLE[(bar // 4) % len(SECTION_CYCLE)])
        labels["bar"].append(bar % 128)
        labels["beat"].append(min(beat, 15))
        labels["chord"].append(
            chords[bar] if chords is not None and bar < len(chords) else 0)
        labels["tempo"].append(tempo_bucket)
    return {k: torch.tensor(v, dtype=torch.long, device=device)
            for k, v in labels.items()}


class StructureConditionEncoder(nn.Module):
    """Additive structure embeddings (§6.2.3), config-driven sizes."""

    def __init__(self, d_model: int, n_section=8, n_bar=128, n_beat=16,
                 n_chord=144, n_tempo=32):
        super().__init__()
        self.section_embed = nn.Embedding(n_section, d_model)
        self.bar_embed = nn.Embedding(n_bar, d_model)
        self.beat_embed = nn.Embedding(n_beat, d_model)
        self.chord_embed = nn.Embedding(n_chord, d_model)
        self.tempo_embed = nn.Embedding(n_tempo, d_model)

    def forward(self, structure_labels):
        """structure_labels: dict of LongTensors [B, L] -> [B, L, d]."""
        return (self.section_embed(structure_labels["section"])
                + self.bar_embed(structure_labels["bar"])
                + self.beat_embed(structure_labels["beat"])
                + self.chord_embed(structure_labels["chord"])
                + self.tempo_embed(structure_labels["tempo"]))
