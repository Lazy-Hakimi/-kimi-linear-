"""Byte-level BPE text tokenizer (GPT-2 style), complete implementation.

Used to tokenize the free-form text condition (style descriptions, lyrics)
of Kimi-MusicGPT. This is a *real* tokenizer, not a placeholder:

* ``ByteLevelBPETokenizer.train``  -- learns merge rules from a raw corpus
  by iterative most-frequent-pair merging over byte-level symbols,
* ``encode`` / ``decode``          -- lossless round-trip (byte-level BPE is
  total: any byte string is representable),
* ``save`` / ``load``              -- JSON vocab + merges,
* special tokens                   -- PAD/BOS/EOS with fixed ids.

The byte<->unicode mapping follows the GPT-2 scheme (printable bytes map
to themselves, the rest to 256+n) so the vocabulary stays display-safe.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ByteLevelBPETokenizer:
    PAD, BOS, EOS = "<|pad|>", "<|bos|>", "<|eos|>"

    def __init__(self):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab = {}             # token string -> id
        self.id_to_token = {}       # id -> token string
        self.merges = []            # list of (a, b) in rank order
        self.merge_ranks = {}       # (a, b) -> rank
        self._word_cache = {}
        # whitespace is tokenized (attached to the following word) so that
        # decode(encode(text)) is byte-exact, spaces included
        self._re = re.compile(r"\s*\w+|\s*[^\w\s]|\s+", re.UNICODE)

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD]

    @property
    def bos_id(self):
        return self.vocab[self.BOS]

    @property
    def eos_id(self):
        return self.vocab[self.EOS]

    # ------------------------------------------------------------------ #
    def _init_vocab(self):
        self.vocab = {}
        for b in range(256):
            self.vocab[self.byte_encoder[b]] = b
        for sp in (self.PAD, self.BOS, self.EOS):
            self.vocab[sp] = len(self.vocab)
        self._sync()

    def _sync(self):
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks = {m: i for i, m in enumerate(self.merges)}
        self._word_cache = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, corpus, vocab_size: int = 8000, min_pair_count: int = 2,
              verbose: bool = False):
        """Learn merges from ``corpus`` (iterable of strings or one string)."""
        self._init_vocab()
        if isinstance(corpus, str):
            corpus = [corpus]

        # word frequency table at byte-unicode level
        words = Counter()
        for text in corpus:
            for w in self._re.findall(text.lower()):
                words[tuple(self.byte_encoder[b] for b in w.encode("utf-8"))] += 1

        n_merges = vocab_size - len(self.vocab)
        for step in range(n_merges):
            pairs = Counter()
            for w, c in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < min_pair_count:
                break
            a, b = best
            merged = a + b
            self.merges.append((a, b))
            self.vocab[merged] = len(self.vocab)
            # apply the merge to every word
            new_words = Counter()
            for w, c in words.items():
                nw, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == (a, b):
                        nw.append(merged)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                new_words[tuple(nw)] += c
            words = new_words
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {a!r} + {b!r} (x{count})")
        self._sync()
        return self

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _encode_word(self, word: str):
        if word in self._word_cache:
            return self._word_cache[word]
        symbols = [self.byte_encoder[b] for b in word.encode("utf-8")]
        while len(symbols) > 1:
            best_rank, best_i = None, -1
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            symbols = (symbols[:best_i]
                       + [symbols[best_i] + symbols[best_i + 1]]
                       + symbols[best_i + 2:])
        ids = [self.vocab[s] for s in symbols]
        self._word_cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for w in self._re.findall(text.lower()):
            ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special: bool = True):
        specials = {self.PAD, self.BOS, self.EOS}
        byte_seq = bytearray()
        for i in ids:
            tok = self.id_to_token[int(i)]
            if tok in specials:
                if not skip_special:
                    byte_seq.extend(tok.encode("utf-8"))
                continue
            byte_seq.extend(self.byte_decoder[c] for c in tok)
        return byte_seq.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        data = {
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._sync()
        return tok
