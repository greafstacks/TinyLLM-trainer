"""
bpe.py

A small byte-pair-encoding (subword) tokenizer, pure Python, no
dependencies. Frequent words end up as single tokens, rare words are
split into a few pieces, and anything typed is still representable
(it falls back to single characters), so there's no "unknown word"
problem.

The Kotlin twin lives in Transformer.kt and must behave identically:
same pre-tokenization, same merge order. The exact tokens and merges
are stored in model.bin (see model_ref.py), and train.py checks that
a tokenizer rebuilt from that file reproduces the training tokenization.

Pipeline:
  pretokenize(text)  -> chunks that merges never cross
  train_bpe(text, n) -> (tokens, n_base, merges)
  BPETokenizer(...)  -> encode / decode

Token ids: the first n_base ids are the single characters (sorted); merge
number r (0-based) creates token id n_base + r out of two earlier ids.
"""

from collections import Counter, defaultdict


def is_letter(ch):
    return ch.isalpha()


def is_digit(ch):
    return "0" <= ch <= "9"


def pretokenize(text):
    """Split text into chunks: a word (letters), optionally with ONE leading
    space; a run of digits; or any other single character. Concatenating the
    chunks gives back the text exactly."""
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == " " and i + 1 < n and is_letter(text[i + 1]):
            j = i + 1
            while j < n and is_letter(text[j]):
                j += 1
        elif is_letter(ch):
            j = i
            while j < n and is_letter(text[j]):
                j += 1
        elif is_digit(ch):
            j = i
            while j < n and is_digit(text[j]):
                j += 1
        else:
            j = i + 1
        out.append(text[i:j])
        i = j
    return out


def train_bpe(text, vocab_size):
    """Learn merges. Returns (tokens, n_base, merges); merges is a list of
    (id_a, id_b) pairs, in the order they were learned."""
    chars = sorted(set(text))
    n_base = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = list(chars)
    merges = []
    n_merges = vocab_size - n_base
    if n_merges <= 0:
        return tokens, n_base, merges

    counts = Counter(pretokenize(text))
    words = [[stoi[c] for c in w] for w in counts]
    freqs = list(counts.values())

    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_words[p].add(wi)

    for _ in range(n_merges):
        if not pair_counts:
            break
        # most frequent pair; ties broken by the pair itself so runs are reproducible
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = best
        new_id = len(tokens)
        tokens.append(tokens[a] + tokens[b])
        merges.append(best)

        for wi in list(pair_words[best]):
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            new_w = []
            k = 0
            while k < len(w):
                if k < len(w) - 1 and w[k] == a and w[k + 1] == b:
                    new_w.append(new_id)
                    k += 2
                else:
                    new_w.append(w[k])
                    k += 1
            words[wi] = new_w
            for p in zip(new_w, new_w[1:]):
                pair_counts[p] += f
                pair_words[p].add(wi)
        pair_counts.pop(best, None)
        pair_words.pop(best, None)

    return tokens, n_base, merges


class BPETokenizer:
    def __init__(self, tokens, n_base, merges):
        self.tokens = list(tokens)
        self.n_base = n_base
        self.stoi = {self.tokens[i]: i for i in range(n_base)}
        self.ranks = {tuple(pair): r for r, pair in enumerate(merges)}
        self._cache = {}

    def _encode_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        ids = [self.stoi[c] for c in chunk if c in self.stoi]  # unknown characters are dropped
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new_id = self.n_base + best_rank
            out = []
            k = 0
            while k < len(ids):
                if k < len(ids) - 1 and ids[k] == a and ids[k + 1] == b:
                    out.append(new_id)
                    k += 2
                else:
                    out.append(ids[k])
                    k += 1
            ids = out
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids)
