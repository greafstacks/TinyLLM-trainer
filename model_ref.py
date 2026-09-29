"""
model_ref.py

From-scratch NumPy implementation of the tiny GPT-style
transformer, plus the binary weight format. No PyTorch here on purpose:
this file is the literal spec that
app/src/main/java/.../Transformer.kt is translated from.

Contents
  - layer_norm / linear / gelu / softmax_rows / causal_self_attention /
    transformer_forward : the plain "whole sequence at once" forward pass.
  - Session : the incremental (KV-cache) version the phone actually runs.
    feed(token) processes ONE new token, reusing cached keys/values from
    earlier tokens, and returns logits for the next token. When the cache
    fills up it restarts from the most recent half of the context (the
    model uses absolute position embeddings, so a cache can't just slide).
  - write_binary / load_binary : the model.bin file format.

train.py imports this after training and checks, on the *real exported
file*, that (a) the manual forward pass matches PyTorch, and (b) the
KV-cache path matches the manual forward pass. If either check fails the
workflow fails instead of shipping a model the app would run differently.

model.bin layout (all little-endian):
    4 bytes   magic "TLM2"
    6 x int32 vocab_size, block_size, n_embd, n_head, n_layer, n_base
    V tokens, each: uint16 length n, then n x uint16 (UTF-16 code units of the token text)
              (the first n_base tokens are the single characters)
    (V - n_base) merges, each: int32 a, int32 b   (merge r creates token id n_base + r)
    float32 arrays, each row-major, in this exact order:
        tok_emb (V, C)
        pos_emb (T, C)
        for each layer, in LAYER_KEYS order:
            ln1_w (C) ln1_b (C)
            attn_q_w (C, C) attn_q_b (C)
            attn_k_w (C, C) attn_k_b (C)
            attn_v_w (C, C) attn_v_b (C)
            attn_proj_w (C, C) attn_proj_b (C)
            ln2_w (C) ln2_b (C)
            mlp_fc_w (C, 4C) mlp_fc_b (4C)
            mlp_proj_w (4C, C) mlp_proj_b (C)
        ln_f_w (C) ln_f_b (C)
        head_w (C, V) head_b (V)
Weight matrices are stored (in, out), i.e. already transposed relative to
PyTorch's nn.Linear, so y = x @ W + b with no transpose anywhere.

The tokenizer (bpe.py) is rebuilt from the token table + merges, so the
phone tokenizes exactly the way training did.
"""

import math
import struct

import numpy as np

MAGIC = b"TLM2"

LAYER_KEYS = [
    "ln1_w", "ln1_b",
    "attn_q_w", "attn_q_b",
    "attn_k_w", "attn_k_b",
    "attn_v_w", "attn_v_b",
    "attn_proj_w", "attn_proj_b",
    "ln2_w", "ln2_b",
    "mlp_fc_w", "mlp_fc_b",
    "mlp_proj_w", "mlp_proj_b",
]


def layer_shapes(c):
    return {
        "ln1_w": (c,), "ln1_b": (c,),
        "attn_q_w": (c, c), "attn_q_b": (c,),
        "attn_k_w": (c, c), "attn_k_b": (c,),
        "attn_v_w": (c, c), "attn_v_b": (c,),
        "attn_proj_w": (c, c), "attn_proj_b": (c,),
        "ln2_w": (c,), "ln2_b": (c,),
        "mlp_fc_w": (c, 4 * c), "mlp_fc_b": (4 * c,),
        "mlp_proj_w": (4 * c, c), "mlp_proj_b": (c,),
    }


# --------------------------------------------------------------------------
# binary format
# --------------------------------------------------------------------------

def write_binary(path, weights):
    cfg = weights["config"]
    v, t, c = cfg["vocab_size"], cfg["block_size"], cfg["n_embd"]
    h, n_layer = cfg["n_head"], cfg["n_layer"]
    tokens, n_base, merges = weights["tokens"], weights["n_base"], weights["merges"]

    if len(tokens) != v:
        raise ValueError("vocab_size does not match the number of tokens")
    if any(len(tok) < 1 or len(tok) > 0xFFFF or any(ord(ch) > 0xFFFF for ch in tok) for tok in tokens):
        raise ValueError("every token must be 1+ BMP characters")
    if not (0 < n_base <= v) or any(len(tokens[i]) != 1 for i in range(n_base)):
        raise ValueError("the first n_base tokens must be single characters")
    if len(merges) != v - n_base:
        raise ValueError("number of merges must equal vocab_size - n_base")
    for r, (a, b) in enumerate(merges):
        if not (0 <= a < n_base + r and 0 <= b < n_base + r):
            raise ValueError(f"merge {r} refers to a token that doesn't exist yet")
    if len(weights["blocks"]) != n_layer:
        raise ValueError("number of blocks does not match n_layer")

    def put(f, arr, shape, name):
        a = np.asarray(arr, dtype="<f4")
        if a.shape != tuple(shape):
            raise ValueError(f"{name}: expected shape {tuple(shape)}, got {a.shape}")
        f.write(np.ascontiguousarray(a).tobytes())

    shapes = layer_shapes(c)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6i", v, t, c, h, n_layer, n_base))
        for tok in tokens:
            f.write(struct.pack("<H", len(tok)))
            f.write(struct.pack(f"<{len(tok)}H", *[ord(ch) for ch in tok]))
        if merges:
            f.write(struct.pack(f"<{2 * len(merges)}i", *[x for pair in merges for x in pair]))
        put(f, weights["tok_emb"], (v, c), "tok_emb")
        put(f, weights["pos_emb"], (t, c), "pos_emb")
        for i, layer in enumerate(weights["blocks"]):
            for key in LAYER_KEYS:
                put(f, layer[key], shapes[key], f"blocks[{i}].{key}")
        put(f, weights["ln_f_w"], (c,), "ln_f_w")
        put(f, weights["ln_f_b"], (c,), "ln_f_b")
        put(f, weights["head_w"], (c, v), "head_w")
        put(f, weights["head_b"], (v,), "head_b")


def load_binary(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise ValueError("not a model.bin file (bad magic)")
    v, t, c, h, n_layer, n_base = struct.unpack_from("<6i", data, 4)
    off = 4 + 24

    tokens = []
    for _ in range(v):
        (n,) = struct.unpack_from("<H", data, off)
        off += 2
        codes = struct.unpack_from(f"<{n}H", data, off)
        off += 2 * n
        tokens.append("".join(chr(code) for code in codes))

    n_merges = v - n_base
    flat = struct.unpack_from(f"<{2 * n_merges}i", data, off) if n_merges else ()
    off += 8 * n_merges
    merges = [(flat[2 * i], flat[2 * i + 1]) for i in range(n_merges)]

    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=off)
        off += 4 * n
        return arr.astype(np.float64).reshape(shape)

    weights = {
        "config": {"vocab_size": v, "block_size": t, "n_embd": c, "n_head": h, "n_layer": n_layer},
        "tokens": tokens,
        "n_base": n_base,
        "merges": merges,
    }
    weights["tok_emb"] = take((v, c))
    weights["pos_emb"] = take((t, c))
    shapes = layer_shapes(c)
    weights["blocks"] = [
        {key: take(shapes[key]) for key in LAYER_KEYS} for _ in range(n_layer)
    ]
    weights["ln_f_w"] = take((c,))
    weights["ln_f_b"] = take((c,))
    weights["head_w"] = take((c, v))
    weights["head_b"] = take((v,))
    if off != len(data):
        raise ValueError(f"model.bin has {len(data) - off} unexpected trailing bytes")
    return weights


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------

def _a(x):
    return np.asarray(x, dtype=np.float64)


def layer_norm(x, weight, bias, eps=1e-5):
    """x: (T, C). Normalizes each row over the last axis, then scale+shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # biased variance, matches nn.LayerNorm
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def linear(x, weight, bias):
    """weight is (in, out): a plain matmul, no transpose."""
    return x @ weight + bias


def gelu(x):
    """tanh approximation (matches nn.GELU(approximate='tanh') in train.py)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax_rows(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def causal_self_attention(x, layer_weights, n_head):
    """Whole-sequence attention. x: (T, C) -> (T, C)."""
    t, c = x.shape
    head_size = c // n_head

    q = linear(x, layer_weights["attn_q_w"], layer_weights["attn_q_b"])
    k = linear(x, layer_weights["attn_k_w"], layer_weights["attn_k_b"])
    v = linear(x, layer_weights["attn_v_w"], layer_weights["attn_v_b"])

    out = np.zeros((t, c), dtype=np.float64)
    for h in range(n_head):
        lo, hi = h * head_size, (h + 1) * head_size
        qh, kh, vh = q[:, lo:hi], k[:, lo:hi], v[:, lo:hi]
        scores = (qh @ kh.T) / math.sqrt(head_size)
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
        out[:, lo:hi] = softmax_rows(scores) @ vh

    return linear(out, layer_weights["attn_proj_w"], layer_weights["attn_proj_b"])


def transformer_forward(weights, token_ids):
    """Logits for EVERY position, shape (T, vocab_size)."""
    n_head = weights["config"]["n_head"]
    t = len(token_ids)

    x = _a(weights["tok_emb"])[list(token_ids)] + _a(weights["pos_emb"])[:t]

    for layer in weights["blocks"]:
        lw = {k: _a(v) for k, v in layer.items()}
        x = x + causal_self_attention(layer_norm(x, lw["ln1_w"], lw["ln1_b"]), lw, n_head)
        hidden = gelu(linear(layer_norm(x, lw["ln2_w"], lw["ln2_b"]), lw["mlp_fc_w"], lw["mlp_fc_b"]))
        x = x + linear(hidden, lw["mlp_proj_w"], lw["mlp_proj_b"])

    x = layer_norm(x, _a(weights["ln_f_w"]), _a(weights["ln_f_b"]))
    return linear(x, _a(weights["head_w"]), _a(weights["head_b"]))


# --------------------------------------------------------------------------
# incremental (KV-cache) inference -- what Transformer.kt runs on the phone
# --------------------------------------------------------------------------

class Session:
    def __init__(self, weights):
        cfg = weights["config"]
        self.t = cfg["block_size"]
        self.c = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.tok_emb = _a(weights["tok_emb"])
        self.pos_emb = _a(weights["pos_emb"])
        self.layers = [{k: _a(v) for k, v in layer.items()} for layer in weights["blocks"]]
        self.ln_f_w, self.ln_f_b = _a(weights["ln_f_w"]), _a(weights["ln_f_b"])
        self.head_w, self.head_b = _a(weights["head_w"]), _a(weights["head_b"])
        self._reset()

    def _reset(self):
        self.keys = [np.zeros((self.t, self.c)) for _ in range(self.n_layer)]
        self.values = [np.zeros((self.t, self.c)) for _ in range(self.n_layer)]
        self.length = 0
        self.window = []  # tokens currently held in the cache (for tests)

    def _step(self, token):
        pos = self.length
        if pos >= self.t:
            raise RuntimeError("cache is full")
        c, n_head = self.c, self.n_head
        head_size = c // n_head

        x = self.tok_emb[token] + self.pos_emb[pos]  # (C,)
        for li, lw in enumerate(self.layers):
            h1 = layer_norm(x[None, :], lw["ln1_w"], lw["ln1_b"])[0]
            q = h1 @ lw["attn_q_w"] + lw["attn_q_b"]
            self.keys[li][pos] = h1 @ lw["attn_k_w"] + lw["attn_k_b"]
            self.values[li][pos] = h1 @ lw["attn_v_w"] + lw["attn_v_b"]

            att = np.zeros(c)
            for h in range(n_head):
                lo, hi = h * head_size, (h + 1) * head_size
                scores = (self.keys[li][:pos + 1, lo:hi] @ q[lo:hi]) / math.sqrt(head_size)
                weights_h = softmax_rows(scores)
                att[lo:hi] = weights_h @ self.values[li][:pos + 1, lo:hi]
            x = x + (att @ lw["attn_proj_w"] + lw["attn_proj_b"])

            h2 = layer_norm(x[None, :], lw["ln2_w"], lw["ln2_b"])[0]
            hidden = gelu(h2 @ lw["mlp_fc_w"] + lw["mlp_fc_b"])
            x = x + (hidden @ lw["mlp_proj_w"] + lw["mlp_proj_b"])

        xf = layer_norm(x[None, :], self.ln_f_w, self.ln_f_b)[0]
        self.length = pos + 1
        self.window.append(token)
        return xf @ self.head_w + self.head_b

    def feed(self, token):
        """Consume one token, return logits for the token after it."""
        if self.length >= self.t:
            keep = self.t // 2
            tail = self.window[-keep:]
            self._reset()
            for tok in tail:
                self._step(tok)
        return self._step(token)
