"""
train.py

Trains a tiny GPT-style transformer on corpus.txt, using a subword (BPE)
tokenizer learned from that same text (see bpe.py): common words are single
tokens, so the model works word-by-word rather than letter-by-letter. Exports
the weights AND the tokenizer as app/src/main/assets/model.bin (raw float32,
see model_ref.py for the layout) for the Android app to run itself -- no
PyTorch/TFLite/ONNX ships in the app.

Meant to run inside GitHub Actions (see .github/workflows/build-apk.yml).

Two safety nets:
  * --max-minutes: training stops after this long and exports whatever it
    has, so a too-big config finishes (undertrained) instead of hitting
    GitHub's 6-hour job limit and producing nothing.
  * After training, the exported model.bin is re-read with model_ref.py
    (plain NumPy, no PyTorch) and checked against PyTorch's own output,
    the KV-cache path used on the phone is checked against the
    whole-sequence path, and the tokenizer rebuilt from the file is checked
    against the one used for training. If anything disagrees the run fails.

Training keeps the checkpoint with the best validation loss, so training
longer than the data can support (overfitting) doesn't make the result worse.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bpe
import model_ref


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(t, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, n_batches):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


def to_np(t):
    return t.detach().cpu().numpy().astype(np.float32)


def export_weights(model, tokens, n_base, merges, cfg):
    """Collects the trained weights into the dict model_ref.write_binary expects.
    Linear weights are transposed (.T) from PyTorch's (out, in) to (in, out)."""
    weights = {
        "config": cfg,
        "tokens": tokens,
        "n_base": n_base,
        "merges": merges,
        "tok_emb": to_np(model.tok_emb.weight),
        "pos_emb": to_np(model.pos_emb.weight),
        "blocks": [],
        "ln_f_w": to_np(model.ln_f.weight),
        "ln_f_b": to_np(model.ln_f.bias),
        "head_w": to_np(model.head.weight.T),
        "head_b": to_np(model.head.bias),
    }
    for block in model.blocks:
        weights["blocks"].append({
            "ln1_w": to_np(block.ln1.weight), "ln1_b": to_np(block.ln1.bias),
            "attn_q_w": to_np(block.attn.q_proj.weight.T), "attn_q_b": to_np(block.attn.q_proj.bias),
            "attn_k_w": to_np(block.attn.k_proj.weight.T), "attn_k_b": to_np(block.attn.k_proj.bias),
            "attn_v_w": to_np(block.attn.v_proj.weight.T), "attn_v_b": to_np(block.attn.v_proj.bias),
            "attn_proj_w": to_np(block.attn.out_proj.weight.T), "attn_proj_b": to_np(block.attn.out_proj.bias),
            "ln2_w": to_np(block.ln2.weight), "ln2_b": to_np(block.ln2.bias),
            "mlp_fc_w": to_np(block.mlp.fc.weight.T), "mlp_fc_b": to_np(block.mlp.fc.bias),
            "mlp_proj_w": to_np(block.mlp.proj.weight.T), "mlp_proj_b": to_np(block.mlp.proj.bias),
        })
    return weights


def self_check(model, bin_path, data, block_size, tok, text):
    """Verify the exported FILE, not the in-memory weights."""
    loaded = model_ref.load_binary(bin_path)

    # 1) the tokenizer rebuilt from the file must tokenize exactly like training did
    rebuilt = bpe.BPETokenizer(loaded["tokens"], loaded["n_base"], loaded["merges"])
    sample_text = text[:5000]
    ids_a, ids_b = tok.encode(sample_text), rebuilt.encode(sample_text)
    if ids_a != ids_b or rebuilt.decode(ids_b) != sample_text:
        raise RuntimeError("Tokenizer rebuilt from model.bin does not match the training tokenizer.")
    print("self-check 1/4: tokenizer rebuilt from model.bin matches training (and round-trips text).")

    check_len = min(block_size, len(data) - 1)
    sample_ids = data[:check_len].tolist()

    model.eval()
    with torch.no_grad():
        torch_logits, _ = model(torch.tensor([sample_ids]))
    torch_logits = torch_logits[0].numpy()

    full = model_ref.transformer_forward(loaded, sample_ids)
    diff_export = float(np.abs(torch_logits - full).max())
    print(f"self-check 2/4: PyTorch vs manual forward on the exported file, max logit diff = {diff_export:.6f}")
    if diff_export > 1e-3:
        raise RuntimeError("model.bin does not reproduce PyTorch's output. Refusing to ship it.")

    session = model_ref.Session(loaded)
    inc = np.stack([session.feed(t) for t in sample_ids])
    diff_cache = float(np.abs(full - inc).max())
    print(f"self-check 3/4: KV-cache vs whole-sequence forward, max logit diff = {diff_cache:.9f}")
    if diff_cache > 1e-6:
        raise RuntimeError("KV-cache path disagrees with the whole-sequence path.")

    # Run past block_size so the cache rollover path is exercised too.
    long_ids = data[:block_size + block_size // 2 + 5].tolist()
    session = model_ref.Session(loaded)
    worst = 0.0
    for t in long_ids:
        logits = session.feed(t)
        ref = model_ref.transformer_forward(loaded, session.window)[-1]
        worst = max(worst, float(np.abs(logits - ref).max()))
    print(f"self-check 4/4: cache rollover past block_size, max logit diff = {worst:.9f}")
    if worst > 1e-6:
        raise RuntimeError("KV-cache rollover disagrees with a fresh forward pass.")

    print("self-check passed: model.bin is verified to match the trained model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus.txt")
    parser.add_argument("--out", default="model.bin")
    parser.add_argument("--vocab-size", type=int, default=4000,
                        help="tokenizer size: characters + learned subword merges. Bigger = more "
                             "whole-word tokens (closer to word-level), but each token is seen less often.")
    parser.add_argument("--block-size", type=int, default=128, help="context length in TOKENS")
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1000000,
                        help="upper bound; with --max-minutes set, the time budget normally ends training first")
    parser.add_argument("--max-minutes", type=float, default=300,
                        help="stop training after this many minutes and export what we have (0 = no limit)")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="stop early after this many evals in a row without a better val loss (0 = off)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    # Keep everything in the Basic Multilingual Plane so one character is
    # always exactly one UTF-16 code unit on the Kotlin side.
    text = "".join(c for c in text if ord(c) <= 0xFFFF)
    print(f"corpus: {len(text)} characters")

    t0 = time.time()
    tokens, n_base, merges = bpe.train_bpe(text, args.vocab_size)
    tok = bpe.BPETokenizer(tokens, n_base, merges)
    ids = tok.encode(text)
    vocab_size = len(tokens)
    print(f"tokenizer: {vocab_size} tokens ({n_base} characters + {len(merges)} merges), "
          f"{len(text) / len(ids):.2f} characters per token, {len(ids)} training tokens "
          f"({time.time() - t0:.0f}s)")
    print("example:", " | ".join(tokens[i] for i in ids[:30]).replace("\n", "\\n"))

    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.95 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
    }
    model = TinyGPT(vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters (~{n_params * 4 / 1e6:.1f} MB as float32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None
    bad_evals = 0
    start = time.time()
    budget = args.max_minutes * 60
    for it in range(args.max_iters):
        # cosine learning-rate decay over whichever runs out first: the time budget or max-iters
        progress = max(it / args.max_iters, (time.time() - start) / budget if budget > 0 else 0.0)
        lr = args.learning_rate * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
        for g in optimizer.param_groups:
            g["lr"] = lr
        xb, yb = get_batch(train_data, args.block_size, args.batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start
        out_of_time = budget > 0 and elapsed > budget
        if it % args.log_interval == 0:
            print(f"step {it:5d} | train loss {loss.item():.4f} | {elapsed / 60:.1f} min elapsed", flush=True)
        if it % args.eval_interval == 0 or it == args.max_iters - 1 or out_of_time:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.eval_batches)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                bad_evals = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_evals += 1
            print(f"step {it:5d} | val loss {val_loss:.4f}{'  (best so far)' if improved else ''}", flush=True)
        if out_of_time:
            print(f"time budget of {args.max_minutes:g} min reached at step {it}; "
                  f"stopping early and exporting what we have")
            break
        if args.patience > 0 and bad_evals >= args.patience:
            print(f"val loss hasn't improved for {args.patience} evals; stopping at step {it} "
                  f"(the best checkpoint is kept)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"using the best checkpoint (val loss {best_val:.4f})")

    model.eval()
    print("--- sample replies ---")
    for q in ["hello", "what level is this", "how are you"]:
        prompt = tok.encode("\ue000" + q + "\ue001")
        out = model.generate(torch.tensor([prompt]), max_new_tokens=40)[0].tolist()
        print(f"{q!r} -> {tok.decode(out[len(prompt):]).split(chr(10))[0]!r}")
    print("----------------------")

    weights = export_weights(model, tokens, n_base, merges, cfg)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model_ref.write_binary(args.out, weights)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    self_check(model, args.out, train_data, args.block_size, tok, text)


if __name__ == "__main__":
    main()
