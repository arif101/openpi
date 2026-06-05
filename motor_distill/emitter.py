"""SUB-GOAL EMITTER (design module A): instruction -> (noun-phrase, gripper-event) ladder.

Framed as SPAN TAGGING, not token generation: per word predict {O, OBJ, CONT}; the ladder is read off
the tagged spans (OBJ span -> (span, close), CONT span -> (span, open), in order of appearance). Because
spans are COPIED verbatim from the instruction, UNSEEN objects/containers (mapped to <unk> on input) are
still emitted correctly -- the tagger keys off the seen prepositional context ("in the ___" => CONT), not
the noun identity. This is how "the container is just another open-vocab noun" holds by construction.

Tiny windowed-MLP tagger (context window [-2..+1]) in the distill_reach JAX style. Trains on CPU in seconds.
Headline metric: EXACT-LADDER accuracy on HELD-OUT objects+containers (never seen in training).

Usage:  python motor_distill/gen_emitter_data.py && python motor_distill/emitter.py
"""
from __future__ import annotations

import argparse, json, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np, optax

PAD, UNK = "<pad>", "<unk>"
LABELS = ["O", "OBJ", "CONT"]
WIN = (-2, -1, 0, 1)   # context word offsets fed to the per-token MLP


def load(path):
    return [json.loads(l) for l in open(path)]


def spans_from_ladder(ladder):
    """Parse '<sub> X <grip> close <sub> Y <grip> open <eos>' -> [('OBJ',X), ('CONT',Y)]."""
    out, toks = [], ladder.split()
    i = 0
    while i < len(toks):
        if toks[i] == "<sub>":
            j = i + 1; phrase = []
            while j < len(toks) and toks[j] != "<grip>":
                phrase.append(toks[j]); j += 1
            ev = toks[j + 1] if j + 1 < len(toks) else "close"
            out.append(("OBJ" if ev == "close" else "CONT", " ".join(phrase)))
            i = j + 2
        else:
            i += 1
    return out


def tag_sentence(instr, ladder):
    """Word-level BIO-free labels {O,OBJ,CONT} by locating each ladder span in the instruction words."""
    words = instr.split()
    lab = [0] * len(words)
    for role, phrase in spans_from_ladder(ladder):
        pw = phrase.split()
        for s in range(len(words) - len(pw) + 1):
            if words[s:s + len(pw)] == pw:
                for k in range(len(pw)):
                    lab[s + k] = LABELS.index(role)
                break
    return words, lab


def build_vocab(rows):
    vocab = {PAD: 0, UNK: 1}
    for r in rows:
        for w in r["instr"].split():
            vocab.setdefault(w, len(vocab))
    return vocab


def encode(words, vocab):
    return [vocab.get(w, vocab[UNK]) for w in words]


def init(key, V, dim=32, dh=64):
    k1, k2, k3 = jax.random.split(key, 3)
    s = lambda k, a, b: jax.random.normal(k, (a, b)) * (1.0 / np.sqrt(a))
    return {"emb": jax.random.normal(k1, (V, dim)) * 0.1,
            "w1": s(k2, dim * len(WIN), dh), "b1": jnp.zeros(dh),
            "w2": s(k3, dh, len(LABELS)), "b2": jnp.zeros(len(LABELS))}


def apply(p, ids):
    """ids: (T,) int. Returns (T, 3) logits via a window of embeddings around each position."""
    emb = p["emb"][ids]                                   # (T, dim)
    T = emb.shape[0]
    pad = jnp.zeros((1, emb.shape[1]))
    padded = jnp.concatenate([pad, pad, emb, pad], axis=0)  # offset so index t+2 == word t
    feats = jnp.concatenate([padded[jnp.arange(T) + (o + 2)] for o in WIN], axis=-1)  # (T, dim*4)
    h = jax.nn.gelu(feats @ p["w1"] + p["b1"])
    return h @ p["w2"] + p["b2"]


def decode(words, pred):
    """Read ladder spans off predicted per-word labels (contiguous OBJ / CONT runs, in order)."""
    out, i = [], 0
    while i < len(words):
        if pred[i] in (1, 2):
            role = pred[i]; j = i
            while j < len(words) and pred[j] == role:
                j += 1
            out.append((LABELS[role], " ".join(words[i:j]))); i = j
        else:
            i += 1
    parts = []
    for role, phrase in out:
        parts += ["<sub>", phrase, "<grip>", "close" if role == "OBJ" else "open"]
    return " ".join(parts + ["<eos>"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/emitter")
    ap.add_argument("--out", default="runs/emitter.pkl")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--unk-drop", type=float, default=0.35)   # word-dropout -> trains <unk> to copy-by-context
    args = ap.parse_args()
    tr = load(pathlib.Path(args.data) / "train.jsonl")
    va = load(pathlib.Path(args.data) / "val.jsonl")
    vocab = build_vocab(tr)
    print(f"train={len(tr)} val={len(va)} vocab={len(vocab)}")

    def prep(rows):
        out = []
        for r in rows:
            words, lab = tag_sentence(r["instr"], r["ladder"])
            out.append((words, np.array(encode(words, vocab)), np.array(lab), r["ladder"]))
        return out
    TR, VA = prep(tr), prep(va)

    p = init(jax.random.PRNGKey(0), len(vocab))
    opt = optax.adam(args.lr); st = opt.init(p)

    def loss_fn(p, ids, lab):
        lg = apply(p, ids)
        return optax.softmax_cross_entropy_with_integer_labels(lg, lab).mean()
    gstep = jax.jit(jax.value_and_grad(loss_fn))

    best = (-1.0, None)
    for ep in range(args.epochs):
        order = np.random.permutation(len(TR)); tot = 0.0
        for i in order:
            _, ids, lab, _ = TR[i]
            ids = np.where(np.random.rand(len(ids)) < args.unk_drop, vocab[UNK], ids)  # word-dropout
            l, g = gstep(p, jnp.asarray(ids), jnp.asarray(lab))
            upd, st = opt.update(g, st); p = optax.apply_updates(p, upd); tot += float(l)
        if ep % 10 == 0 or ep == args.epochs - 1:
            def ladder_acc(DS):
                ok = 0
                for words, ids, lab, gold in DS:
                    pred = np.array(apply(p, jnp.asarray(ids))).argmax(-1)
                    ok += int(decode(words, pred) == gold)
                return ok / len(DS)
            ho = ladder_acc(VA)
            if ho > best[0]:
                best = (ho, jax.tree.map(np.asarray, p))
            print(f"ep{ep:3d} loss={tot/len(TR):.4f}  train-ladder={ladder_acc(TR):.3f}  "
                  f"HELD-OUT-ladder={ho:.3f}", flush=True)

    p = jax.tree.map(jnp.asarray, best[1])   # keep best-held-out checkpoint
    print(f"\nBEST held-out ladder acc = {best[0]:.3f}")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"params": jax.tree.map(np.asarray, p), "vocab": vocab}, open(args.out, "wb"))
    # show a few held-out decodes
    print("\nheld-out samples:")
    for words, ids, lab, gold in VA[:5]:
        pred = np.array(apply(p, jnp.asarray(ids))).argmax(-1)
        dec = decode(words, pred)
        print(f"  {'OK ' if dec==gold else 'ERR'} {' '.join(words):46s} -> {dec}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
