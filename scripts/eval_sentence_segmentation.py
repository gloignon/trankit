#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Sentence-segmentation benchmark on raw multi-sentence paragraphs (exact sentence-span F1).

  eval_sentence_segmentation.py prepare GOLD.conllu OUT.json [--seed 7 --max 12]
  eval_sentence_segmentation.py run OUT.json SYSTEM [--limit N]
SYSTEM: syntok | stanza | sat[:model] | trankit:MODEL_DIR:CATEGORY[:EMBEDDING]
A predicted sentence counts as correct only if its (start,end) character span, after whitespace trimming,
equals a gold sentence span. Systems return sentence strings that are aligned to the paragraph in order.
"""
import json
import os
import random
import sys
import time


def prepare(conllu, out, seed=7, kmax=12):
    texts = [l[9:].rstrip("\n") for l in open(conllu, encoding="utf-8") if l.startswith("# text = ")]
    rng = random.Random(seed)
    paras, i = [], 0
    while i < len(texts):
        k = rng.randint(1, kmax)
        sents = texts[i:i + k]
        i += k
        text, pos, gold = "", 0, []
        for s in sents:
            if text:
                text += " "
            start = len(text)
            text += s
            gold.append([start, len(text)])
        paras.append({"text": text, "gold": gold})
    json.dump(paras, open(out, "w"), ensure_ascii=False)
    print(f"{len(texts)} sentences, {len(paras)} paragraphs (mean {len(texts) / len(paras):.1f})")


def align(text, sentences):
    """Character spans of consecutive sentence strings inside text (whitespace-trimmed)."""
    spans, cur = [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        j = text.find(s, cur)
        if j < 0:  # tokenizer changed characters: fall back to first-word anchoring
            j = text.find(s.split()[0], cur) if s.split() else cur
            j = cur if j < 0 else j
            spans.append([j, min(len(text), j + len(s))])
            cur = spans[-1][1]
        else:
            spans.append([j, j + len(s)])
            cur = j + len(s)
    return spans


def make_system(spec):
    kind, _, rest = spec.partition(":")
    if kind == "syntok":
        import syntok.segmenter as segmenter

        def f(t):
            return ["".join(map(str, s)).strip() for p in segmenter.process(t) for s in p]
        return f
    if kind == "stanza":
        import stanza
        nlp = stanza.Pipeline("fr", processors="tokenize", verbose=False, use_gpu=False)
        return lambda t: [s.text for s in nlp(t).sentences]
    if kind == "sat":
        from wtpsplit import SaT
        sat = SaT(rest or "sat-3l-sm")
        return lambda t: sat.split(t)
    if kind == "trankit":
        model_dir, category, *emb = rest.split(":")
        sys.path.insert(0, os.environ.get("TRANKIT_HOME", "."))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from trankit import Pipeline
        p = Pipeline(lang=category, cache_dir=model_dir, gpu=False, device="cpu",
                     embedding=emb[0] if emb else "xlm-roberta-base")
        return lambda t: [s["text"] for s in p.tokenize(t)["sentences"]]
    raise SystemExit(f"unknown system {spec}")


def run(data, spec, limit=None):
    paras = json.load(open(data))[:limit]
    f = make_system(spec)
    tp = npred = ngold = exact_par = 0
    t0 = time.time()
    for p in paras:
        pred = [tuple(x) for x in align(p["text"], f(p["text"]))]
        gold = [tuple(x) for x in p["gold"]]
        tp += len(set(pred) & set(gold))
        npred += len(pred)
        ngold += len(gold)
        exact_par += pred == gold
    P, R = tp / max(npred, 1), tp / max(ngold, 1)
    F = 2 * P * R / max(P + R, 1e-9)
    print(json.dumps({"system": spec, "paragraphs": len(paras), "gold_sents": ngold, "pred_sents": npred,
                      "P": round(100 * P, 2), "R": round(100 * R, 2), "F1": round(100 * F, 2),
                      "paragraphs_exact": round(100 * exact_par / len(paras), 1), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prepare":
        prepare(a[1], a[2])
    else:
        run(a[1], a[2], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
