#!/usr/bin/env python3
"""Gold-tokenization evaluation of a Trankit model (any SentencePiece encoder).

Feeds the gold *surface* tokens (multiword-token ranges such as "du" are kept as
one token, so the MWT expander is exercised) sentence by sentence, flattens the
expanded words and scores them against the gold syntactic words:
UPOS, LEMMA, UAS, LAS (token-level accuracy, as in eval_trankit.py).

Gold must be the ALSI-cleaned, retagged file (copulas = VERB, AUX only for true
auxiliaries), e.g. eval/gold/fr_gsd-test_alsi_cleaned.conllu.

Usage (on the mini: CPU only; MPS jobs belong on the MacBook):
  HF_HUB_CACHE=/Volumes/Data/research_data/hf_cache/hub HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=2 nice -n 10 python scripts/eval_gold_tok.py \
    --model models/camembert_fr --embedding almanach/camembert-base \
    --gold eval/gold/fr_gsd-test_alsi_cleaned.conllu [--device cpu] [--errors errs.tsv]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path


def read_conllu(path):
    """Yield (sent_id, surface_tokens, words); words = list of dicts."""
    sid, surface, words, mwt_end = None, [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in list(f) + ["\n"]:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sid = line.split("=", 1)[1].strip()
                continue
            if not line:
                if words:
                    yield sid, surface, words
                sid, surface, words, mwt_end = None, [], [], 0
                continue
            c = line.split("\t")
            if len(c) < 10 or "." in c[0]:
                continue
            if "-" in c[0]:
                mwt_end = int(c[0].split("-")[1])
                surface.append(c[1])
                continue
            i = int(c[0])
            words.append(dict(form=c[1], lemma=c[2], upos=c[3],
                              head=int(c[6]) if c[6].isdigit() else 0,
                              deprel=c[7]))
            if i > mwt_end:
                surface.append(c[1])


def flatten(tokens):
    out = []
    for t in tokens:
        out.extend(t.get("expanded", [t]))
    return out


def pct(n, d):
    return round(100 * n / d, 2) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--embedding", default="almanach/camembert-base")
    ap.add_argument("--category", default="customized-mwt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--errors", help="write per-token UPOS/LAS errors to this TSV")
    ap.add_argument("--limit", type=int, default=0, help="only first N sentences")
    args = ap.parse_args()

    from trankit import Pipeline
    p = Pipeline(lang=args.category, cache_dir=args.model, gpu=False,
                 embedding=args.embedding, device=args.device)

    n = Counter()
    upos_conf = Counter()
    errs = []
    skipped = 0
    for k, (sid, surface, gold) in enumerate(read_conllu(args.gold)):
        if args.limit and k >= args.limit:
            break
        pred = flatten(p(surface, is_sent=True).get("tokens", []))
        if len(pred) != len(gold):
            skipped += 1
            print(f"WARNING {sid}: gold {len(gold)} words, pred {len(pred)}; skipped",
                  file=sys.stderr)
            continue
        for i, (g, q) in enumerate(zip(gold, pred)):
            n["tok"] += 1
            ok_u = q.get("upos") == g["upos"]
            ok_l = q.get("lemma") == g["lemma"]
            ok_h = q.get("head") == g["head"]
            ok_r = (q.get("deprel", "") or "").lower() == g["deprel"].lower()
            n["upos"] += ok_u
            n["lemma"] += ok_l
            n["uas"] += ok_h
            n["las"] += ok_h and ok_r
            if not ok_u:
                upos_conf[(g["upos"], q.get("upos"))] += 1
            if args.errors and not (ok_u and ok_h and ok_r):
                errs.append((sid, i + 1, g["form"], g["upos"], q.get("upos"),
                             g["head"], q.get("head"), g["deprel"], q.get("deprel")))

    print("sentences_skipped\ttok\tupos\tlemma\tuas\tlas")
    print("\t".join(str(x) for x in [skipped, n["tok"]] +
                    [pct(n[m], n["tok"]) for m in ("upos", "lemma", "uas", "las")]))
    print("\nTop UPOS confusions (gold -> pred):", file=sys.stderr)
    for (g, q), c in upos_conf.most_common(10):
        print(f"  {g:6s} -> {q!s:6s} {c}", file=sys.stderr)
    if args.errors:
        with open(args.errors, "w", encoding="utf-8") as f:
            f.write("sent_id\ttok\tform\tgold_upos\tpred_upos\tgold_head\tpred_head\tgold_rel\tpred_rel\n")
            for e in errs:
                f.write("\t".join(map(str, e)) + "\n")


if __name__ == "__main__":
    main()
