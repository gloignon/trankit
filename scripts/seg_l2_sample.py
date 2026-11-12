#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Stratified sample of CEAAL2 texts + per-system sentence spans, for building a gold sentence-boundary set.

  seg_l2_sample.py sample CEAAL2.tsv OUT.json [--native 6 --learner 18 --seed 11]
  seg_l2_sample.py predict TEXTS.json SYSTEM OUT.json      (SYSTEM as in eval_sentence_segmentation.py)
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sentence_segmentation import align, make_system  # noqa: E402


def sample(tsv, out, n_native=6, n_learner=18, seed=11):
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(tsv, encoding="utf-8-sig", newline=""), delimiter="\t"))
    rng = random.Random(seed)
    nat = [r for r in rows if r["writer_group"] == "native"]
    lea = [r for r in rows if r["writer_group"] == "learner" and 60 <= len(r["text"].split()) <= 500]
    by_l1 = defaultdict(list)
    for r in lea:
        by_l1[r["l1_group"]].append(r)
    chosen = rng.sample(nat, n_native)
    groups = sorted(by_l1)
    per = [n_learner // len(groups) + (1 if i < n_learner % len(groups) else 0) for i in range(len(groups))]
    for g, k in zip(groups, per):
        chosen += rng.sample(by_l1[g], min(k, len(by_l1[g])))
    texts = [{"id": r["id"], "group": r["writer_group"], "l1": r["l1_group"], "genre": r["genre"],
              "text": re.sub(r"\s+", " ", r["text"]).strip()} for r in chosen]
    json.dump(texts, open(out, "w"), ensure_ascii=False, indent=1)
    print(len(texts), "texts,", sum(len(t["text"].split()) for t in texts), "words")


def predict(texts_json, spec, out):
    texts = json.load(open(texts_json))
    f = make_system(spec)
    res = {t["id"]: align(t["text"], f(t["text"])) for t in texts}
    json.dump(res, open(out, "w"))
    print(spec, sum(len(v) for v in res.values()), "sentences")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "sample":
        sample(a[1], a[2], **{k: int(a[a.index("--" + k) + 1]) for k in ("native", "learner", "seed") if "--" + k in a} and {
            {"native": "n_native", "learner": "n_learner", "seed": "seed"}[k]: int(a[a.index("--" + k) + 1])
            for k in ("native", "learner", "seed") if "--" + k in a})
    else:
        predict(a[1], a[2], a[3])
