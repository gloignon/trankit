#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Build a boundary-review sheet from per-system sentence spans, and later turn decisions into gold.

  seg_l2_review.py sheet TEXTS.json OUT.tsv P_syntok.json P_stanza.json P_sat.json [--extra P_old.json ...]
  seg_l2_review.py gold  TEXTS.json SHEET.tsv OUT_GOLD.json
Sheet rows are candidate boundaries (sentence-start offsets) that the baseline systems do not all agree on, plus
sentences longer than 45 words in the agreed segmentation (possible merges: add a boundary there by writing a new row).
Fill the `decision` column with y (boundary) or n (no boundary); an empty decision accepts `suggest` (majority vote).
"""
import csv
import json
import re
import sys


def starts(spans):
    return {s for s, _ in spans[1:]}


def sheet(texts_json, out, base_files, extra_files, trim_file=None):
    texts = json.load(open(texts_json))
    trim = json.load(open(trim_file)) if trim_file else {}
    base = [json.load(open(f)) for f in base_files]
    extra = [json.load(open(f)) for f in extra_files]
    names = [f.rsplit("p_", 1)[-1].replace(".json", "") for f in base_files + extra_files]
    rows = []
    for t in texts:
        tx = t["text"]
        sets = [starts(p[t["id"]]) for p in base]
        xsets = [starts(p[t["id"]]) for p in extra]
        cut = trim.get(t["id"], 0)
        cand = sorted(c for c in set().union(*sets, *xsets) if c > cut)  # header/title excluded
        # glued punctuation ("faces.Sur"): a typical L2 pattern that rule-based and neural splitters can all miss
        glued = {m.end() for m in re.finditer(r"(?<=[a-zà-ÿ])[.!?](?=[A-ZÀ-Ý][a-zà-ÿ])", tx) if m.end() > cut}
        for c in sorted(glued - set(cand)):
            rows.append({"text_id": t["id"], "offset": c, "left": tx[max(0, c - 90):c], "right": tx[c:c + 90],
                         "proposed_by": "glued-punctuation-rule", "suggest": "y", "decision": ""})
        for c in cand:
            votes = sum(c in s for s in sets)
            if votes == len(sets):
                continue  # all baselines agree: accepted
            who = [n for n, s in zip(names, sets + xsets) if c in s]
            rows.append({"text_id": t["id"], "offset": c, "left": tx[max(0, c - 90):c], "right": tx[c:c + 90],
                         "proposed_by": "+".join(who), "suggest": "y" if votes * 2 > len(sets) else "n", "decision": ""})
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "boundaries to review ->", out)


def gold(texts_json, sheet_tsv, out, base_files):
    texts = json.load(open(texts_json))
    base = [json.load(open(f)) for f in base_files]
    dec = {}
    for r in csv.DictReader(open(sheet_tsv, encoding="utf-8", newline=""), delimiter="\t"):
        d = (r["decision"] or r["suggest"]).strip().lower()
        dec[(r["text_id"], int(r["offset"]))] = d == "y"
    res = {}
    for t in texts:
        sets = [starts(p[t["id"]]) for p in base]
        agreed = set.intersection(*sets)
        cuts = set(agreed) | {o for (tid, o), v in dec.items() if tid == t["id"] and v}
        cuts = sorted(c for c in cuts if not any((t["id"], c) in dec and not dec[(t["id"], c)] for _ in [0]))
        bounds = [0] + cuts + [len(t["text"])]
        spans = []
        for a, b in zip(bounds, bounds[1:]):
            seg = t["text"][a:b]
            if seg.strip():  # trim whitespace so spans compare equal to the (trimmed) predicted spans
                a2 = a + len(seg) - len(seg.lstrip())
                spans.append([a2, a2 + len(seg.strip())])
        res[t["id"]] = spans
    json.dump(res, open(out, "w"))
    print(sum(len(v) for v in res.values()), "gold sentences in", len(res), "texts")


if __name__ == "__main__":
    a = sys.argv[1:]
    trim_file = a[a.index("--trim") + 1] if "--trim" in a else None
    a = [x for i, x in enumerate(a) if x != "--trim" and (i == 0 or a[i - 1] != "--trim")]
    extra = a[a.index("--extra") + 1:] if "--extra" in a else []
    pos = a[:a.index("--extra")] if "--extra" in a else a
    if pos[0] == "sheet":
        sheet(pos[1], pos[2], pos[3:], extra, trim_file)
    else:
        gold(pos[1], pos[2], pos[3], pos[4:])
