#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Score systems against the L2 gold sentence spans, ignoring headers (everything before the trim offset).
  seg_l2_score.py GOLD.json TRIM.json P_a.json [P_b.json ...]   (system name = file name after 'p_')
Exact-span F1 and boundary F1 (sentence starts), overall and by writer group (native / learner)."""
import json
import sys


def clip(spans, cut):
    out = []
    for s, e in spans:
        if e <= cut:
            continue
        out.append((max(s, cut), e))
    return out


def prf(tp, npred, ngold):
    p, r = tp / max(npred, 1), tp / max(ngold, 1)
    return round(100 * p, 1), round(100 * r, 1), round(100 * 2 * p * r / max(p + r, 1e-9), 1)


if __name__ == "__main__":
    gold, trim = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    texts = {t["id"]: t for t in json.load(open(sys.argv[3].rsplit("/", 1)[0] + "/l2_texts.json"))}
    for f in sys.argv[3:]:
        pred = json.load(open(f))
        name = f.rsplit("p_", 1)[-1].replace(".json", "")
        for grp in ("all", "native", "learner"):
            ex = [0, 0, 0]
            bd = [0, 0, 0]
            for tid, g in gold.items():
                if grp != "all" and texts[tid]["group"] != grp:
                    continue
                cut = trim.get(tid, 0)
                G = set(map(tuple, clip(g, cut)))
                P = set(clip(pred[tid], cut))
                ex[0] += len(G & P); ex[1] += len(P); ex[2] += len(G)
                gb, pb = {s for s, _ in G} - {min((s for s, _ in G), default=0)}, {s for s, _ in P} - {min((s for s, _ in P), default=0)}
                bd[0] += len(gb & pb); bd[1] += len(pb); bd[2] += len(gb)
            print(f"{name:12s} {grp:8s} span P/R/F1 {prf(*ex)}  boundary P/R/F1 {prf(*bd)}  (gold sents {ex[2]})")
