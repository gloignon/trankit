#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Locate where the body of each letter/essay starts, so headers (name, address, date, addressee, salutation,
essay title) are excluded from sentence-segmentation scoring.  usage: seg_l2_trim.py TEXTS.json OUT_TRIM.json"""
import json
import re
import sys

SALUT = re.compile(
    r"(?:(?:Cher|Chère|Bonjour)\s+)*(?:Madame|Monsieur|Mme|M\.)(?:\s*/\s*(?:Madame|Monsieur))?[^,.!?;:]{0,40},"
    r"(?:\s*(?:bonjour|Bonjour|Madame|Monsieur)[^,.!?;:]{0,20},)*")


def body_start(t):
    tx = t["text"]
    if t["genre"] == "essai_argumente":
        m = re.search(r"\?", tx[:120])
        return m.end() if m else 0
    last = None
    for m in SALUT.finditer(tx[:300]):
        last = m
    return last.end() if last else 0


if __name__ == "__main__":
    T = json.load(open(sys.argv[1]))
    trim = {t["id"]: body_start(t) for t in T}
    json.dump(trim, open(sys.argv[2], "w"))
    for t in T:
        c = trim[t["id"]]
        print(f"{c:4d} | cut: ...{t['text'][max(0, c - 45):c]!r} | body: {t['text'][c:c + 45]!r}")
