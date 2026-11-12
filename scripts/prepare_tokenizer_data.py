#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Build paragraph-structured tokenizer training text from a sentence-level CoNLL-U file.

Trankit's tokenizer learns sentence boundaries from *running text*. If every paragraph in the
training txt holds exactly ONE sentence, the model never sees a boundary inside a text and
under-splits at inference (one long "sentence"). This script groups consecutive sentences into
paragraphs of 1..max sentences (blank-line separated) so boundaries occur in context.
The CoNLL-U is passed through unchanged (sentence order preserved).

usage: prepare_tokenizer_data.py IN.conllu OUT.txt [--min 1 --max 12 --seed 1]
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conllu")
    ap.add_argument("out_txt")
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    texts = []
    with open(a.conllu, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# text = "):
                texts.append(line[len("# text = "):].rstrip("\n"))
    rng = random.Random(a.seed)
    paras, i = [], 0
    while i < len(texts):
        k = rng.randint(a.min, a.max)
        paras.append(" ".join(texts[i:i + k]))
        i += k
    with open(a.out_txt, "w", encoding="utf-8") as out:
        out.write("\n\n".join(paras) + "\n")
    print(f"{len(texts)} sentences -> {len(paras)} paragraphs (mean {len(texts) / len(paras):.1f} sentences)")


if __name__ == "__main__":
    main()
