#!/usr/bin/env python3
# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Train a Trankit pipeline (tokenize, mwt, posdep, lemmatize) on a chosen encoder and device.

Run with the runtime python from the repo root:
  runtime/bin/python scripts/train.py --train-conllu T.conllu --dev-conllu D.conllu \
      --train-txt T.txt --dev-txt D.txt --output models/camembert_fr \
      --embedding almanach/camembert-base --device mps --category customized-mwt

Output: <output>/<embedding key>/<category>/ ; use it with
  bin/trankit-run ... --model-dir <output> --embedding almanach/camembert-base --category customized-mwt
Tokenizer data must have multi-sentence paragraphs (see prepare_tokenizer_data.py).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import trankit  # noqa: E402
from trankit.encoder import embedding_key  # noqa: E402

BATCH = {"tokenize": 16, "mwt": 50, "posdep": 8, "lemmatize": 16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-conllu", type=Path, required=True)
    ap.add_argument("--dev-conllu", type=Path, required=True)
    ap.add_argument("--train-txt", type=Path)
    ap.add_argument("--dev-txt", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--embedding", default="xlm-roberta-base")
    ap.add_argument("--category", default="customized-mwt")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--tasks", nargs="+", default=["tokenize", "mwt", "posdep", "lemmatize"],
                    choices=["tokenize", "mwt", "posdep", "lemmatize"])
    ap.add_argument("--max-epoch", type=int, default=None, help="Trankit default is 100")
    ap.add_argument("--batch-size", nargs="*", default=[], metavar="TASK=N")
    a = ap.parse_args()
    bs = dict(BATCH)
    bs.update({k: int(v) for k, v in (x.split("=") for x in a.batch_size)})
    a.output.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        if task == "tokenize" and not (a.train_txt and a.dev_txt):
            sys.exit("tokenize needs --train-txt and --dev-txt (paragraph-structured; see prepare_tokenizer_data.py)")
        cfg = {"category": a.category, "task": task, "save_dir": str(a.output.resolve()),
               "train_conllu_fpath": str(a.train_conllu.resolve()), "dev_conllu_fpath": str(a.dev_conllu.resolve()),
               "embedding": a.embedding, "batch_size": bs[task]}
        if a.device:
            cfg["device"] = a.device
        else:
            cfg["gpu"] = False
        if task == "tokenize":
            cfg["train_txt_fpath"] = str(a.train_txt.resolve())
            cfg["dev_txt_fpath"] = str(a.dev_txt.resolve())
        if a.max_epoch:
            cfg["max_epoch"] = a.max_epoch
        print(f"\n=== {task} | encoder {a.embedding} | device {a.device} | batch {bs[task]} ===", file=sys.stderr, flush=True)
        trankit.TPipeline(training_config=cfg).train()
    if set(a.tasks) >= {"tokenize", "mwt", "posdep", "lemmatize"}:
        trankit.verify_customized_pipeline(category=a.category, save_dir=str(a.output.resolve()),
                                           embedding_name=embedding_key(a.embedding))
    print(f"done: {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
