# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
# Derived from the ALSI trankit_fr runtime.
"""Trankit -> CoNLL-U batch CLI (python -m trankit.service.cli).

Input  : a .txt file, a directory of .txt files, or a .tsv/.csv with doc_id,text.
Output : one CoNLL-U file per document in --out, plus manifest.tsv.
Sentence splitting: --split syntok (default; ALSI segmenter), spacy (bundled
ALSI model), lines (one sentence per nonblank line), or none (Trankit's own splitter).
"""
import argparse, csv, os, sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from . import backend  # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get("TRANKIT_MODEL_DIR") or str(
    Path(os.environ.get("TRANKIT_HOME", Path.cwd())) / "model")


def val(v):
    v = "" if v is None else str(v)
    return "_" if v in {"", "NA", "None"} else v


def read_docs(path: Path, split: str):
    """Yield (doc_id, text, sentences-or-None)."""
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    for f in files:
        if f.suffix.lower() in {".tsv", ".csv"}:
            delim = "\t" if f.suffix.lower() == ".tsv" else ","
            with f.open(encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh, delimiter=delim):
                    text = r["text"]
                    yield r["doc_id"], text, ([l.strip() for l in text.splitlines() if l.strip()] if split == "lines" else None)
        else:
            text = f.read_text(encoding="utf-8-sig")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            yield f.stem, " ".join(lines), (lines if split == "lines" else None)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Remote batch execution (through bin/trankit-run):\n"
            "  bin/trankit-run INPUT --out OUTDIR --remote [USER@]HOST\n"
            "  Add --remote-dir DIR to choose the remote install path, or --remote-sync "
            "to recopy runtime/ and model/.\n"
            "  The remote run defaults to MPS. SSH/rsync are required; connection "
            "failure is an error, not a local fallback.\n"
            "  The JSON Lines API (bin/trankit-run-api) can also run on the remote over ssh."
        ),
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["lines", "syntok", "spacy", "none"], default="syntok")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Trankit cache dir holding <embedding>/<category>/ (env TRANKIT_MODEL_DIR; default $TRANKIT_HOME/model)")
    ap.add_argument("--category", default="customized-mwt", help="saved pipeline category (default customized-mwt)")
    ap.add_argument("--embedding", default="xlm-roberta-base",
                    help="encoder: xlm-roberta-base/large, or a HF id / local dir with a SentencePiece tokenizer")
    ap.add_argument("--max-sent-tokens", type=int, default=120,
                    help="warn when a sentence exceeds this many tokens (0 disables; default 120)")
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of warning when a sentence exceeds --max-sent-tokens")
    ap.add_argument("--overwrite", action="store_true", help="reparse docs whose .conllu already exists")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / "manifest.tsv"
    done = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            done = {r["doc_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    new_manifest = not manifest.exists()
    mf = manifest.open("a", encoding="utf-8", newline="")
    w = csv.writer(mf, delimiter="\t")
    if new_manifest:
        w.writerow(["doc_id", "sentences", "tokens", "conllu"])

    n = 0
    for doc_id, text, sents in read_docs(a.input, a.split):
        safe = doc_id.replace("/", "_")
        out = a.out / f"{safe}.conllu"
        if out.exists() and doc_id in done and not a.overwrite:
            print(f"skip {doc_id} (done)", file=sys.stderr); continue
        if not text.strip():
            raise ValueError(f"{doc_id}: empty text")
        kw = dict(texts=[text], doc_ids=[doc_id], model_dir=str(a.model_dir), category=a.category,
                  gpu=False, device=a.device, embedding=a.embedding)
        if a.split == "none":
            rows = backend.trankit_parse_texts(**kw)
        else:
            rows = backend.trankit_parse_presplit(
                **kw,
                sentences={doc_id: sents} if sents else None,
                sentence_splitter=a.split,
            )
        if not rows:
            raise RuntimeError(f"{doc_id}: no tokens returned")
        by = {}
        for r in sorted(rows, key=lambda r: (r["sentence_id"], r["token_id"])):
            by.setdefault(int(r["sentence_id"]), []).append(r)
        if sents and len(by) != len(sents):
            raise RuntimeError(f"{doc_id}: parsed {len(by)} of {len(sents)} sentences")
        if a.max_sent_tokens > 0:
            long_ = [(sid, len(t)) for sid, t in by.items() if len(t) > a.max_sent_tokens]
            if long_:
                for sid, k in long_:
                    print(f"WARNING {doc_id}-{sid}: sentence has {k} tokens (> {a.max_sent_tokens})", file=sys.stderr)
                msg = (f"{doc_id}: {len(long_)} sentence(s) > {a.max_sent_tokens} tokens "
                       f"(max {max(k for _, k in long_)}). Input lines may be whole paragraphs; "
                       f"use --split syntok/spacy or re-split. Downstream tools (e.g. CorPipe) may crash.")
                if a.strict:
                    raise RuntimeError(msg)
                print("WARNING " + msg, file=sys.stderr)
        lines = [f"# newdoc id = {doc_id}"]
        for sid, toks in by.items():
            lines += [f"# sent_id = {doc_id}-{sid}"]
            if toks[0].get("sentence"):
                lines.append(f"# text = {toks[0]['sentence']}")
            for r in toks:
                lines.append("\t".join([str(r["token_id"]), val(r["token"]), val(r["lemma"]), val(r["upos"]),
                                        val(r["xpos"]), val(r["feats"]), str(r["head_token_id"]),
                                        val(r["dep_rel"]), "_", "_"]))
            lines.append("")
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(out)  # atomic: no half-written files on interruption
        w.writerow([doc_id, len(by), len(rows), out.name]); mf.flush()
        n += 1
        print(f"{doc_id}: {len(by)} sentences, {len(rows)} tokens", file=sys.stderr)
    print(f"done: {n} parsed", file=sys.stderr)


if __name__ == "__main__":
    main()
