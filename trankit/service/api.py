# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
# Derived from the ALSI trankit_fr runtime.
"""JSON Lines API: one JSON request per stdin line, one JSON response per stdout line.

Requests: {"id":..., "text":..., "doc_id":"doc", "split":"syntok|spacy|lines|none"}
          {"id":..., "op":"info"}   -> runtime/model/device description
Responses: {"id":..., "annotations":[...]} | {"id":..., "info":{...}} | {"id":..., "error":{...}}
"""

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path  # noqa: E402

from . import backend  # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get("TRANKIT_MODEL_DIR") or str(
    Path(os.environ.get("TRANKIT_HOME", Path.cwd())) / "model")
SPLITS = {"lines", "syntok", "spacy", "none"}


def annotate(text, doc_id="doc", split="syntok", device=None, model_dir=None,
             category="customized-mwt", embedding="xlm-roberta-base"):
    """Return token-level UD annotations (list of dicts) for one text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(doc_id, str):
        raise TypeError("doc_id must be a string")
    if split not in SPLITS:
        raise ValueError("split must be 'lines', 'syntok', 'spacy', or 'none'")
    if device not in {None, "cpu", "mps", "cuda"}:
        raise ValueError("device must be cpu, mps or cuda")
    if not text.strip():
        raise ValueError("text must not be empty")

    options = dict(texts=[text], doc_ids=[doc_id], model_dir=str(model_dir or DEFAULT_MODEL_DIR),
                   category=category, gpu=False, device=device, embedding=embedding)
    if split == "none":
        return backend.trankit_parse_texts(**options)
    sentences = [l.strip() for l in text.splitlines() if l.strip()] if split == "lines" else None
    return backend.trankit_parse_presplit(
        **options, sentences={doc_id: sentences} if sentences else None, sentence_splitter=split)


def info(device=None, model_dir=None, category="customized-mwt", embedding="xlm-roberta-base"):
    import torch
    import trankit
    return {
        "trankit": getattr(trankit, "__version__", None),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "model_dir": str(model_dir or DEFAULT_MODEL_DIR),
        "category": category,
        "embedding": embedding,
        "device_requested": device,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
    }


def serve(device=None, model_dir=None, category="customized-mwt", embedding="xlm-roberta-base"):
    cfg = dict(device=device, model_dir=model_dir, category=category, embedding=embedding)
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            if request.get("op") == "info":
                response = {"id": request_id, "info": info(**cfg)}
            else:
                annotations = annotate(request.get("text"), doc_id=request.get("doc_id", "doc"),
                                       split=request.get("split", "syntok"), **cfg)
                response = {"id": request_id, "annotations": annotations}
        except Exception as exc:
            response = {"id": request_id, "error": {"type": type(exc).__name__, "message": str(exc)}}
        try:
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except (BrokenPipeError, OSError):
            return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--category", default="customized-mwt")
    parser.add_argument("--embedding", default="xlm-roberta-base")
    a = parser.parse_args(argv)
    serve(device=a.device, model_dir=a.model_dir, category=a.category, embedding=a.embedding)


if __name__ == "__main__":
    main()
