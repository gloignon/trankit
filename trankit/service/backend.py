# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
# Derived from the ALSI trankit_fr runtime.
"""Trankit parsing backend (token-level rows, sentence splitters, cached pipeline).

The pipeline is initialised lazily on first call and cached in the module-level
_pipeline variable so it is not reloaded on every call.
"""

import io
import os
import re
import sys
from pathlib import Path

_SKIP = ("Active language:", "=" * 10)


class _FilteredStderr(io.TextIOBase):
    """Forwards stdout to stderr, dropping Trankit's internal banner lines."""
    def write(self, s):
        if s and not any(s.startswith(p) for p in _SKIP) and s.strip():
            print(s.rstrip(), file=sys.stderr, flush=True)
        return len(s)


_pipeline = None
_model_dir = None
_category = None
_device = None
_embedding = None
_SPACY_SENTENCE_MODEL = Path(os.environ.get("TRANKIT_SPACY_MODEL", "")) if os.environ.get("TRANKIT_SPACY_MODEL") else None
_NONSPLIT_END_RE = re.compile(
    r"(?<=[^\w\s,;:\-])([.!?])(?=\s+[A-ZÀ-ÖØ-ÝŸ])",
    re.UNICODE,
)


def trankit_load(model_dir: str, category: str = "customized-mwt", gpu: bool = False,
                 device=None, embedding: str = "xlm-roberta-base"):
    """Load (or reload) the Trankit pipeline. Safe to call multiple times."""
    global _pipeline, _model_dir, _category, _device, _embedding

    if (_pipeline is not None and _model_dir == model_dir and _category == category
            and _device == device and _embedding == embedding):
        return _pipeline

    from trankit import Pipeline
    print("Loading ALSI French model (Trankit) …", file=sys.stderr, flush=True)
    import contextlib
    with contextlib.redirect_stdout(_FilteredStderr()):
        options = {"lang": category, "cache_dir": model_dir, "gpu": gpu, "embedding": embedding}
        if device is not None:
            options["device"] = device
        _pipeline = Pipeline(**options)
    if device is not None and _pipeline._device.type != device:
        raise RuntimeError("Trankit initialized on {!r}, requested {!r}".format(
            _pipeline._device.type, device))
    _model_dir = model_dir
    _category = category
    _device = device
    _embedding = embedding
    print("Trankit pipeline ready.", file=sys.stderr, flush=True)
    return _pipeline


def trankit_parse_texts(texts, doc_ids, model_dir: str,
                        category: str = "customized-mwt", gpu: bool = False,
                        device=None, embedding: str = "xlm-roberta-base"):
    """Parse a list of texts and return a list of token-level dicts.

    Parameters
    ----------
    texts    : list[str]  — one entry per document
    doc_ids  : list[str]  — parallel doc identifiers
    model_dir: str        — path to trankit_fr_v1 (or equivalent)
    category : str        — category name used at training time
    gpu      : bool       — whether to use GPU for inference
    device   : str|None   — explicit cpu/cuda/mps device, when supported

    Returns
    -------
    list[dict] — one dict per token with keys matching udpipe output columns:
        doc_id, sentence_id, token_id, token, lemma, upos, xpos,
        feats, head_token_id, dep_rel
    """
    trankit_load(model_dir, category, gpu, device, embedding)

    rows = []
    for doc_id, text in zip(doc_ids, texts):
        if not text or not text.strip():
            continue

        doc_result = _pipeline(text)
        global_sent_id = 0

        for sent in doc_result.get("sentences", []):
            global_sent_id += 1
            tokens = sent.get("tokens", [])
            local_id = 0

            for tok in tokens:
                # MWT tokens have an 'expanded' list — iterate over subtokens
                if "expanded" in tok:
                    for sub in tok["expanded"]:
                        local_id += 1
                        rows.append(_make_row(doc_id, global_sent_id, local_id, sub))
                else:
                    local_id += 1
                    rows.append(_make_row(doc_id, global_sent_id, local_id, tok))

    return rows


def syntok_sentences(text):
    """Split text with ALSI's syntok segmenter without merging short sentences."""
    if not text:
        return []
    import syntok.segmenter as segmenter
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NONSPLIT_END_RE.sub(r"\1\n\n", text)
    sentences = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            sentence_text = "".join(map(str, sentence)).strip()
            if sentence_text:
                sentences.append(sentence_text)
    return sentences


_nlp_sents = None


def spacy_sentences(text):
    """Split text with the optional bundled ALSI spaCy sentence model."""
    global _nlp_sents
    if _nlp_sents is None:
        if _SPACY_SENTENCE_MODEL is None:
            raise RuntimeError("split='spacy' needs TRANKIT_SPACY_MODEL=<path to a spaCy sentence model>")
        import spacy
        _nlp_sents = spacy.load(
            str(_SPACY_SENTENCE_MODEL),
            exclude=["ner", "lemmatizer", "attribute_ruler", "morphologizer"],
        )
    sentences = (re.sub(r"\s+", " ", sentence.text).strip() for sentence in _nlp_sents(text).sents)
    return [sentence for sentence in sentences if sentence]


def trankit_parse_presplit(texts, doc_ids, model_dir: str,
                           category: str = "customized-mwt", gpu: bool = False,
                           device=None, sentences=None, sentence_splitter="syntok",
                           embedding: str = "xlm-roberta-base"):
    """As trankit_parse_texts(), but each document is first split into sentences
    with the selected splitter and every sentence is parsed on its own (is_sent=True).
    sentences: optional {doc_id: [sentence, ...]} overrides automatic splitting.
    Rows carry the sentence text in "sentence" for the CoNLL-U "# text" line."""
    if sentence_splitter not in {"lines", "syntok", "spacy"}:
        raise ValueError("sentence_splitter must be 'lines', 'syntok', or 'spacy'")
    trankit_load(model_dir, category, gpu, device, embedding)

    rows = []
    for doc_id, text in zip(doc_ids, texts):
        if not text or not text.strip():
            continue
        given = sentences.get(str(doc_id)) if sentences else None
        if given is not None:
            split_sentences = given
        elif sentence_splitter == "spacy":
            split_sentences = spacy_sentences(text)
        elif sentence_splitter == "lines":
            split_sentences = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            split_sentences = syntok_sentences(text)
        for sentence_id, sentence in enumerate(split_sentences, start=1):
            local_id = 0
            for tok in _pipeline(sentence, is_sent=True).get("tokens", []):
                for sub in tok.get("expanded", [tok]):
                    local_id += 1
                    row = _make_row(doc_id, sentence_id, local_id, sub)
                    row["sentence"] = sentence
                    rows.append(row)
    return rows


def _make_row(doc_id, sentence_id, token_id, tok):
    head_raw = tok.get("head", 0)
    # Trankit uses 0 for root; keep as 0 to match udpipe convention
    return {
        "doc_id":        str(doc_id),
        "sentence_id":   int(sentence_id),
        "token_id":      int(token_id),
        "token":         str(tok.get("text", "")),
        "lemma":         str(tok.get("lemma", "")),
        "upos":          str(tok.get("upos", "")),
        "xpos":          str(tok.get("xpos", "")),
        "feats":         str(tok.get("feats", "")) if tok.get("feats") else NA_STRING,
        "head_token_id": int(head_raw),
        "dep_rel":       str(tok.get("deprel", "")),
    }


# Sentinel for missing values — R will coerce "NA" strings to NA
NA_STRING = "NA"
