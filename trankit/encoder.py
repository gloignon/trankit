# Added for this fork of Trankit (https://github.com/nlp-uoregon/trankit), Apache License 2.0.
"""Pluggable transformer encoder support.

Stock Trankit hardcodes XLM-RoBERTa. Here ``embedding`` may be

* ``xlm-roberta-base`` / ``xlm-roberta-large`` (the official models, unchanged), or
* any Hugging Face model id or local directory whose tokenizer is SentencePiece
  based (``▁`` word-start marker), e.g. ``almanach/camembert-base``.

``embedding_name`` stays the short key used for cache directories; ``embedding_source``
is what is passed to ``from_pretrained``.
"""

import os
import re

from transformers import AutoConfig, AutoModel, AutoTokenizer

from .utils.tbinfo import supported_embeddings

_KNOWN_DIMS = {'xlm-roberta-base': 768, 'xlm-roberta-large': 1024}


def is_official(embedding):
    return embedding in supported_embeddings


def embedding_key(embedding):
    """Directory-safe key for an embedding (identity for the official ones)."""
    if is_official(embedding):
        return embedding
    path = os.path.abspath(embedding) if os.path.isdir(embedding) else embedding
    return re.sub(r'[^A-Za-z0-9._-]+', '--', path).strip('-')


def set_embedding(config, embedding):
    """Record the embedding on a Config: key in ``embedding_name``, real id in ``embedding_source``."""
    config.embedding_source = embedding
    config.embedding_name = embedding_key(embedding)
    config.embedding_dim = None


def embedding_dim(config):
    """Hidden size of the encoder (cached on the config)."""
    dim = getattr(config, 'embedding_dim', None)
    if dim is None:
        source = getattr(config, 'embedding_source', config.embedding_name)
        if source in _KNOWN_DIMS:
            dim = _KNOWN_DIMS[source]
        else:
            dim = AutoConfig.from_pretrained(source, cache_dir=_hf_cache(config)).hidden_size
        config.embedding_dim = dim
    return dim


def adapter_reduction_factor(dim):
    return 6 if dim <= 768 else 4  # matches stock base (768 -> 6) and large (1024 -> 4)


def _hf_cache(config):
    return os.path.join(config._cache_dir, config.embedding_name)


def load_encoder(config):
    return AutoModel.from_pretrained(config.embedding_source, cache_dir=_hf_cache(config),
                                     output_hidden_states=True)


def load_tokenizer(config, cache_dir=None):
    tok = AutoTokenizer.from_pretrained(config.embedding_source, cache_dir=cache_dir or _hf_cache(config),
                                        use_fast=False)  # slow tokenizer: matches stock Trankit behavior
    pieces = tok.tokenize('hello world')
    if not any(p.startswith('▁') for p in pieces):
        raise ValueError(
            "Encoder {!r} does not use a SentencePiece tokenizer ('▁' word markers); "
            "Trankit's tokenizer utilities require one (got pieces {}).".format(config.embedding_source, pieces))
    return tok
