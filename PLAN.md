# Plan: patched Trankit as a deployable runtime (CPU/CUDA/MPS, pluggable encoder, local + remote clients)

Branch `mps-device`. `trankit_fr` stays untouched as the working fallback until phase 6 passes.

## Deliverable
One repo that provides:
1. The patched `trankit` library (device selection, pluggable SentencePiece encoder).
2. A self-contained runtime: CLI, JSON Lines API, resumable batch parse, optional remote execution over SSH.
3. Thin clients (Python, R, agent-friendly CLI) in `/Volumes/Data/pipelines/clients` that call it locally or on a named remote host.
4. The French model as an optional, separately downloaded asset. The user may point at any model dir/encoder.

## Layout (target)
```
trankit/                  patched library (done: device.py, encoder.py)
trankit/service/          backend.py (parse logic), parse.py (batch CLI), api.py (JSONL API), info
bin/trankit-run           batch CLI wrapper (local or --remote)
bin/trankit-run-api       JSONL API wrapper (stdin/stdout; works over ssh)
bin/trankit-run-remote    SSH/rsync runner (ported from trankit_fr)
scripts/build_runtime.sh  builds runtime/ (python3.10 venv + pip install .)
```
Deployment dir ("TRANKIT_HOME") = repo checkout or synced copy; holds `runtime/` and, by default, `model/`.
Model location: `--model-dir` / `TRANKIT_MODEL_DIR`, default `$TRANKIT_HOME/model`.

## Phases

### 1. Library (DONE)
- [x] Device selection cpu/cuda/mps; MWT + lemmatizer on CPU under mps.
- [x] gc.collect throttle.
- [x] Apache modification headers; torch pin relaxed.
- [x] Pluggable encoder (`trankit/encoder.py`), regression-identical on the French model.
- [x] Real encoders verified: `almanach/camembert-base` (768) and `camembert-large` (1024) load from the shared HF cache (`HF_HOME=/Volumes/Data/research_data/hf_cache`); CamemBERT routed through the XLM-R/RoBERTa class because `adapters` lacks a `camembert` module. A parser on them still needs training.
- [ ] Equivalence check CPU vs MPS on the French model after the encoder change (phase 6).

### 2. Service code into the package (DONE)
- Port `trankit_backend.py`, `parse.py`, `api.py` into `trankit/service/` as importable modules (no `sys.path` hacks, no hardcoded ROOT).
- Parameters: `model_dir`, `category` (default `customized-mwt`), `embedding` (default `xlm-roberta-base`, or any HF id/local dir), `device`.
- API: accept `device` (cpu/mps/cuda) and `model_dir`/`embedding` startup flags; add a `{"op":"info"}` request (versions, device, model, encoder).
- Keep output schema identical to `trankit_fr` (columns, `NA` handling, MWT expansion).
- Gate: unit tests + output identical to `trankit_fr` on sample text (CPU).

### 3. Runtime + shell entry points (DONE)
Design: `runtime/` = third-party deps only (`requirements-runtime.txt`, pinned from the working trankit_fr runtime); `trankit/` is loaded from the checkout, so code sync to a remote is a cheap rsync and needs no reinstall. Model dir is external (`TRANKIT_MODEL_DIR`).
Verified 2026-11-12: freshly built runtime; `bin/trankit-run` CoNLL-U output byte-identical to `trankit_fr` (syntok + lines, CPU); MPS == CPU on the sample.
- `scripts/build_runtime.sh`: Python 3.10 venv, `pip install .[service]` (extras: syntok, optional spacy).
- `bin/trankit-run`, `bin/trankit-run-api`, `bin/trankit-run-remote` ported from `trankit_fr` (env vars renamed `TRANKIT_REMOTE`, `TRANKIT_REMOTE_DIR`, `TRANKIT_MODEL_DIR`; old names still read).
- Gate: batch CLI output (CoNLL-U + manifest) identical to `trankit_fr`.

### 4. Remote (DONE)
Verified 2026-11-12 on c64917.local (Apple M5, MPS): batch output == local CPU; Python and R clients stream through `ssh ... trankit-run-api` (device defaults to mps). Remote install: `~/trankit_run` (2.8 GB; standalone python via uv was not needed).
- Remote host chosen by the user (`--remote`, `TRANKIT_REMOTE`, or client `remote=`); default device on the remote = mps if available.
- Batch mode: existing SSH/rsync flow (resume via manifest).
- API mode: `ssh HOST $DIR/bin/trankit-run-api` gives a persistent remote JSONL stream for Python/R/agents.
- Sync: code always; `runtime/` and `model/` only with `--remote-sync`. Never touch the remote `trankit_fr`.
- Gate: run on the MacBook; results match local CPU.

### 5. Clients (`/Volumes/Data/pipelines/clients`) (DONE; local and remote verified)
New files `trankit_client.py` / `trankit_client.R` (`trankit_run`, `trankit_annotate`, `TrankitClient`); `nlp_pipelines.*` and the legacy `trankit_fr()` are untouched until cutover (phase 6).
- `nlp_pipelines.py/.R`: point to this repo (`TRANKIT_HOME`, default `/Volumes/Data/pipelines/trankit`), keep `trankit_fr(src, out_dir, split, device, overwrite)` signature, add `remote=`, `model_dir=`, `embedding=`.
- New text-level calls: `trankit_annotate(texts, ...)` returning a DataFrame / data.frame, backed by one persistent API process (local or ssh), closed at end of job (pattern from `embedder.py`/`embedder.R`).
- Agent use: the CLI is self-documenting (`--help`, `info`, JSON output). MCP server only if requested.
- Gate: existing callers run unchanged against the new runtime.

### 6. Verification and cutover
- 600-sentence benchmark: CPU vs MPS speed, CPU vs MPS output equality, vs `trankit_fr` baseline (stock/CPU 144 s, patched/CPU 96 s, mps 74 s).
- Custom encoder smoke test with a real French SentencePiece encoder (e.g. CamemBERT); a trained model on it requires retraining (separate job).
- Only after the above: repoint clients permanently, archive `trankit_fr`.

### 7. Packaging and compliance
- Rename package (decision pending, e.g. `trankit-fr`/other), setup.py metadata, README attribution to nlp-uoregon.
- Check licenses of UD treebank data / French model before redistributing weights (HANDOFF task 7). Keep LICENSE.

## Open decisions
- Final package/CLI name (working names: `trankit-run`).
- MCP server for agents (deferred).
- Which French encoder to retrain on, if any.
