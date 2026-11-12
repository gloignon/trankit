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

## Benchmark log (2026-11-12, 600 sentences: UD French GSD test (416) + first 184 dev; one sentence per line)
| model | device | time | aligned sents | UPOS | LEMMA | UAS | LAS |
|---|---|---|---|---|---|---|---|
| custom French (`customized-mwt`) | CPU | 94 s | 575/600 | .9686 | .9718 | .9514 | .9364 |
| custom French | MPS | 89 s | 575/600 | .9686 | .9718 | .9514 | .9364 |
| stock multilingual `french` | CPU | 120 s | 563/600 | .9781 | .9574 | .9433 | .9238 |
| stock multilingual `french` | MPS | 117 s | 563/600 | .9781 | .9574 | .9433 | .9238 |
CPU and MPS outputs are byte-identical for both models. Scores only cover sentences whose token count matches gold (sets differ, so not strictly comparable). Bug found and fixed: `masked_scatter_` in `crf_layer.py` is unsupported on MPS (stock model loads NER).

## Phase 8: train a new French model (CamemBERT encoder, real sentence splitting, MPS on the MacBook)
Goal: a Trankit model that (1) uses `almanach/camembert-base` (large/distil optional), (2) segments sentences inside running text, (3) trains on MPS on c64917.local.

Finding (2026-11-12): the existing custom model's tokenizer was trained on 14,586 paragraphs of exactly ONE sentence each (mean 1.0 in `fr_gsd-ud-train.txt.character`), so it never saw a sentence boundary inside a text and under-splits. ALSI works around it with syntok + per-sentence parsing (`is_sent=True`). The posdep/lemma tasks are unaffected (sentence-level by nature).
Rule: tokenizer training text MUST be multi-sentence paragraphs; the training-data prep script enforces it.

Steps
- [x] `scripts/prepare_tokenizer_data.py`: group consecutive sentences into paragraphs of 1..12 (mean ~6.6), blank-line separated; conllu unchanged.
- [x] `scripts/train.py`: tokenize / mwt / posdep / lemmatize, `--embedding`, `--device`, `--max-epoch`, per-task batch size.
- [ ] Local smoke test on MPS (300 train sentences, 1 epoch, all 4 tasks).
- [ ] Full run on the MacBook (data: ALSI `data/augmented` train+dev, test = `data/filtered/fr_gsd-ud-test.conllu`); pick epochs/batch from smoke timings.
- [ ] Evaluate: (a) gold-tokenization UPOS/LEMMA/UAS/LAS (as ALSI `eval_trankit.py`), (b) sentence-boundary F1 on multi-sentence paragraphs, (c) against current XLM-R model + syntok.
- [ ] Optional: reuse existing MWT/lemma models (encoder-independent) if retraining them is slow on CPU.

### Phase 8 status notes (2026-11-12)
- RULE: no training or smoke tests on the Mac mini (16 GB; MPS runs freeze it). All training/smoke runs happen on the MacBook (c64917.local).
- MPS fixes found while smoke-testing (code committed, NOT yet verified end to end):
  1. torch 2.0.1 MPS `cross_entropy` crashes (SIGBUS) -> `DeviceSafeCrossEntropyLoss` computes the loss on CPU (device.py).
  2. `transformers.AdamW.step` on MPS clears `requires_grad` on all params after step 1 (batch 2 then fails) -> use `torch.optim.AdamW(eps=1e-6)` (tpipeline.py).
  3. Observed ~14 s/iteration (batch 16) on the mini for the tokenizer task; MacBook speed unknown.
- If MPS training stays unreliable on torch 2.0.1, try a newer torch in a separate runtime on the MacBook (the pin was already relaxed in setup.py).

### Full training run (started 2026-11-12 on c64917.local)
`scripts/train_camembert_fr.sh` (in `~/trankit_run`, log `logs/full.log`, output `models/camembert_fr/almanach--camembert-base/customized-mwt/`): tokenizer 20 epochs, posdep 25 epochs, then MWT + lemmatizer copied from the XLM-R model (encoder-independent). Data: ALSI `data/augmented` train/dev, paragraph-structured tokenizer text (mean 6.6 sentences). Smoke test on the MacBook passed (tokenizer + posdep, 1 epoch). Tokenizer ran at ~1 s/batch of 16 (133 batches/epoch).
Next: evaluate (gold-tok UPOS/LEMMA/UAS/LAS on `data/filtered` test; sentence-boundary F1 on multi-sentence paragraphs vs the old model and syntok), then repoint the clients.

## Training gotchas (learned the hard way)
1. **CoNLL-U must end with an empty line.** Trankit scores the dev set after every epoch with the CoNLL-18 evaluator, which raises `UDError: The CoNLL-U file does not end with empty line`. The failure only appears AFTER epoch 0, so a tiny smoke subset that happens to end with a blank line hides it. The ALSI `data/augmented/*.conllu` files end with a single `\n` and need one more `\n` appended. `scripts/train.py` now refuses to start if train/dev CoNLL-U do not end with `\n\n`.
2. **Tokenizer text needs multi-sentence paragraphs** (blank-line separated, mean ~6 sentences), otherwise the tokenizer learns one sentence per text and never splits. Build it with `scripts/prepare_tokenizer_data.py`.
3. **Shell scripts: do not write `[ -f x ] || { cmd && touch x; }` under `set -e`.** A failing `cmd` inside a `||` list does not stop the script, so a crashed task looked finished and the next task started. Use `if [ ! -f x ]; then cmd; touch x; fi`.
4. **MPS (torch 2.0.1, macOS 26):** `cross_entropy` crashes (SIGBUS), so the loss is computed on CPU for MPS tensors; `transformers.AdamW` clears `requires_grad` after step 1, so `torch.optim.AdamW` is used. Do not revert these.
5. **A smoke test must run the whole loop, including dev scoring, for at least 2 epochs** (or at least 1 epoch plus the dev evaluation), and with data prepared the same way as the full run.
6. Shell hook caveat on the Mac mini: top-level `rsync` calls with extra flags may be silently rewritten; use `bin/trankit-run-deploy`.

### Evaluation protocol (decided 2026-11-12)
ALSI's `eval_trankit.py --end-to-end` ("Sent 100" in the ALSI README) feeds one sentence per input and counts inputs with exactly 1 sentence returned, so it cannot detect under-splitting. For the new model, report CoNLL-18 `ud_eval` Sentences/Tokens/Words/UPOS/Lemma/UAS/LAS F1 on RAW multi-sentence paragraphs from `data/filtered/fr_gsd-ud-test.conllu`, for: new CamemBERT model (own splitting), old XLM-R model (own splitting), old model + syntok. Targets (README, same test set family): Sent 93.6 (UDPipe 1) .. 95.27 (UDPipe remix_3) .. 96.63 (Trankit base) .. 98.80 (Trankit large). Epoch 0 dev already gave Sentences F1 93.75 on 6.6-sentence paragraphs.

### Reference benchmarks for French sentence segmentation (collected 2026-11-12)
French-GSD, raw text, CoNLL-18 F1. Different UD releases, so compare as order of magnitude; ALSI's test set is a corrected GSD, so run reference tools on it for a strict comparison.
| System | UD release | Sentences F1 | Tokens/Words F1 | Source |
|---|---|---|---|---|
| Trankit large | 2.5 | 98.80 | 99.77 / 99.72 | trankit.readthedocs.io/en/latest/performance.html |
| Trankit base | 2.5 | 96.63 | 99.70 | ALSI `R/eval_model.R` (not re-checked online) |
| Stanza 1.5.1 | 2.12 | 95.03 | 99.71 / 99.50 | stanfordnlp.github.io/stanza/performance (fetched) |
| Stanza (paper) | 2.5 | 94.92 | 99.68 | Stanza paper / Trankit docs |
| UDPipe 2 `french-gsd-ud-2.17-251125` | 2.17 | 93.82 | words 98.88 | ufal.mff.cuni.cz/udpipe/2/models (fetched) |
| UDPipe 1 | 2.5 | 93.59 | 99.68 | Stanza paper / Trankit docs |
| spaCy v2.3 (old) | 2.5 | 77.30 | 98.34 | Trankit docs |
Not found: current spaCy `fr_core_news_*` senter F1 (see model card accuracy.json), SaT/wtpsplit French UD F1 (SaT paper arXiv 2406.16678 appendix). Both can be run directly on the ALSI test set.

### Measured sentence-segmentation benchmark (2026-11-12)
Data: ALSI `data/filtered/fr_gsd-ud-test.conllu`, 401 gold sentences in 67 raw paragraphs (seed 7, 1-12 sentences, mean 6.0). Metric: exact sentence-span P/R/F1 (`scripts/eval_sentence_segmentation.py`). CPU, no tuning.
| System | Pred. sents | P | R | F1 | Paragraphs exactly right |
|---|---|---|---|---|---|
| Old ALSI Trankit (XLM-R), own splitting | 210 | 59.05 | 30.92 | **40.59** | 14.9% |
| syntok | 381 | 92.65 | 88.03 | 90.28 | 70.1% |
| Stanza 1.15 (fr, tokenize) | 404 | 95.30 | 96.01 | 95.65 | 85.1% |
| SaT `sat-3l-sm` (wtpsplit) | 408 | 95.83 | 97.51 | 96.66 | 88.1% |
| New CamemBERT Trankit | (pending training) | | | | |

### L2 sentence-boundary gold set (CEAAL2) - in progress
LICENSE (checked 2026-11-12): CEAAL2 (Catherine David, LIDILEM / ORTOLANG v2, handle 11403/ceaal2/v2, 292 texts) is **CC BY-NC-ND 3.0 unported** according to the Hugging Face mirror's metadata (FrancophonIA/ceaal2); the ORTOLANG page itself could not be fetched (TLS error), so confirm it there. NC = no commercial use; ND = no derivative works, so do not redistribute annotated/segmented/corrected versions or excerpts. Therefore `eval/` (texts, spans, review sheet) is git-ignored and stays local; publish at most stand-off offsets, after confirming with the depositor. Policy: CEAAL2 is used for EVALUATION only, never as training data (L2 text contains errors; we do not want the model to learn erroneous categories). Robustness comes from perturbing CLEAN GSD data (e.g. no space after a period, missing capital letters, missing final punctuation) with the original gold labels.
Local TSV has 233 of the 292 texts (the PDF extraction may have missed some).
Data: `/Volumes/Data/research_data/CEAAL2/derived/ceaal2.tsv` (233 texts: 160 learner [sinophone, polish, russian, indonesian] + 73 native; formal letters / argumentative essays; median 261 words). Sample: 24 texts (6 native + 18 learner), seed 11, `scripts/seg_l2_sample.py`; per-system spans in `eval/p_*.json`.
Gold = boundaries on which syntok, Stanza and SaT all agree + manual decisions on the disputed ones (`eval/l2_review.tsv`, 40 rows, `suggest` = majority vote; fill `decision` with y/n to override). Then `scripts/seg_l2_review.py gold` builds the gold spans. Old XLM-R Trankit found 226 sentences on these texts vs 368-388 for the others.
Decision (user, 2026-11-12): letter headers and essay titles are EXCLUDED from scoring (`scripts/seg_l2_trim.py`; cut at the salutation or title question). Review sheet now has 25 rows.

### ALSI tagging scheme: copulas are VERB, AUX only for real auxiliaries (decided by the user, 2026-11-12)
Example: "Le chat est gris": standard UD gives AUX for "est"; the ALSI model gives VERB. Implemented by `ALSI_dev/training/R/retag_aux.R` (UPOS AUX + base deprel `cop` -> VERB; `aux:tense/pass/caus` stay AUX). The targets pipeline retags only when `use_retag = TRUE`, and **the files currently on disk (`data/modified|filtered|augmented/*.conllu`) are NOT retagged** (copulas are AUX). So:
- Training data for the new model = `data/augmented/fr_gsd-ud-{train,dev}.conllu` run through `retag_aux.R` (4,786 / 473 copulas -> VERB), plus one extra `\n` at the end (retag_aux.R strips the final blank line, which breaks Trankit's dev scoring). Deployed to the MacBook before the tagger phase started.
- Evaluate on a retagged gold only: `data/filtered/fr_gsd-ud-test_retagged.conllu` (or retag the test file), never on `treebanks/` or non-retagged files.
- CORRECTION to the 600-sentence benchmark above: the custom model's UPOS was scored against official gold (copulas AUX). Against ALSI-scheme gold (221 copulas retagged in the sample) it is **UPOS 98.39** (not 96.86); LEMMA/UAS/LAS are unchanged (97.18 / 95.14 / 93.64). Stock multilingual `french` follows the official scheme, 97.81 against official gold.
Deferred (user, 2026-11-12): causative `faire` (`aux:caus`, 217 tokens in train) stays AUX for now; revisit later (it would need a rule in `retag_aux.R` and a retrain).
