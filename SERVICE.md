# Trankit runtime (this fork)

Build once: `scripts/build_runtime.sh /opt/homebrew/bin/python3.10` (creates `runtime/`, ~600 MB).
Models live outside the repo: set `TRANKIT_MODEL_DIR` (or pass `--model-dir`). Layout: `<model_dir>/<embedding>/<category>/`.

```sh
bin/trankit-run INPUT --out OUT [--split syntok|lines|spacy|none] [--device cpu|mps|cuda]
                [--model-dir DIR] [--category customized-mwt] [--embedding xlm-roberta-base]
                [--remote HOST [--remote-dir DIR] [--remote-sync]]
printf '%s\n' '{"op":"info"}' '{"id":1,"text":"Les enfants jouent."}' | bin/trankit-run-api --device mps
```
`--embedding` accepts `xlm-roberta-base/large` or any HF id / local dir with a SentencePiece tokenizer
(e.g. `almanach/camembert-base`, read from the standard HF cache). A saved Trankit model is tied to the
encoder it was trained on.

Remote: `--remote HOST` (or `TRANKIT_REMOTE`) needs key-based ssh + rsync. Code (`bin/`, `trankit/`) syncs every run;
`runtime/` and `model/` only first time or with `--remote-sync`. Default remote dir `~/trankit_run`; device defaults to mps.
The API runs remotely with `ssh HOST ~/trankit_run/bin/trankit-run-api`. Clients: `/Volumes/Data/pipelines/clients/trankit_client.{py,R}`.

## Training a model
```sh
python3 scripts/prepare_tokenizer_data.py TRAIN.conllu TRAIN.txt      # multi-sentence paragraphs (REQUIRED for sentence splitting)
python3 scripts/prepare_tokenizer_data.py DEV.conllu DEV.txt
runtime/bin/python scripts/train.py --train-conllu TRAIN.conllu --dev-conllu DEV.conllu \
    --train-txt TRAIN.txt --dev-txt DEV.txt --output models/NAME --embedding almanach/camembert-base --device mps
```
Requirements: CoNLL-U files must end with an empty line (`train.py` checks); tokenizer text must contain multi-sentence
paragraphs, not one sentence each. Train on the remote host (`bin/trankit-run-deploy HOST --data DIR --hf-model DIR`),
not on a small machine. MWT expander and lemmatizer do not depend on the encoder and can be copied from an existing model.
See PLAN.md "Training gotchas".
