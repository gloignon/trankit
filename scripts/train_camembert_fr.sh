#!/bin/sh
# Full French training on CamemBERT-base: tokenizer + posdep trained, MWT/lemma reused from the XLM-R model.
# Run on the training host from ~/trankit_run:  nohup caffeinate -i nice -n 5 scripts/train_camembert_fr.sh > logs/full.log 2>&1 &
set -eu
cd "$(dirname "$0")/.."
ENC=almanach/camembert-base; KEY=almanach--camembert-base; CAT=customized-mwt; OUT=models/camembert_fr
TOK_EPOCHS="${TOK_EPOCHS:-20}"; POS_EPOCHS="${POS_EPOCHS:-25}"
OLD="${OLD_MODEL:-model/xlm-roberta-base/$CAT}"      # existing XLM-R custom model (for MWT + lemma)
export HF_HUB_CACHE="$PWD/hf_cache/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=runtime/bin/python
COMMON="--train-conllu training/train.conllu --dev-conllu training/dev.conllu --train-txt training/train.txt --dev-txt training/dev.txt --output $OUT --embedding $ENC --category $CAT --device mps"
mkdir -p logs "$OUT"
echo "[$(date)] start tokenizer ($TOK_EPOCHS epochs)"
if [ ! -f "$OUT/.tokenize.done" ]; then
    $PY scripts/train.py $COMMON --tasks tokenize --max-epoch "$TOK_EPOCHS"
    touch "$OUT/.tokenize.done"
fi
echo "[$(date)] start posdep ($POS_EPOCHS epochs)"
if [ ! -f "$OUT/.posdep.done" ]; then
    $PY scripts/train.py $COMMON --tasks posdep --max-epoch "$POS_EPOCHS"
    touch "$OUT/.posdep.done"
fi
echo "[$(date)] reuse MWT + lemma from $OLD"
D="$OUT/$KEY/$CAT"
cp "$OLD/$CAT""_mwt_expander.pt" "$OLD/$CAT""_lemmatizer.pt" "$D/"
touch "$D/$CAT.downloaded"
$PY -I -c "import sys; sys.path.insert(0,'.'); import trankit; trankit.verify_customized_pipeline(category='$CAT', save_dir='$OUT', embedding_name='$KEY')"
echo "[$(date)] DONE: $D"
