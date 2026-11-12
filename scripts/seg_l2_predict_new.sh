#!/bin/sh
# Score the new CamemBERT Trankit model on the L2 (CEAAL2) sentence-segmentation sample. CPU only, low priority.
#   scripts/seg_l2_predict_new.sh MODEL_DIR
# MODEL_DIR = Trankit cache dir that holds almanach--camembert-base/customized-mwt/ (copy it from the MacBook first,
# e.g. ~/trankit_run/models/camembert_fr). Needs eval/l2_texts.json, l2_trim.json, l2_gold*.json (git-ignored, local).
set -eu
MODEL_DIR="${1:?usage: seg_l2_predict_new.sh MODEL_DIR}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
[ -d "$MODEL_DIR/almanach--camembert-base/customized-mwt" ] || { echo "missing $MODEL_DIR/almanach--camembert-base/customized-mwt" >&2; exit 1; }
export OMP_NUM_THREADS=2 HF_HUB_CACHE=/Volumes/Data/research_data/hf_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANKIT_HOME="$HERE"
nice -n 10 runtime/bin/python scripts/seg_l2_sample.py predict eval/l2_texts.json \
    "trankit:$MODEL_DIR:customized-mwt:almanach/camembert-base" eval/p_newtrankit.json
for G in eval/l2_gold.json eval/l2_gold_agent.json; do
    echo "== gold: $G"
    python3 scripts/seg_l2_score.py "$G" eval/l2_trim.json eval/p_oldtrankit.json eval/p_syntok.json eval/p_stanza.json eval/p_sat.json eval/p_newtrankit.json | grep -E " (all|native|learner) "
done
