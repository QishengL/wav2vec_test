#!/bin/bash
# eval_final.sh — Evaluate a trained model on the FULL test set
# Usage: ./eval_final.sh <config_name>
# Example: ./eval_final.sh mt_100-ar53
#          ./eval_final.sh af_100-en53

NAME=$1
if [ -z "$NAME" ]; then
    echo "Usage: ./eval_final.sh <config_name>"
    echo "Example: ./eval_final.sh mt_100-ar53"
    exit 1
fi

CONFIG="/mnt/storage/qisheng/github/wav2vec_test/src/config/lora/26_6_5/${NAME}.py"
if [ ! -f "$CONFIG" ]; then
    CONFIG="/mnt/storage/qisheng/github/wav2vec_test/src/config/lora/26_6_5_base/${NAME}.py"
fi
if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $NAME"
    exit 1
fi

# Extract checkpoint directory from config
CKPT_DIR=$(grep "output_dir" "$CONFIG" | head -1 | sed 's/.*"output_dir": "//' | sed 's/",.*//')
CKPT_PATH="/mnt/storage/qisheng/github/wav2vec_test/${CKPT_DIR#../}"

# Find latest checkpoint
LATEST=$(ls -d "$CKPT_PATH"/checkpoint-* 2>/dev/null | sort -t'-' -k2 -n | tail -1)
if [ -z "$LATEST" ]; then
    echo "No checkpoint found in $CKPT_PATH"
    exit 1
fi

echo "Config: $NAME"
echo "Checkpoint: $(basename $LATEST)"

# Create temp config with full test set
TMP="/tmp/eval_${NAME}.py"
cp "$CONFIG" "$TMP"
sed -i 's/"max_eval_samples_per_language":\[[0-9]*\]/"max_eval_samples_per_language":[100000]/' "$TMP"
sed -i 's/"do_train": True/"do_train": False/' "$TMP"
sed -i 's/resume = False/resume = True/' "$TMP"
sed -i "/^resume_dir/d" "$TMP"
# Remove last newline and add resume_dir
echo "" >> "$TMP"
echo "resume_dir = '${LATEST}'" >> "$TMP"

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

cd /mnt/storage/qisheng/github/wav2vec_test/src
python3 main.py --config "$TMP" 2>&1 | grep -E "wer|eval_samples|loss|Evaluate"

rm -f "$TMP"
