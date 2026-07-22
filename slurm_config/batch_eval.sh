#!/bin/bash
# Submit all 8 eval groups simultaneously
echo "=== Submitting 8 parallel eval jobs ==="
for i in $(seq 1 8); do
    sbatch /nfs/qisheng/github/wav2vec_test/slurm_config/eval_g${i}.sh
    sleep 0.2
done
echo "Done! 8 eval jobs submitted."
