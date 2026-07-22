#!/bin/bash
# Batch submit all 26_6_16 experiments (run from login node)
# Login node uses /nfs/, but slurm scripts keep /mnt/storage/ paths

DIR53=/nfs/qisheng/github/wav2vec_test/slurm_config/lora/26_6_16
DIRBASE=/nfs/qisheng/github/wav2vec_test/slurm_config/lora/26_6_16_base

count53=$(find $DIR53 -name "*.sh" 2>/dev/null | wc -l)
countbase=$(find $DIRBASE -name "*.sh" 2>/dev/null | wc -l)
echo "Found $count53 XLSR-53 + $countbase Base = $((count53+countbase)) jobs"

echo ""
echo "=== Submitting 26_6_16 XLSR-53 ==="
for f in $(find $DIR53 -name "*.sh" | sort); do
    echo "  $(basename $f)"
    sbatch "$f"
    sleep 0.2
done

echo ""
echo "=== Submitting 26_6_16_base ==="
for f in $(find $DIRBASE -name "*.sh" | sort); do
    echo "  $(basename $f)"
    sbatch "$f"
    sleep 0.2
done

echo ""
echo "Done!"
