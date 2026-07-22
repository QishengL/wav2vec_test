#!/bin/bash
#SBATCH --job-name=hubert_diag
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/hubert_diag.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/hubert_diag.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/extract_hubert_features.py"

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/hubert_diagnostic.py"
