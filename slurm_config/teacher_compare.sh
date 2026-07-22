#!/bin/bash
#SBATCH --job-name=teacher_compare
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/teacher_compare.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/teacher_compare.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/teacher_comparison.py"
