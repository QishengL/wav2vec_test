#!/bin/bash
#SBATCH --job-name=hist_compare
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/hist_compare.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/hist_compare.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/compare_histogram_methods.py"
