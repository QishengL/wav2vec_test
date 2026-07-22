#!/bin/bash
#SBATCH --job-name=supcon36_select
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/supcon36_select.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/supcon36_select.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/supcon_36_source_selection.py"
