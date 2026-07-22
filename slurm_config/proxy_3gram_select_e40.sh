#!/bin/bash
#SBATCH --job-name=p3g_select_e40
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/proxy3gram_select_e40.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/proxy3gram_select_e40.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/proxy_3gram_source_selection.py 40"
