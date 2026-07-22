#!/bin/bash
#SBATCH --job-name=build_3gram_js
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/build_3gram_js.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/build_3gram_js.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test && python3 src/source_selection/build_3gram_js.py"
