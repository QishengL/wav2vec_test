#!/bin/bash
#SBATCH --job-name=eval_g1
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/eval_g1.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/eval_g1.err
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test;python3 batch_eval.py --group 1"
