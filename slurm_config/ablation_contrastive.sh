#!/bin/bash
#SBATCH --job-name=ablation_contrastive
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ablation_contrastive.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ablation_contrastive.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test;python3 src/source_selection/run_single_ablation.py --method contrastive --n_samples 2;python3 src/source_selection/run_single_ablation.py --method contrastive --n_samples 5;python3 src/source_selection/run_single_ablation.py --method contrastive --n_samples 10;python3 src/source_selection/run_single_ablation.py --method contrastive --n_samples 20;python3 src/source_selection/run_single_ablation.py --method contrastive --n_samples 50"
