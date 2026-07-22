#!/bin/bash
#SBATCH --job-name=ablation_espeak_cvipa_n5
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ablation_espeak_cvipa_n5.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ablation_espeak_cvipa_n5.err
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test;python3 src/source_selection/run_single_ablation.py --method espeak_cvipa --n_samples 5"
