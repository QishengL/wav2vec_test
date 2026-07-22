#!/bin/bash
#SBATCH --job-name=pseudo_units_ablation
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/pseudo_units_ablation.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/pseudo_units_ablation.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test;python3 src/source_selection/extract_frame_features.py"

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test;python3 src/source_selection/recluster.py --k_vals 50,100"
