#!/bin/bash
#SBATCH --job-name=ss_cv
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ss_cv.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ss_cv.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src/source_selection;python3 run_new_targets.py --methods cv"
