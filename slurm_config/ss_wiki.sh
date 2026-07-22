#!/bin/bash
#SBATCH --job-name=ss_wiki
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ss_wiki.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ss_wiki.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src/source_selection;python3 run_new_targets.py --methods wiki"
