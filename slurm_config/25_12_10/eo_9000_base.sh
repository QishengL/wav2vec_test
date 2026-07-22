#!/bin/bash
#SBATCH --job-name=base_eo_9000
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/eo_9000-base.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/eo_9000-base.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

nvidia-smi
singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "df -h;wandb login 67129c4138cabfa6fe40ff02f228f65339bbba0d;cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 main.py --config ./config/pretrainv2/2025_12_10/eo_9000-base.py"