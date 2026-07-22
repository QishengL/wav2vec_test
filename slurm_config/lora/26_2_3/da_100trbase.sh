#!/bin/bash
#SBATCH --job-name=da_100-trbase
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/da_100-trbase.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/da_100-trbase.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

nvidia-smi
singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 main.py --config ./config/lora/26_2_3_ablation/da_100-trbase.py"