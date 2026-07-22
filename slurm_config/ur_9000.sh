#!/bin/bash
#SBATCH --job-name=ur_9000-300m
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ur_9000-300m.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/ur_9000-300m.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

nvidia-smi
singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 main.py --config ./config/pretrainv2/ur_9000-300m.py"