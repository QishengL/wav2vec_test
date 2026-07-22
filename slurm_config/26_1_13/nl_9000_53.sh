#!/bin/bash
#SBATCH --job-name=53_nl_9000
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/nl_9000-53.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/nl_9000-53.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

nvidia-smi
singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 main.py --config ./config/pretrainv2/2026_1_13/nl_9000-53.py"