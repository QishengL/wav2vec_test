#!/bin/bash
#SBATCH --job-name=proxy_continue
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/proxy_continue.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/proxy_continue.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=67129c4138cabfa6fe40ff02f228f65339bbba0d
export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_contrastive/customized;python3 train_with_proxy.py --config ./config/proxy_continue.py --use_proxy --lambda_proxy 0.1 --proxy_temperature 0.1 --resume_path ../weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt"
