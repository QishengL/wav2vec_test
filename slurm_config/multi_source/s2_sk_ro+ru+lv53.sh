#!/bin/bash
#SBATCH --job-name=s2_sk_ro+ru+lv53
#SBATCH --output=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/s2_sk_ro+ru+lv53.out
#SBATCH --error=/mnt/storage/qisheng/github/wav2vec_test/slurm_config/out/s2_sk_ro+ru+lv53.err
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

export WANDB_API_KEY=67129c4138cabfa6fe40ff02f228f65339bbba0d
export HF_HOME=/mnt/storage/ldl_linguistics/hf_home

nvidia-smi

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 main.py --config ./config/lora/multi_source/sk_100-ro+ru+lv53.py"
singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/wav2vec_test/src;python3 ../eval_final.py --config ./config/lora/multi_source/sk_100-ro+ru+lv53.py --results /mnt/storage/qisheng/github/wav2vec_test/results/s2_results.json --name multi_ro+ru+lv_sk"
