#!/bin/bash
#SBATCH --job-name=gemma4b_qlora_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/lustre/home/%u/qlora/logs/train_qlora_%j.log

cd /scratch/lustre/home/$USER/gemma

# katalogai
mkdir -p logs
mkdir -p /scratch/lustre/home/$USER/hf_cache
mkdir -p /scratch/lustre/home/$USER/hf_home
mkdir -p /scratch/lustre/home/$USER/hf_datasets

export TRANSFORMERS_CACHE=/scratch/lustre/home/$USER/hf_cache
export HF_HOME=/scratch/lustre/home/$USER/hf_home
export HF_DATASETS_CACHE=/scratch/lustre/home/$USER/hf_datasets

# reikia įrašyti savo tokeną
export HF_TOKEN="jūsų_tokenas"

source venv/bin/activate
nvidia-smi
python train-qlora.py

