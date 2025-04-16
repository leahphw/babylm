#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-train_baseline
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --mail-user=jan.stipl@richmond.edu
#SBATCH --mail-type=ALL

# Resource Allocation
#SBATCH --gres=gpu:tesla_a40:2
#SBATCH --partition=NLP
#SBATCH --mem=64000
#SBATCH --cpus-per-task=16

# =============================================
# Environment Setup
# =============================================

# Print job start time and node information
date
echo "Running on node: $SLURM_NODELIST"

# Create scratch directory
mkdir -p /scratch/nlp_G1
mkdir -p /scratch/nlp_G1/models/base_line/


# Set environment variables
export HF_HOME=/scratch/nlp_G1/hf_cache
export WANDB_MODE=offline

# Change to submission directory
cd $SLURM_SUBMIT_DIR
cd ..
echo "Working dir: $PWD"

# Initialize conda
conda init bash

# =============================================
# Training Execution
# =============================================


## Train 

# Baseline 1: GPT-2 Small  
echo "Training GPT-2 small..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/base_line/GPT2-small-97M-strict.yaml

# # Baseline 2: DistilledGPT
# # DistilledGPT uses GPT2-44M and Llama-60M as teacher models.
echo "Training DistilledGPT: teacher GPT2-44M..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/base_line/DistilledGPT-Teacher-GPT2-44M-strict.yaml

echo "Training DistilledGPT: teacher Llama-60M..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/base_line/DistilledGPT-Teacher-Llama-60M-strict.yaml

echo "Distilling DistilledGPT..."
/usr/bin/time -v ./scripts/distill-DistilledGPT-44M-strict.sh

# Baseline 3 (optional):** BabyLlama-1 (BabyLM 1) and BabyLlama-2 (BabyLM 2)  
# BabyLlama-1 uses GPT2-705M and Llama-360M as teacher models.
echo "Training BabyLlama-1: teacher GPT2-705M..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/base_line/BabyLlama1-Teacher-GPT2-705M-strict.yaml

echo "Training BabyLlama-1: teacher Llama-360M..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/base_line/BabyLlama1-Teacher-Llama-360M-strict.yaml

echo "Distilling BabyLlama-1..."
/usr/bin/time -v ./scripts/distill-BabyLlama1-58M-strict.sh
# Print job end time
date
