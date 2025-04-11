#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=nlp_g1_train
#SBATCH --ntasks=1
#SBATCH --time=01:00:30
#SBATCH --mail-user=leah.le@richmond.edu
#SBATCH --mail-type=TIME_LIMIT_90

# Resource Allocation
#SBATCH --gres=gpu:tesla_a40:2
#SBATCH --partition=NLP
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8

# =============================================
# Environment Setup
# =============================================

# Print job start time and node information
date
echo "Running on node: $SLURM_NODELIST"

# Create scratch directory
mkdir -p /scratch/pl3nt

# Set environment variables
export HF_HOME=/scratch/pl3nt/hf_cache
export WANDB_MODE=offline

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Initialize conda
conda init bash

# =============================================
# Training Execution
# =============================================

# Train Llama model
echo "Training Llama model..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/llama-16M.yaml

# Train GPT model
echo "Training GPT model..."
/usr/bin/time -v torchrun --nproc_per_node=2 train.py --config ./config/gpt-97M.yaml

# Print job end time
date

