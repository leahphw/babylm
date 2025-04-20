#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=1gpu
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=kasym.manatayev@richmond.edu
#SBATCH --mail-type=ALL

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


# Set environment variables
export HF_HOME=/scratch/pl3nt/hf_cache

# Change to submission directory
cd $SLURM_SUBMIT_DIR
cd ..

# Initialize conda
conda init bash

# =============================================
# Distillation Execution
# =============================================

# Run distillation training
echo "Starting distillation training..."
export CUDA_VISIBLE_DEVICES=1
python grid_search.py \
    --config "/home/km3nc/babylm/config/dsllama-58m.yaml" \
    --output_dir "/home/km3nc/babylm/results" \
    --num_epochs 6 \
    --train_fraction 0.2

# Print job end time
date
