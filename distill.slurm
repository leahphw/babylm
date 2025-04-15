#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-distill
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

# Create scratch directory
mkdir -p /scratch/pl3nt

# Set environment variables
export HF_HOME=/scratch/pl3nt/hf_cache

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Initialize conda
conda init bash

# =============================================
# Distillation Execution
# =============================================

# Run distillation training
echo "Starting distillation training..."
/usr/bin/time -v torchrun --nproc_per_node=2 dkds-exp.py

# Print job end time
date

