#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-eval_baseline
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
/usr/bin/time -v ./scripts/eval-blimp.sh /scratch/nlp_G1/models/base_line/GPT2-small-97M-strict


# # Baseline 2: DistilledGPT
/usr/bin/time -v ./scripts/eval-blimp.sh /scratch/nlp_G1/models/base_line/DistilledGPT-44M-strict

# Baseline 3 (optional):** BabyLlama-1 (BabyLM 1) and BabyLlama-2 (BabyLM 2)  
# BabyLlama-1 
/usr/bin/time -v ./scripts/eval-blimp.sh /scratch/nlp_G1/models/base_line/BabyLlama1-58M-strict


# Print job end time
date
