#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-eval
#SBATCH --ntasks=1
#SBATCH --time=01:00:30
#SBATCH --mail-user=kasym.manatayev@richmond.edu
#SBATCH --mail-type=ALL

# Resource Allocation
#SBATCH --gres=gpu:tesla_a40:2
#SBATCH --partition=NLP
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4

# =============================================
# Environment Setup
# =============================================

# Print job start time and node information
date
echo "Running on node: $SLURM_NODELIST"

# Create scratch directory
mkdir -p /scratch/pl3nt

# Set environment variables
export HF_HOME=/scratch/cfinegan/hf_cache

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Change to evaluation pipeline directory
echo "Changing to evaluation pipeline directory..."
cd /home/km3nc/shared_cfinegan/evaluation-pipeline-2024/

# Initialize conda
conda init bash

# =============================================
# Evaluation Execution
# =============================================

# Run BLiMP evaluation
echo "Running BLiMP evaluation..."
/usr/bin/time -v /home/km3nc/babylm/eval-blimp.sh /scratch/nlp_G1/models/student/Baby-Llama-58M-paperbase_20250416_020559

# Print job end time
date