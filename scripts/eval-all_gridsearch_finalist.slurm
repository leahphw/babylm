#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-eval_baseline
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=jan.stipl@richmond.edu
#SBATCH --mail-type=ALL

# Resource Allocation
#SBATCH --gres=gpu:tesla_a40:2
#SBATCH --partition=NLP
#SBATCH --mem=64000
#SBATCH --cpus-per-task=31

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
# Eval Execution
# =============================================


## Eval 


/usr/bin/time -v ./scripts/eval-all.sh /scratch/nlp_G1/models/student/PalenkaLlama1-58M-strict-L0.1-H0.1_20250419_165524

/usr/bin/time -v ./scripts/eval-all.sh /scratch/nlp_G1/models/student/PalenkaLlama1-58M-strict-L0.1-H0.3_20250419_192616

/usr/bin/time -v ./scripts/eval-all.sh /scratch/nlp_G1/models/student/PalenkaLlama1-58M-strict-L0.1-H0.5_20250419_215705


# Print job end time
date
