#!/bin/bash -e

# =============================================
# SLURM Configuration
# =============================================

# Job Configuration
#SBATCH --job-name=g1-final_train
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
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
mkdir -p /scratch/nlp_G1/models/student/


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
set +e  # Don't exit on error

torchrun --nproc_per_node=2 --master_port=29501 dkds-noaux.py \
  --config ./config/distillation/dkds_noaux/PalenkaLlama1-58M-strict-L0.1-H0.1.yaml \
  2>&1 | tee /scratch/nlp_G1/slurm_out/run_H0.1.log

torchrun --nproc_per_node=2 --master_port=29501 dkds-noaux.py \
  --config ./config/distillation/dkds_noaux/PalenkaLlama1-58M-strict-L0.1-H0.3.yaml \
  2>&1 | tee /scratch/nlp_G1/slurm_out/run_H0.3.log

torchrun --nproc_per_node=2 --master_port=29501 dkds-noaux.py \
  --config ./config/distillation/dkds_noaux/PalenkaLlama1-58M-strict-L0.1-H0.5.yaml \
  2>&1 | tee /scratch/nlp_G1/slurm_out/run_H0.5.log

date | tee -a /scratch/nlp_G1/slurm_outrun_H0.1.log \
    /scratch/nlp_G1/slurm_out/run_H0.3.log\
    /scratch/nlp_G1/slurm_out/run_H0.5.log

date
