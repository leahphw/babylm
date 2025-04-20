#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)
echo "Changing to evaluation pipeline directory..."
cd /home/pl3nt/shared_cfinegan/evaluation-pipeline-2024/
export HF_HOME=/scratch/cfinegan/hf_cache

# This is the fastest config we found
python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal" \
    --tasks ewok_filtered \
    --device cuda:0,1 \
    --batch_size 128 \
    --log_samples \
    --output_path /scratch/nlp_G1/results/ewok/${MODEL_BASENAME}/ewok_results.json \
    --trust_remote_code \
