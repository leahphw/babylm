#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0,1 \
    --batch_size 1 \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json \
    --trust_remote_code \