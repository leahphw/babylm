#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)
echo "Changing to evaluation pipeline directory..."
cd /home/pl3nt/shared_cfinegan/evaluation-pipeline-2024/
export HF_HOME=/scratch/cfinegan/hf_cache

# This is the fastest config we found
python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal" \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0,1 \
    --batch_size 1 \
    --log_samples \
    --output_path /scratch/nlp_G1/results/blimp/${MODEL_BASENAME}/blimp_results.json \
    --trust_remote_code \


# Config testing
# MODEL_PATH="/scratch/nlp_G1/models/base_line/GPT2-small-97M-strict"
# python -m lm_eval --model hf \
#     --model_args pretrained=$MODEL_PATH,backend="causal" \
#     --tasks blimp_filtered,blimp_supplement \
#     --device cuda:0 \
#     --batch_size 1 \
#     --log_samples \
#     --output_path /scratch/nlp_G1/results/blimp/${MODEL_BASENAME}/blimp_results.json \
#     --trust_remote_code \

# |     Groups     |Version|Filter|n-shot|Metric|Value |   |Stderr|
# |----------------|-------|------|-----:|------|-----:|---|-----:|
# |blimp_supplement|N/A    |none  |     0|acc   |0.6140|±  |0.0057|
# |blimp_filtered  |N/A    |none  |     0|acc   |0.6678|±  |0.0017|


# real    44m32.207s
# user    34m28.265s
# sys     0m10.837s

# torchrun --nproc_per_node=2 -m lm_eval --model hf \
#     --model_args pretrained=$MODEL_PATH,backend="causal" \
#     --tasks blimp_filtered,blimp_supplement \
#     --device cuda:0 \
#     --batch_size 1 \
#     --log_samples \
#     --output_path /scratch/nlp_G1/results/blimp/${MODEL_BASENAME}/blimp_results.json \
#     --trust_remote_code \

# |     Groups     |Version|Filter|n-shot|Metric|Value |   |Stderr|
# |----------------|-------|------|-----:|------|-----:|---|-----:|
# |blimp_supplement|N/A    |none  |     0|acc   |0.6140|±  |0.0057|
# |blimp_filtered  |N/A    |none  |     0|acc   |0.6678|±  |0.0017|

# [rank0]:[W415 17:50:31.797750294 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

# real    29m40.762s
# user    19m8.624s
# sys     0m31.251s

# MODEL_PATH="/scratch/nlp_G1/models/base_line/GPT2-small-97M-strict"
# python -m lm_eval --model hf \
# --device cuda:0,1 \
# |     Groups     |Version|Filter|n-shot|Metric|Value |   |Stderr|
# |----------------|-------|------|-----:|------|-----:|---|-----:|
# |blimp_supplement|N/A    |none  |     0|acc   |0.6140|±  |0.0057|
# |blimp_filtered  |N/A    |none  |     0|acc   |0.6678|±  |0.0017|


# real    24m1.790s
# user    14m2.236s
# sys     0m11.142s


# torchrun --nproc_per_node=2 -m lm_eval --model hf \
#     --model_args pretrained=$MODEL_PATH,backend="causal" \
#     --tasks blimp_filtered,blimp_supplement \
#     --batch_size 1 \
#     --log_samples \
#     --output_path /scratch/nlp_G1/results/blimp/${MODEL_BASENAME}/blimp_results.json \
#     --trust_remote_code \


# |     Groups     |Version|Filter|n-shot|Metric|Value |   |Stderr|
# |----------------|-------|------|-----:|------|-----:|---|-----:|
# |blimp_supplement|N/A    |none  |     0|acc   |0.6140|±  |0.0057|
# |blimp_filtered  |N/A    |none  |     0|acc   |0.6678|±  |0.0017|

# [rank0]:[W415 17:03:30.427703126 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

# real    28m8.393s
# user    16m18.688s
# sys     1m15.533s
