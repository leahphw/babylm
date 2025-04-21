#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)
echo "Changing to evaluation pipeline directory..."
cd /home/pl3nt/shared_cfinegan/evaluation-pipeline-2024/
export HF_HOME=/scratch/cfinegan/hf_cache

# This is the fastest config we found
    # --tasks glue,ewok_filtered,blimp_filtered,blimp_supplement  \
    # --device cuda:0,1 
# datasets.exceptions.DatasetNotFoundError: Dataset 'glue' doesn't exist on the Hub or cannot be 
# accessed. If the dataset is private or gated, make sure to log in with `huggingface-cli login`
#  or visit the dataset page at https://huggingface.co/datasets/glue to ask for access.

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal" \
    --tasks ewok_filtered,blimp_filtered,blimp_supplement  \
    --device cuda:0,1 \
    --batch_size 256 \
    --log_samples \
    --output_path /scratch/nlp_G1/results/all/${MODEL_BASENAME}/all_results.json \
    --trust_remote_code \

#         Command being timed: "python -m lm_eval --model hf --model_args pretrained=/scratch/nlp_G1/models/base_line/DistilledGPT-44M-strict,backend=causal --tasks ewok_filtered,blimp_filtered,blimp_supplement --device cuda:0,1 --batch_size 256 --log_samples --output_path /scratch/nlp_G1/results/all/DistilledGPT-44M-strict/all_results.json --trust_remote_code"
#         User time (seconds): 340.63
#         System time (seconds): 6.95
#         Percent of CPU this job got: 33%
#         Elapsed (wall clock) time (h:mm:ss or m:ss): 17:20.83
#         Average shared text size (kbytes): 0
#         Average unshared data size (kbytes): 0
#         Average stack size (kbytes): 0
#         Average total size (kbytes): 0
#         Maximum resident set size (kbytes): 1980148
#         Average resident set size (kbytes): 0
#         Major (requiring I/O) page faults: 0
#         Minor (reclaiming a frame) page faults: 1429580
#         Voluntary context switches: 57226
#         Involuntary context switches: 1039
#         Swaps: 0
#         File system inputs: 0
#         File system outputs: 151560
#         Socket messages sent: 0
#         Socket messages received: 0
#         Signals delivered: 0
#         Page size (bytes): 4096
#         Exit status: 0

# /usr/bin/time -v torchrun --nproc_per_node=2 --master_port=29503 -m lm_eval \
#     --model hf \
#     --model_args pretrained=$MODEL_PATH,backend="causal" \
#     --tasks ewok_filtered,blimp_filtered,blimp_supplement  \
#     --device cuda \
#     --batch_size 256 \
#     --log_samples \
#     --output_path /scratch/nlp_G1/results/all/${MODEL_BASENAME}/all_results.json \
#     --trust_remote_code \

#         Command being timed: "torchrun --nproc_per_node=2 --master_port=29503 -m lm_eval --model hf --model_args pretrained=/scratch/nlp_G1/models/base_line/DistilledGPT-44M-strict,backend=causal --tasks ewok_filtered,blimp_filtered,blimp_supplement --device cuda --batch_size 256 --log_samples --output_path /scratch/nlp_G1/results/all/DistilledGPT-44M-strict/all_results.json --trust_remote_code"
#         User time (seconds): 499.63
#         System time (seconds): 28.71
#         Percent of CPU this job got: 32%
#         Elapsed (wall clock) time (h:mm:ss or m:ss): 26:51.25
#         Average shared text size (kbytes): 0
#         Average unshared data size (kbytes): 0
#         Average stack size (kbytes): 0
#         Average total size (kbytes): 0
#         Maximum resident set size (kbytes): 2203204
#         Average resident set size (kbytes): 0
#         Major (requiring I/O) page faults: 562
#         Minor (reclaiming a frame) page faults: 2785311
#         Voluntary context switches: 185829
#         Involuntary context switches: 1541
#         Swaps: 0
#         File system inputs: 480
#         File system outputs: 151872
#         Socket messages sent: 0
#         Socket messages received: 0
#         Signals delivered: 0
#         Page size (bytes): 4096
#         Exit status: 0
