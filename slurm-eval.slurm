#!/bin/bash -e

# This is the name that will show when jobs are listed.
# You can name your jobs however you like; it is a good
# idea to make the name recognizeable in the first few 
# characters.
#SBATCH --job-name=CFD_EVAL

# This is usually the right value -- most of the time
# you are running one program at a time no matter how
# many cores you are using.
#SBATCH --ntasks=1

# This is one hour 30 seconds. Adjust accordingly.
#SBATCH --time=01:00:30
# Change /netid/ to *your* netid.
#SBATCH --mail-user=jan.stipl@richmond.edu
#SBATCH --mail-type=ALL
#####
#BEGIN — send mail when the job begins.
#END — send mail when the job ends.
#FAIL — send mail if the job fails.
#REQUEUE — send mail if the job is requeued.
#ALL — send mail for all events (i.e., BEGIN, END, FAIL, REQUEUE).
#STAGE_OUT — send mail when stage-out is completed (for advanced I/O setups).
#TIME_LIMIT — send mail when the job approaches its time limit.
#TIME_LIMIT_90 — job has used 90% of its time limit.
#TIME_LIMIT_80 — job has used 80% of its time limit.
#TIME_LIMIT_50 — job has used 50% of its time limit.
#ARRAY_TASKS — send separate mail for each array task (instead of aggregate).
#####


###
# This statement requests the use of a GPU. The type of GPU
# is not required if there is only one type on the node.
# The final ":1" says, "I want one GPU."
###
#SBATCH --gres=gpu:tesla_a40:1

# basic is the default collection of compute nodes. They
# each have 52 cores and 384GB of memory.
#SBATCH --partition=NLP

# Memory requests are in megabytes by default. This is 24 GB.
#SBATCH --mem=32000

# This figure means cores not CPUs. 
#SBATCH --cpus-per-task=4

# Print the start date/time
date

# This step makes sure you have a directory on the /scratch
# mount point. Be sure to change netid to your netid.
mkdir -p /scratch/km3nc

# Print the node your job is running on
echo "I ran on:"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Return the context/PWD to the directory where *this* file is located.
cd $SLURM_SUBMIT_DIR

# Set any environment variables like PATH, LD_LIBRARY_PATH, etc.
export HF_HOME=/scratch/cfinegan/hf_cache

echo "cd to target directory"
cd /home/km3nc/shared_cfinegan/evaluation-pipeline-2024/

# Load the necessary program dependencies.
echo "setting up conda"
conda init bash

echo "checking pip installs"
pip freeze | grep minicons



# Run jobs. /usr/bin/time -v will print a number of useful
# diagnostics that will help us understand how the cluster
# is being used. Sleep is a program that does nothing, in this
# case for 60 seconds. Your program probably does a bit more.

# echo "Running python script"
# python /home/pl3nt/shared_cfinegan/evaluation-pipeline-2024/sample_pipeline.py

echo "Run eval script"
/usr/bin/time -v /home/km3nc/babylm/neweval_blimp.sh /home/km3nc/babylm/models/Baby-Llama-58M


# echo "Run eval script"
# /usr/bin/time -v ./eval_blimp.sh distilbert/distilgpt2

# Print the ending date/time
date