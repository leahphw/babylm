#!/bin/bash -e

# This is the name that will show when jobs are listed.
# You can name your jobs however you like; it is a good
# idea to make the name recognizeable in the first few 
# characters.
#SBATCH --job-name=nlp_g1_train

# This is usually the right value -- most of the time
# you are running one program at a time no matter how
# many cores you are using.
#SBATCH --ntasks=1

# This is ten seconds not ten minutes. Adjust accordingly.
# hours:min:sec
#SBATCH --time=00:15:00
# Change /netid/ to *your* netid.
#SBATCH --mail-user=
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
#SBATCH --gres=gpu:tesla_a40:2

# basic is the default collection of compute nodes. They
# each have 52 cores and 384GB of memory.

# Partition for node18
#SBATCH --partition=NLP

# Memory requests are in megabytes by default.
#SBATCH --mem=32000

# This figure means cores not CPUs. 
# Node 18 has 52 cores
#SBATCH --cpus-per-task=8

# Print the start date/time
date

# This step makes sure you have a directory on the /scratch
# mount point. Be sure to change netid to your netid.
mkdir -p /scratch/jstipl

# Print the node your job is running on
echo "I ran on:"
echo "SLURM_NODELIST=$SLURM_NODELIST"

# Return the context/PWD to the directory where *this* file is located.
cd $SLURM_SUBMIT_DIR

# Set any environment variables like PATH, LD_LIBRARY_PATH, etc.

# Load the necessary program dependencies.

# Run jobs. /usr/bin/time -v will print a number of useful
# diagnostics that will help us understand how the cluster
# is being used. Sleep is a program that does nothing, in this
# case for 60 seconds. Your program probably does a bit more.

/usr/bin/time -v sleep 60

# Print the ending date/time
date

