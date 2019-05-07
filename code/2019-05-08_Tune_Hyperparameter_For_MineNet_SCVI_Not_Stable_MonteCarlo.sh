#!/usr/bin/env bash

#!/bin/bash

# The name of the job:
SBATCH --job-name="2019-05-08_Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo"

SBATCH --ntasks=1
SBATCH --cpus-per-task=8

# The maximum running time of the job in days-hours:mins:sec
SBATCH --time=0-6:0:00

# Batch arrays
SBATCH --array=0-31

# Send yourself an email when the job:
# aborts abnormally (fails)
SBATCH --mail-type=FAIL

# begins
SBATCH --mail-type=BEGIN

# ends successfully
SBATCH --mail-type=END

# Use this email address:
SBATCH --mail-user=hui.li3@student.unimelb.edu.au

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)
# The job command(s):
python 2019-05-08_Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py ${SLURM_ARRAY_TASK_ID}
