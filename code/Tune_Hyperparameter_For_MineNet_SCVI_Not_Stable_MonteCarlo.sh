#!/bin/bash

#SBATCH --job-name="2019-05-08_Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --time=1-8:0:00
#SBATCH --array=0-99
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=hui.li3@student.unimelb.edu.au::

# Load required modules
module load Python/3.7.1-GCC-6.2.0
module load web_proxy

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)
# The job command(s):
python 2019-05-08_Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py ${SLURM_ARRAY_TASK_ID}
