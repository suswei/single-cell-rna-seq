#!/usr/bin/env bash

# The name of the job:
#SBATCH --job-name="MINE_simulation"
#SBATCH --account=punim0890
#SBATCH -p physical

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-6:0:00

# Batch arrays
#SBATCH --array=0-319

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# Use this email address:
#SBATCH --mail-user=hui.li3@student.unimelb.edu.au

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The job command(s):
module load anaconda3/2020.02
source activate sharedenv
python3 MINE_simulation2_sweep.py ${SLURM_ARRAY_TASK_ID}
