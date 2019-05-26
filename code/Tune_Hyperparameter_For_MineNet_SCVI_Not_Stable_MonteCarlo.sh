#!/bin/bash
for num in {1..100} ; do
echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --time=2-0:0:00 -e \"slurm_output/slurm-%A_%a.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py $num \"
sleep 1

sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au  --time=2-0:0:00 -e "slurm_output/slurm-%A_%a.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python Tune_Hyperparameter_For_MineNet_SCVI_Not_Stable_MonteCarlo.py $num"
sleep 1
done
echo "All jobs submitted!\n"
