#!/bin/bash
for num in {0..99} ; do
echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=2 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((0*num)) --time=7-0:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python tune_hyperparameter_for_SCVI_MI_MonteCarlo.py $num muris_tabula batch NN \"
sleep 1

sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=2 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((0*num)) --time=7-0:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python tune_hyperparameter_for_SCVI_MI_MonteCarlo.py $num muris_tabula batch NN"
sleep 1
done
echo "All jobs submitted!\n"