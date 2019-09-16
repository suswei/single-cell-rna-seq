#!/bin/bash
for num in {0..39} ; do
echo sbatch -p physical --job-name lihui10098777.${num} --account punim0890 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((0*num)) --time=4-0:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python tune_hyperparameter_for_SCVI_MI.py 0 muris_tabula batch MINE_Net4_3 $num \"
sleep 1

sbatch -p physical --job-name lihui10098777.${num} --account punim0890 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((0*num)) --time=4-0:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python tune_hyperparameter_for_SCVI_MI.py 0 muris_tabula batch $num"
sleep 1
done
echo "All jobs submitted!\n"
