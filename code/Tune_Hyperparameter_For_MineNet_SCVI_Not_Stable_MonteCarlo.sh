#!/bin/bash
for num in {0..116} ; do
echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --time=3-0:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; python Compare_MINE_with_True_mutual_information.py $num \"
sleep 1

sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --time=3-0:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; python Compare_MINE_with_True_mutual_information.py $num"
sleep 1
done
echo "All jobs submitted!\n"
