#!/bin/bash
#For change library size
for num in {0..3} ; do
echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python break_SCVI_main.py Change_Library_Size $num \"
sleep 1

sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python break_SCVI_main.py Change_Library_Size $num"
sleep 1
done
echo "All jobs submitted!\n"

#For change number of genes with non-zero count
#for num in {0..5} ; do
#echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python break_SCVI_main.py Change_Expressed_Gene_Number $num \"
#sleep 1

#sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python break_SCVI_main.py Change_Expressed_Gene_Number $num"
#sleep 1
#done
#echo "All jobs submitted!\n"


#for num in {0..11} ; do
#echo sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o \"slurm_output/slurm-%A_$num.out\" --wrap=\"module load Python/3.7.1-GCC-6.2.0\; source venv/bin/activate\; module load web_proxy\; python break_SCVI_main.py Change_Gene_Expression_Proportion $num \"
#sleep 1

#sbatch -p mig --job-name lihui10098777.${num} --account punim0614 --ntasks=1 --cpus-per-task=3 --mem=10000 --mail-type=END --mail-user=hui.li3@student.unimelb.edu.au --begin=now+$((60*num)) --time=0-12:0:00 -o "slurm_output/slurm-%A_$num.out" --wrap="module load Python/3.7.1-GCC-6.2.0; source venv/bin/activate; module load web_proxy; python break_SCVI_main.py Change_Gene_Expression_Proportion $num"
#sleep 1
#done
#echo "All jobs submitted!\n"

