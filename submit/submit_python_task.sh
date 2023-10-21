#!/bin/sh
#SBATCH -N 1
#SBATCH --job-name=data
#SBATCH -o out/data_out
#SBATCH -e err/data_err
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem 64G
#SBATCH -p short

###############################
# Run experiments in main.py (by submitting a slurm job)
# How to use this script? 
# in Cluster Head Node terminal, type: sbatch --array=1-5 main.sbatch
# here the ids are the task ids to run
###############################

echo $(hostname)
echo "Starting to run SBATCH SCRIPT"
source /home/yguan2/anaconda3/bin/activate myenv

code_path="../frame_pipeline.py"
echo "python3 $code_path"
python3 $code_path