#!/bin/bash
#SBATCH --job-name=MLTI_FT
#SBATCH --output=output_%j.txt     # Save standard output
#SBATCH --error=error_%j.txt       # Save errors
#SBATCH --partition=a100          # Change this based on GPU availability

#SBATCH --gres=gpu:a100:2         # Request 1 A100 GPU
#SBATCH --nodes=1                  # Use one node
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --time=20:00:00            # Set a 1-hour time limit



export CUDA_LAUNCH_BLOCKING=1
source /etc/profile
source ~/.bashrc
conda activate /home/woody/iwi5/iwi5276h/software/private/conda/envs/my_env

deepspeed --num_gpus 2 multigpu_ft.py
