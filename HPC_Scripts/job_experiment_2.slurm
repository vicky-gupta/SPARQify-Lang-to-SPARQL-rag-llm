#!/bin/bash
#SBATCH --job-name=LORA_IF
#SBATCH --output=output_%j.txt     # Save standard output
#SBATCH --error=error_%j.txt       # Save errors
#SBATCH --partition=a100          # Change this based on GPU availability

#SBATCH --gres=gpu:a100:1          # Request 1 A100 GPU
#SBATCH --nodes=1                  # Use one node
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --cpus-per-task=8          # Allocate 4 CPU cores
#SBATCH --time=02:00:00            # Set a 1-hour time limit




source /etc/profile
#module load python/3.12-conda      # Load Python module (adjust version if needed)
source ~/.bashrc
conda activate /home/woody/iwi5/iwi5276h/software/private/conda/envs/my_env
python -c "import huggingface_hub, transformers; print('Packages loaded')"
python experiment_2.py             # Run your script
