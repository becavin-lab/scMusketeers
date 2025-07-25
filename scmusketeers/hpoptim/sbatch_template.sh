#!/bin/bash
#SBATCH --job-name=ax_trial
#SBATCH --output=slurm_logs/trial_%j.out
#SBATCH --error=slurm_logs/trial_%j.err
#SBATCH --partition=your_gpu_partition # Specify your GPU partition
#SBATCH --gres=gpu:1                   # Request 1 GPU per trial
#SBATCH --cpus-per-task=4              # Adjust as needed
#SBATCH --mem=32G                      # Adjust as needed
#SBATCH --time=02:00:00                # Set a reasonable time limit

# Activate your Python environment (conda, venv, etc.)
source /path/to/your/environment/bin/activate

# Create a directory for slurm logs if it doesn't exist
mkdir -p slurm_logs

# Run the training script passed from the orchestrator
# $1: trial_index, $2: params_json, $3: run_file_path, $4: output_dir
python train_trial.py \
    --trial_index "$1" \
    --params_json "$2" \
    --run_file_path "$3" \
    --output_dir "$4"