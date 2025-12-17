#!/bin/bash
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.log
#SBATCH --error=gpu_test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G

echo "Loading miniconda module..."
module load miniconda

echo "Activating conda environment scmusk-dev..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scmusk-dev

echo "Environment activated."
echo "Running nvidia-smi to check GPU visibility..."
nvidia-smi

echo "Running Python GPU check..."
python -c "import torch; print(f'Torch available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
echo "Done."
