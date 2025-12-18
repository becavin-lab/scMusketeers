#!/bin/bash
#SBATCH --job-name=0_test_dataset_scmusk_hp
#SBATCH --output=./test_output/logs/sbatch/test_dataset_hp_trial_0.out
#SBATCH --error=./test_output/logs/sbatch/test_dataset_hp_trial_0.err
#SBATCH --account=cell
#SBATCH --partition=cpucourt
#SBATCH --mem=8G
#SBATCH --time=10:00:00

# Environment setup
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load miniconda
source activate scmusk-dev

echo "Running trial 0"
sc-musketeers hp_optim_single --ref_path data.h5ad --other_arg value --trial_params_path ./test_output/hp_trials/trial_0_params.json --trial_result_path ./test_output/hp_trials/trial_0_result.json
