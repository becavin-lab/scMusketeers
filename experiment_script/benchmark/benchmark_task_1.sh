#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

# Example: experiment_script/benchmark/benchmark_task_1.sh ajrccm_by_batch celltype manip True
working_dir="/data/analysis/data_becavin/scMusketeers"
output_dir="/data/analysis/data_becavin/scMusketeers-data"

dataset_name=$1
class_key=$2
batch_key=$3
gpu_models=$4

python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task1_$dataset_name'_'$gpu_models.log

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task1_$dataset_name'_'$gpu_models.log