#!/bin/bash
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

working_dir=$1
scmusk_path=$2
task=$3
neptune_name=$4
dataset_name=$5
class_key=$6
batch_key=$7
gpu_models=$8
bestparam_path=$9

### Dataset settings
out_dir=${working_dir}"/results"
data_path=${working_dir}"/data"
bestparam_path=${scmusk_path}"/experiment_script/hyperparam/"${bestparam_path}

echo "|--- BASH  #####     Running scMusketeers Benchmark for Task1 with dataset=$dataset_name"
json_dataset_h5ad_path=${scmusk_path}"/experiment_script/datasets_h5ad.json"

# Use `jq` and a `while` loop with a pipe
# This creates a subshell, so you need to be careful with variable scope
# but it's a common and portable pattern.
while IFS='=' read -r key value; do
    if [ "$key" = "$dataset_name" ]; then
        h5ad_suffix="$value"
        break  # Exit the loop once the key is found
    fi
done <<< "$(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$json_dataset_h5ad_path")"
# Construct the full path
if [ -z "$h5ad_suffix" ]; then
    echo "Error: Dataset name '${dataset_name}' not found in the JSON file." >&2
    exit 1
fi
h5ad_path=${data_path}/${h5ad_suffix}".h5ad"
echo "|--- BASH  the dataset will be loaded from $h5ad_path"

echo "|--- BASH  Running python script for benchmark task1"
#python ${scmusk_path}/experiment_script/benchmark/task1/01_label_transfer_between_batch.py benchmark ${h5ad_path} --out_dir ${out_dir} --neptune_name ${neptune_name} --dataset_name ${dataset_name} --class_key ${class_key} --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models


echo "|--- BASH  Running python script for benchmark scMusketeers task1"
python ${scmusk_path}/experiment_script/benchmark/task1/01_label_transfer_between_batch_scPermut.py --working_dir ${working_dir} --dataset_name ${dataset_name} --class_key ${class_key} --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key
