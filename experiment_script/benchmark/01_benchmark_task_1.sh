#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.


working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"
dataset_name=$1
class_key=$2
batch_key=$3
gpu_models=$4

python ${working_dir}01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models

# singularity version
# module load singularity
# singularity_working_dir="/data/scPermut"
# singularity_path=$working_dir"/scanvi_sin_copy.sif"
# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task1_$dataset_name'_'$gpu_models.log