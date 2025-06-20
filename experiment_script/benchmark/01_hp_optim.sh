#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=logs/ajrccm_hyperparam_2.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

#dataset_name=$1
#class_key=$2
#batch_key=$3
### Tests
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"

### Sc-Musketeers parameters
neptune_name="scmusk-review"
working_dir="/workspace/cell/scMusketeers"
out_dir=${working_dir}"/experiment_script/results"
python_path=${working_dir}"/scmusketeers/__main__.py"
data_path=${working_dir}"/data"
hparam_path=${working_dir}"/experiment_script/hp_ranges/generic_r1_debug.json"


# Read dataset json to get h5ad path
json_dataset_h5ad_path=${working_dir}"/experiment_script/datasets_h5ad.json"
declare -A MY_SH_DICT
while IFS='=' read -r key value; do
    MY_SH_DICT["$key"]="$value"
done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$json_dataset_h5ad_path")
h5ad_path=${data_path}/${MY_SH_DICT[${dataset_name}]}".h5ad"


echo "|--- BASH  #####     Hyperparameters optimization of Sc-Musketeers with dataset=$dataset_name"
echo "|--- BASH  the dataset will be loaded from $h5ad_path"

json_test=$(cat $working_dir/experiment_script/benchmark/hp_test_obs.json)
test_obs=$(echo "$json_test" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
test_obs=$(echo "$test_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo "|--- BASH  test_obs=$test_obs"

json_train=$(cat $working_dir/experiment_script/benchmark/hp_train_obs.json)
keep_obs=$(echo "$json_train" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
keep_obs=$(echo "$keep_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo "|--- BASH  train_obs=$keep_obs"

### Run scMusketeers hyperparameters optimization
#warmup_epoch=100   # default 100, help - Number of epoch to warmup DANN
#fullmodel_epoch=100   # default = 100, help = Number of epoch to train full model
#permonly_epoch=100   # default = 100, help = Number of epoch to train in permutation only mode
#classifier_epoch=50   # default = 50, help = Number of epoch to train te classifier only
warmup_epoch=1   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=1   # default = 100, help = Number of epoch to train full model
permonly_epoch=1   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=1   # default = 50, help = Number of epoch to train te classifier only



python ${python_path} hp_optim ${h5ad_path} --debug --training_scheme="training_scheme_debug_1" --neptune_name ${neptune_name} --out_dir ${out_dir} \
--hparam_path ${hparam_path} --dataset_name ${dataset_name} \
--class_key $class_key --batch_key $batch_key --test_obs $test_obs \
--mode entire_condition --obs_key $batch_key --keep_obs $keep_obs \
--test_split_key TRAIN_TEST_split_batch \
--classifier_epoch=${classifier_epoch} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch}


#module load singularity

#singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/scpermut/run_hp.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --keep_obs $keep_obs --working_dir $working_dir &> $working_dir/experiment_script/benchmark/logs/hp_optim_$dataset_name.log

#!/bin/bash
# Get GPU
# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i


