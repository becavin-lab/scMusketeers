#!/bin/bash
#
#SBATCH --account=cell     # The account name for the job.

working_dir=$1
scmusk_path=$2
task=$3
neptune_name=$4
dataset_name=$5
class_key=$6
batch_key=$7
bestparam_path=$8
hparam_path=$9
total_trial=${10}


### Tests
#dataset_name="ajrccm_by_batch"
#class_key="celltype"
#batch_key="manip"

#dataset_name="hlca_par_dataset_harmonized"
#class_key="ann_finest_level"
#batch_key="dataset"


### Dataset settings
out_dir=${working_dir}"/experiment_script/results"
python_path=${scmusk_path}"/scmusketeers/__main__.py"
data_path=${working_dir}"/data"
hparam_path=${scmusk_path}"/experiment_script/hp_ranges/"${hparam_path}
bestparam_path=${scmusk_path}"/experiment_script/hyperparam/"${bestparam_path}


# Read dataset json to get h5ad path
# json_dataset_h5ad_path=${scmusk_path}"/experiment_script/datasets_h5ad.json"
# declare -A MY_SH_DICT
# while IFS='=' read -r key value; do
#     MY_SH_DICT["$key"]="$value"
# done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' "$json_dataset_h5ad_path")
# h5ad_path=${data_path}/${MY_SH_DICT[${dataset_name}]}".h5ad"
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

# Optional: Print the final path for verification
echo "h5ad_path: ${h5ad_path}"


echo "|--- BASH  #####     Hyperparameters optimization of Sc-Musketeers with dataset=$dataset_name"
echo "|--- BASH  the dataset will be loaded from $h5ad_path"

json_test=$(cat $scmusk_path/experiment_script/benchmark/hp_test_obs.json)
test_obs=$(echo "$json_test" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
test_obs=$(echo "$test_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo "|--- BASH  test_obs=$test_obs"

json_train=$(cat $scmusk_path/experiment_script/benchmark/hp_train_obs.json)
keep_obs=$(echo "$json_train" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
keep_obs=$(echo "$keep_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo "|--- BASH  train_obs=$keep_obs"

### Run scMusketeers hyperparameters optimization
#warmup_epoch=20   # default 100, help - Number of epoch to warmup DANN
#fullmodel_epoch=50   # default = 100, help = Number of epoch to train full model
#permonly_epoch=50   # default = 100, help = Number of epoch to train in permutation only mode
#classifier_epoch=50   # default = 50, help = Number of epoch to train te classifier only
warmup_epoch=1   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=1   # default = 100, help = Number of epoch to train full model
permonly_epoch=1   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=1   # default = 50, help = Number of epoch to train te classifier only

training_scheme="training_scheme_1"

sc-musketeers hp_optim ${h5ad_path} --debug --bestparam_path=${bestparam_path} --training_scheme=${training_scheme} --task ${task} --log_neptune "True" \
--neptune_name ${neptune_name} --out_dir ${out_dir} --total_trial ${total_trial} \
--hparam_path ${hparam_path} --dataset_name ${dataset_name} \
--class_key $class_key --batch_key $batch_key --test_obs $test_obs \
--mode entire_condition --obs_key $batch_key --keep_obs $keep_obs \
--test_split_key TRAIN_TEST_split_batch \
--classifier_epoch=${classifier_epoch} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch}



