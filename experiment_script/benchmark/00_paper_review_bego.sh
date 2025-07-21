# Hp_optim for hyperparaters optim
task="hp_hpparam_2"
neptune_name="scmusk-hp"
total_trial=300

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
hparam_path="ajrccm_r2_sch3.json"
nohup ./experiment_script/benchmark/00_hp_optim.sh ${dataset_name} ${class_key} ${batch_key} ${task} \
        ${neptune_name} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#### HLCA Parenchyma
dataset_name="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
hparam_path="hlca_r3.json"
nohup ./experiment_script/benchmark/00_hp_optim.sh ${dataset_name} ${class_key} ${batch_key} ${task} \
        ${neptune_name} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &


# Hp_optim for training_scheme comparison
#nohup ./experiment_script/benchmark/00_hp_optim.sh > experiment_script/benchmark/logs/training_scheme.log 2>&1 &
total_trial=10

# Benchmark models
#In the shell configuration file.
export CELLTYPIST_FOLDER='/data/analysis/data_becavin/celltypist_models/'