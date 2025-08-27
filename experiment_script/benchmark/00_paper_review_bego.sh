### Sc-Musketeers directory parameters
#working_dir="/workspace/cell/scMusketeers"
#working_dir="/data/analysis/data_becavin/scMusketeers"
working_dir="/data/analysis/data_becavin/scMusketeers-data"
scmusk_path="/data/analysis/data_becavin/scMusketeers"
#scmusk_path=$working_dir


#######################################
###         Hp_optim for hyperparaters optim
#######################################
task="hp_param"
neptune_name="scmusk-hp"
total_trial=50

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
hparam_path="ajrccm_r2_sch3.json"
bestparam_path="none.csv"
# nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

###### Lake
dataset_name="lake_2021"
class_key="Original_annotation"
batch_key="batch"
hparam_path="ajrccm_r2_sch3.json"
bestparam_path="none.csv"
nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#### HLCA Parenchyma
dataset_name="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
hparam_path="hlca_r3.json"
bestparam_path="none.csv"
# nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} \
#        ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} 
#           ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#######################################
###         Hp_optim for hyperparaters optim
#######################################
# For training_scheme comparison
#task="hp_tscheme"
#neptune_name="scmusk-scheme"
#hparam_path=${scmusk_path}"/experiment_script/hp_ranges/besthp_tscheme.json"

task="hp_tscheme"
neptune_name="scmusk-scheme"
total_trial=10

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
hparam_path="besthp_tscheme.json"
bestparam_path="hp_best_ajrccm.csv"
# nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_tscheme.log 2>&1 &

#### HLCA Parenchyma
dataset_name="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
hparam_path="besthp_tscheme.json"
bestparam_path="hp_best_hlca_par.csv"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_tscheme.log 2>&1 &

#######################################
###         Benchmark models
#######################################
#In the shell configuration file.
export CELLTYPIST_FOLDER='/data/analysis/data_becavin/celltypist_models/'

task="bench_task1"
neptune_name="scmusk-tasks"

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
hparam_path="ajrccm_r2_sch3.json"

gpu_models=True
#sh ./experiment_script/benchmark/task1/benchmark_task_1.sh ${working_dir} ${scmusk_path} ${task} \
#        ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${gpu_models}

gpu_models=False
#sh ./experiment_script/benchmark/task1/benchmark_task_1.sh ${working_dir} ${scmusk_path} ${task} \
#        ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${gpu_models}
