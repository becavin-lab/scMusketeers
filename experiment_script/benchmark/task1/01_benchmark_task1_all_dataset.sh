### Sc-Musketeers directory parameters
#working_dir="/workspace/cell/scMusketeers"
#working_dir="/data/analysis/data_becavin/scMusketeers"
working_dir="/data/analysis/data_becavin/scMusketeers-data"
scmusk_path="/data/analysis/data_becavin/scMusketeers"
#scmusk_path=$working_dir

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
./experiment_script/benchmark/task1/01_benchmark_task_1.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${gpu_models} 
#&> ${scmusk_path}/experiment_script/benchmark/logs/task1_${dataset_name}'_'${gpu_models}.log

gpu_models=False
#sh ./experiment_script/benchmark/task1/benchmark_task_1.sh ${working_dir} ${scmusk_path} ${task} \
#        ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${gpu_models}
