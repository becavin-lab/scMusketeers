### Sc-Musketeers directory parameters
working_dir="/workspace/cell/scMusketeers"
scmusk_path=$working_dir
#working_dir="/data/analysis/data_becavin/scMusketeers"
#working_dir="/data/analysis/data_becavin/scMusketeers-data"
#scmusk_path="/data/analysis/data_becavin/scMusketeers"


#######################################
###         Hp_optim for hyperparaters optim
#######################################
sh 00_hp_optim_all_dataset.sh
# Best hyperparameters saved in : experiment_script/results/all_datasets_best_hyperparameters.csv

#######################################
###         Hp_optim for training scheme
#######################################
sh training_scheme/00_training_scheme_all_dataset.sh

#######################################
###         Benchmark models
#######################################
# sh task1/01_benchmark_task1_all_dataset.sh