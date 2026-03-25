#srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate
sc-musketeers --version

# Running task1 for scmusketeers and all datasets
sh experiment_script/benchmark/task1/task1_scMusk_all_dataset.sh 
#01_scPermut_all_dataset.sh and 01_all_benchmark.sh are equivalent

# Running task1 for all other models and all datasets
sh experiment_script/benchmark/task1/task1_all_benchmark.sh

# Running task2 for scmusketeers and all datasets
sh experiment_script/benchmark/task2/task2_scMusk_all_dataset.sh 
#01_scPermut_all_dataset.sh and 01_all_benchmark.sh are equivalent in their function and structure

# Running task2 for all other models and all datasets
sh experiment_script/benchmark/task2/task2_all_benchmark.sh

