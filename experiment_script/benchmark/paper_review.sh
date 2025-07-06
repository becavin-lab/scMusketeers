# Hp_optim for hyperparaters optim
nohup ./experiment_script/benchmark/00_hp_optim.sh > experiment_script/benchmark/logs/hyperparameters_optim.log 2>&1 &

# Hp_optim for training_scheme comparison
nohup ./experiment_script/benchmark/00_hp_optim.sh > experiment_script/benchmark/logs/training_scheme.log 2>&1 &