#!/bin/bash
# Get GPU
# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i

log_neptune=False
neptune_name="scmusk-review"

working_dir="/workspace/cell/Review_scMusk"
out_dir=${working_dir}"/experiment_script/results"
python_path="/workspace/cell/scMusketeers/scmusketeers/__main__.py"
hparam_path="/workspace/cell/scMusketeers/experiment_script/hp_ranges/generic_r1_debug.json"
dataset="/workspace/cell/Review_scMusk/data/celltypist_dataset/yoshida_2021/yoshida_2021.h5ad"
dataset_name=Yoshida_2021

warmup_epoch=2   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=2   # default = 100, help = Number of epoch to train full model
permonly_epoch=2   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=2   # default = 50, help = Number of epoch to train te classifier only

training_scheme="training_scheme_1"

python ${python_path} hp_optim ${dataset} --log_neptune=${log_neptune} --neptune_name=${neptune_name} --out_dir=${out_dir} \
--classifier_epoch=${classifier_epoch} --hparam_path ${hparam_path} --dataset_name ${dataset_name} \
--class_key Original_annotation --batch_key batch --mode entire_condition --obs_key batch \
--keep_obs AN1 AN11 AN12 AN13 AN3 AN6 AN7 --training_scheme=${training_scheme} --test_split_key TRAIN_TEST_split_batch \
--warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch}


