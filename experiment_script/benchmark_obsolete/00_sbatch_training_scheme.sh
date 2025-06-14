#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/scPermut/"
singularity_working_dir="/data/scPermut/"
singularity_path=$working_dir"/singularity_scPermut.sif"
dataset_name=$1
class_key=$2
batch_key=$3

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/scpermut/run_hp.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --keep_obs $keep_obs --working_dir $working_dir &> $working_dir/experiment_script/benchmark/logs/hp_optim_$dataset_name.log


neptune_name="scmusk-review"

working_dir="/workspace/cell/Review_scMusk"
python_path="/workspace/cell/scMusketeers/scmusketeers/__main__.py"
hparam_path="/workspace/cell/scMusketeers/experiment_script/hp_ranges/generic_r1_debug.json"
dataset="/workspace/cell/Review_scMusk/data/celltypist_dataset/yoshida_2021/yoshida_2021.h5ad"
dataset_name=Yoshida_2021

warmup_epoch=2   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=2   # default = 100, help = Number of epoch to train full model
permonly_epoch=2   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=2   # default = 50, help = Number of epoch to train te classifier only

training_scheme="training_scheme_1"

python_script="""python ${python_path} hp_optim ${dataset} --log_neptune=${log_neptune} --neptune_name=${neptune_name} --working_dir ${working_dir} \
--warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} \
--classifier_epoch=${classifier_epoch} --hparam_path ${hparam_path} --dataset_name ${dataset_name} \
--class_key Original_annotation --batch_key batch --mode entire_condition --obs_key batch \
--keep_obs AN1 AN11 AN12 AN13 AN3 AN6 AN7 --training_scheme=${training_scheme} --test_split_key TRAIN_TEST_split_batch"""

echo python_script
