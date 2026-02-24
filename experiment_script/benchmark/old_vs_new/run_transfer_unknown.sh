# Get GPU
# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
scmusk_exec=$1
neptune_name=$2
dataset=$3
out_dir=$4
outname=$5
class_key=$6
batch_key=$7
unlabeled_category=$8
bestparam_path=$9

### Run scMusketeers hyperparameters optimization
warmup_epoch=20   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=50   # default = 100, help = Number of epoch to train full model
permonly_epoch=50   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=50   # default = 50, help = Number of epoch to train te classifier only
#warmup_epoch=1   # default 100, help - Number of epoch to warmup DANN
#fullmodel_epoch=1   # default = 100, help = Number of epoch to train full model
#permonly_epoch=1   # default = 100, help = Number of epoch to train in permutation only mode
#classifier_epoch=1   # default = 50, help = Number of epoch to train te classifier only

task="old_vs_new"

training_scheme="training_scheme_1"

log_neptune=True

##### Sampling_percentage 20%
${scmusk_exec} transfer ${dataset} --log_neptune=${log_neptune} \
    --neptune_name=${neptune_name} --training_scheme=${training_scheme} --task ${task}\
    --class_key=${class_key} --unlabeled_category=${unlabeled_category} --batch_key=${batch_key} --out_dir=${out_dir} --out_name=${outname} \
    --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
    # > nohup_transfer.out &
