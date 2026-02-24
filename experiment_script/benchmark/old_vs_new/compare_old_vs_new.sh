working_dir="/workspace/cell/scMusketeers"
scmusk_path=${working_dir}

# Activate Conda Environment for new sc-musketeers
# Environment setup
module load miniconda
#source activate scmusk-dev
source activate scmusk-old

bestparam_path="default_df_t11.csv"
bestparam_path=${working_dir}"/experiment_script/hyperparam/"${bestparam_path}


new_scmusk="sc-musketeers"
old_scmusk="/home/cbecavin/.local/bin/sc-musketeers"
#######################################
###         Run on all datasets
#######################################
task="transfer_old_new"
neptune_name="sc-musketeers"

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
unlabeled_category="Unknown"

log_file=${working_dir}"/experiment_script/results/logs/${dataset_name}_transfer_new.log"
dataset=${working_dir}"/data/Deprez-Lung-unknown-0.2.h5ad"
out_dir=${working_dir}"/experiment_script/results/${dataset_name}"
outname=Deprez-Lung-unknown-0.2-pred


# echo "HP LOOP - ${dataset_name}"
### New ScMusk
log_file=${working_dir}"/experiment_script/results/logs/${dataset_name}_transfer_new.log"
scmusk_exec=$new_scmusk
outname=${outname}-new
nohup sh ${working_dir}"/experiment_script/benchmark/old_vs_new/run_transfer_unknown.sh" ${scmusk_exec} ${neptune_name} ${dataset} ${out_dir} ${outname} ${class_key} ${batch_key} ${unlabeled_category} ${bestparam_path} > ${log_file} 2>&1 &

### Old ScMusk
log_file=${working_dir}"/experiment_script/results/logs/${dataset_name}_transfer_old.log"
scmusk_exec=$old_scmusk
outname=${outname}-old
nohup sh ${working_dir}"/experiment_script/benchmark/old_vs_new/run_transfer_unknown.sh" ${scmusk_exec} ${neptune_name} ${dataset} ${out_dir} ${outname} ${class_key} ${batch_key} ${unlabeled_category} ${bestparam_path} > ${log_file} 2>&1 &

