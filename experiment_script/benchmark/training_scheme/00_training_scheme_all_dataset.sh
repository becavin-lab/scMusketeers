### Sc-Musketeers directory parameters
working_dir="/workspace/cell/scMusketeers"
scmusk_path=$working_dir

#working_dir="/data/analysis/data_becavin/scMusketeers"
#working_dir="/data/analysis/data_becavin/scMusketeers-data"
#scmusk_path="/data/analysis/data_becavin/scMusketeers"


#######################################
###         Hp_optim for training scheme
#######################################
# For training_scheme comparison
#task="hp_tscheme"
#neptune_name="scmusk-scheme"
#hparam_path=${scmusk_path}"/experiment_script/hp_ranges/besthp_tscheme.json"
hparam_path="besthp_tscheme.json"
bestparam_path="default_df_t11.csv"

task="hp_tscheme"
neptune_name="scmusk-scheme"
total_trial=11

#### AJRCCM
dataset_name="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
log_file="experiment_script/results/logs/${dataset_name}_tscheme_optim.log"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > ${log_file} 2>&1 &

#### HTAP
dataset_name="htap"
class_key="ann_finest_level"
batch_key="donor"
log_file="experiment_script/results/logs/${dataset_name}_tscheme_optim.log"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > ${log_file} 2>&1 &

#### HLCA Trac
dataset_name="hlca_trac_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
log_file="experiment_script/results/logs/${dataset_name}_tscheme_optim.log"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > ${log_file} 2>&1 &

#### HLCA Parenchyma
dataset_name="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
log_file="experiment_script/results/logs/${dataset_name}_tscheme_optim.log"
echo "HP LOOP - ${dataset_name}"
nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > ${log_file} 2>&1 &

###### All CellTypist datasets
# dominguez_2022_lymph" "tosti_2021"; #   #"tran_2021" "tabula_2022_spleen"   # "yoshida_2021"
class_key="Original_annotation"
batch_key="batch"
#for dataset_name in  "tran_2021";
#for dataset_name in  "dominguez_2022_lymph" "dominguez_2022_spleen";
for dataset_name in  "tosti_2021" "tran_2021" "tabula_2022_spleen" "yoshida_2021" "dominguez_2022_lymph" "dominguez_2022_spleen";
do
    log_file="experiment_script/results/logs/${dataset_name}_tscheme_optim.log"
    echo "HP LOOP - ${dataset_name}"
    nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > ${log_file} 2>&1 &
done
