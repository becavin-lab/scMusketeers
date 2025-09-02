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
#echo "HP LOOP - ${dataset_name}"
# nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#### HTAP
dataset_name="htap"
class_key="ann_finest_level"
batch_key="donor"
hparam_path="generic_r1.json"
bestparam_path="none.csv"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#### HLCA Trac
dataset_name="hlca_trac_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
hparam_path="hlca_r3.json"
bestparam_path="none.csv"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

#### HLCA Parenchyma
dataset_name="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
hparam_path="hlca_r3.json"
bestparam_path="none.csv"
#echo "HP LOOP - ${dataset_name}"
#nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &

###### All CellTypist datasets
# dominguez_2022_lymph" "tosti_2021"; #   #"tran_2021" "tabula_2022_spleen"   # "yoshida_2021"
class_key="Original_annotation"
batch_key="batch"
hparam_path="generic_r1.json"
bestparam_path="none.csv"
for dataset_name in  "dominguez_2022_spleen";
#for dataset_name in  "dominguez_2022_lymph" "dominguez_2022_spleen" "tosti_2021" "tran_2021" "tabula_2022_spleen" "yoshida_2021" "litvinukova_2020" "koenig_2022" "lake_2021" "dominguez_2022_spleen";
do
    echo "HP LOOP - ${dataset_name}"
    nohup ./experiment_script/benchmark/00_hp_optim.sh ${working_dir} ${scmusk_path} ${task} ${neptune_name} ${dataset_name} ${class_key} ${batch_key} ${bestparam_path} ${hparam_path} ${total_trial} > experiment_script/benchmark/logs/${dataset_name}_hyperparameters_optim.log 2>&1 &
done

