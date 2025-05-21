outdir="/data/analysis/data_becavin/scmusketeers"
dataset=${outdir}"/data/Deprez-Lung-unknown-0.2.h5ad"
outname=Deprez-Lung-unknown-0.2-pred
classkey="celltype"
unlabeled="Unknown"
batchkey="donor"
#dataset=${outdir}"/data/CellTypist-Lung-unknown-0.2.h5ad"
#outname="CellTypist-Lung-unknown-0.2-pred"
#classkey="cell_type"
#batchkey="donor_id"

ref_dataset=data/Deprez-2020-ref-batch-0.2.h5ad
query_dataset=data/Deprez-2020-query-batch-0.2.h5ad
outname_query="Deprez-2020-query-0.2-pred"

warmup_epoch=2   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=2   # default = 100, help = Number of epoch to train full model
permonly_epoch=5   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=2   # default = 50, help = Number of epoch to train te classifier only

log_neptune=False
neptune_name="scmusk-review"

working_dir="/workspace/cell/Review_scMusk"
python_path="/workspace/cell/scMusketeers/scmusketeers/__main__.py"
hparam_path="/workspace/cell/scMusketeers/experiment_script/hp_ranges/hlca_r2.json"
dataset="/workspace/cell/Review_scMusk/data/celltypist_dataset/yoshida_2021/yoshida_2021.h5ad" 
python ${python_path} optim ${dataset} --working_dir ${working_dir} --hparam_path ${hparam_path} --dataset_name ${dataset_name} --class_key Original_annotation --batch_key batch --mode entire_condition --obs_key batch --keep_obs AN1 AN11 AN12 AN13 AN3 AN6 AN7 --training_scheme training_scheme_1 --test_split_key TRAIN_TEST_split_batch 


--dataset_name yoshida_2021 