#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"

# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# srun -A cell -p cpucourt -t 10:00:00 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate
# python /workspace/cell/scMusketeers/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name ajrccm_by_batch --class_key celltype --use_hvg 3000 --batch_key manip --mode entire_condition --obs_key manip --gpu_models

sc-musketeers --version


task="task1_allmodels"
sh_file=${working_dir}/01_benchmark_task_1.sh

cpu_sbatch="--partition=cpucourt --time=71:00:00"
gpu_sbatch="--partition=gpu --gres=gpu:1 --time=35:00:00"

#for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
#do
#    echo $dataset
#    #${sh_file} $dataset Original_annotation batch
#done

dataset=ajrccm_by_batch
sh ${sh_file} ajrccm_by_batch celltype manip False&> ${working_dir}/logs/scMusk_${task}_${dataset}.log
#${sh_file}  htap ann_finest_level donor
#${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset
#${sh_file} hlca_trac_dataset_harmonized ann_finest_level dataset


####### sbatch for scPermut

# for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
# do
#     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
#     sbatch ${gpu_sbatch} --output ${log_out} --job-name t1_scm_$dataset ${sh_file} $dataset Original_annotation batch True
#     sbatch ${cpu_sbatch} --output ${log_out} --job-name t1_scm_$dataset ${sh_file} $dataset Original_annotation batch False
# done

# dataset="ajrccm_by_batch"
# class_key=Original_annotation
# batch_key=batch
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch ${gpu_sbatch} --output ${log_out} --job-name t1_scm_$dataset ${sh_file} $dataset $class_key $batch_key True
# sbatch ${cpu_sbatch} --output ${log_out} --job-name t1_scm_$dataset ${sh_file} $dataset Original_annotation batch False

# dataset="htap"
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name t1_scm_htap ${sh_file} ${dataset} ann_finest_level donor

# dataset="hlca_par_dataset_harmonized"
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name t1_scm_hlca_par_dataset_harmonized ${sh_file} ${dataset} ann_finest_level dataset

# dataset="hlca_trac_dataset_harmonized"
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name t1_scm_hlca_trac_dataset_harmonized ${sh_file} ${dataset} ann_finest_level dataset


# for dataset in "litvinukova_2020" #"tran_2021" #"tabula_2022_spleen" "yoshida_2021"  "tosti_2021" "lake_2021" "dominguez_2022_spleen" "koenig_2022" "dominguez_2022_lymph" 
# do
#     # sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_$dataset benchmark_task_1.sh $dataset Original_annotation batch True
#     sbatch ${cpu_sbatch} --job-name t1_$dataset ${sh_file} $dataset Original_annotation batch False
# done

# dataset="ajrccm_by_batch"
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch ${gpu_sbatch} --job-name t1_ajrccm_by_batch ${sh_file} ajrccm_by_batch celltype manip True
# sbatch ${gpu_sbatch} --job-name t1_htap ${sh_file} htap ann_finest_level donor True
# sbatch ${gpu_sbatch} --job-name t1_hlca_par_dataset_harmonized ${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset True
# sbatch ${gpu_sbatch} --job-name t1_hlca_trac_dataset_harmonized ${sh_file} hlca_trac_dataset_harmonized ann_finest_level dataset True

# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
# sbatch ${cpu_sbatch} --job-name t1_ajrccm_by_batch ${sh_file} ajrccm_by_batch celltype manip False
# sbatch ${cpu_sbatch} --job-name t1_htap ${sh_file} htap ann_finest_level donor False
# sbatch ${cpu_sbatch} --job-name t1_hlca_par_dataset_harmonized ${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset False

# dataset="hlca_trac_dataset_harmonized"
# log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_task1_${dataset}.log
# sbatch --partition=cpucourt --time=71:00:00 --job-name t1_hlca_trac_dataset_harmonized benchmark_task_1.sh hlca_trac_dataset_harmonized ann_finest_level dataset False



