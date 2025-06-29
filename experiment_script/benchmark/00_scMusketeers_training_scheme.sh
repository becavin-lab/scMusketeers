#!/bin/sh
#

# for training_scheme in "training_scheme_17" "training_scheme_18" "training_scheme_12" "training_scheme_13" "training_scheme_14" "training_scheme_15" "training_scheme_8" "training_scheme_9" "training_scheme_16" "training_scheme_4" "training_scheme_11" "training_scheme_19" "training_scheme_20" "training_scheme_22" "training_scheme_23" "training_scheme_24" "training_scheme_5" "training_scheme_6" "training_scheme_7" "training_scheme_25" "training_scheme_26"
# do
#     singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task1_focal_$dataset_name.log
# done

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_cell_lv3 benchmark_task_4.sh tenx_hlca_par_cell ann_level_3 dataset True
sbatch --partition=cpucourt --time=71:00:00 --job-name t4_cell_lv3 benchmark_task_4.sh tenx_hlca_par_cell ann_level_3 dataset False

#sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_nuc_lv3 benchmark_task_4.sh tenx_hlca_par_nuc ann_level_3 dataset True
#sbatch --partition=cpucourt --time=71:00:00 --job-name t4_nuc_lv3 benchmark_task_4.sh tenx_hlca_par_nuc ann_level_3 dataset False

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_bench_tenx_lv4 benchmark_task_4.sh tenx_hlca_par ann_level_4 dataset True
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_bench_tenx_lv4 benchmark_task_4.sh tenx_hlca_par ann_level_4 dataset False

# sbatch ---partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method


# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method  True
# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method True
