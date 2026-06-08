# Claude Code — Work Summary

## task1_New sbatch refactoring
- Removed the inner `for test_fold_j_nb in 0..4` loop from `task1_New_all_benchmark.sh` and `task1_New_scMusk_all_dataset.sh` — one sbatch job per (dataset, model, fold_i) instead of one per j value
- Updated `task1_New_allmodels_benchmark.sh` and `task1_New_scMusk_benchmark.sh` to accept optional j indices as trailing positional args (`${@:N}`) passed to `--test_fold_j_nb`
- Updated `task1_New_allmodels_label_transfer_between_batch.py` and `task1_New_scMusk_label_transfer_between_batch.py` to accept `--test_fold_j_nb` as `nargs='+'` (list of ints), with a per-j skip guard

## task1_New complete script
- Rewrote `task1_New_all_benchmark_complete.sh`: AWK groups missing runs by `(dataset, model, fold_i)` and collects missing j values per group, passing them as trailing args to sbatch — avoids rerunning already-completed j folds

## task2_New seed handling
- Updated `task2_New_allmodels_label_transfer_pct_split.py` and `task2_New_scMusk_label_transfer_pct_split.py` to accept `--random_seed_nb` as `nargs='+'`
- Updated `task2_New_allmodels_benchmark.sh` and `task2_New_scMusk_benchmark.sh` to pass optional seed indices as trailing args
- Rewrote `task2_New_all_benchmark_complete.sh`: AWK groups by `(dataset, model, fold, pct)`, converts seed values to indices (`seed_val - 30`), passes only missing seeds to sbatch

## Figure generation — csv_process separation
- Added `task_filter` parameter to `csv_process()` in `csv_processing.py` so only selected tasks are processed
- Updated `figure_generation.py` to call `csv_process` separately for task1 and for the task2_New group (task1_New + task2 + task2_New), driven by the active task flags

## Figure generation — new datasets and size grouping
- Added `SmallIntestine-All` and `SmallIntestine-20k` to `generate_figure_task1_New.py` and `generate_figure_task2_New.py`
- Replaced the `tabula_sapiens / others` dataset grouping with 3 size-based groups:
  - **small** (≤ 20k cells): TS-Neural, TS-Skin, SmallIntestine-20k
  - **medium** (20k–100k cells): TS-Liver, TS-BoneMarrow, PBMC-Lee, TS-Blood
  - **large** (> 100k cells): Ageing-Mouse-All, SmallIntestine-All, CellCards-Lung

## UCE / GPU diagnostics
- Diagnosed `ModuleNotFoundError: No module named 'pkg_resources'` in UCE env → fix: `pip install setuptools`
- Diagnosed `cudaErrorNoKernelImageForDevice` on gpu02 (V100, sm_70) → cause: PyTorch 2.7.1 dropped sm_70 support
- Fix: create `UCE_v100` conda env with PyTorch 2.4.0+cu121 (last version with solid V100 support)
- Workaround for `conda create --clone` failure with pip packages: export via `pip freeze`, filter out torch/nvidia/triton, recreate env manually

## SLURM job management
- Cancelled batches of `t2n_complete_*` and `t2n_allm` jobs (GPU queue overflow / wrong GPU node)
