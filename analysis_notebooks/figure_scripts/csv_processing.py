import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# scMusketeers folders use this legacy display name in the output DataFrames
_MODEL_DISPLAY_NAME = {
    "scMusketeers": "scPermut_default",
}


def _load_completed_entries(paper_review_dir):
    """Read completed_benchmark_runs_taskX.csv files and return list of (dataset, task, model, parameter) tuples."""
    entries = []
    for task_num in ["1", "2"]:
        csv_path = os.path.join(paper_review_dir, f"completed_benchmark_runs_task{task_num}.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                entries.append((row["Dataset Name"], str(row["Task"]), row["Model"], row["Parameter"]))
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
    return entries


def _reconstruct_folder_name(dataset, task, model, parameter):
    """Reconstruct the result folder name from CSV row fields.

    Naming conventions (mirroring 00_benchmark_sbatch_review.py):
      task1 scMusketeers : {dataset}_task_1_{parameter}
      task1 others       : {dataset}_task_1_{model}_{parameter}
      task2 any model    : {dataset}_task_2_{model}_{parameter}
    """
    if task == "1" and model == "scMusketeers":
        return f"{dataset}_task_1_{parameter}"
    elif task == "1":
        return f"{dataset}_task_1_{model}_{parameter}"
    else:
        return f"{dataset}_task_2_{model}_{parameter}"


def _extract_test_fold_nb(parameter):
    """Extract integer test fold number from the first component of a parameter string."""
    if parameter and "_" in parameter:
        try:
            first_part = parameter.split("_")[0]
            if "." not in first_part:
                return int(first_part)
        except ValueError:
            pass
    return -1


def _read_metrics(metrics_file):
    metrics = pd.read_csv(metrics_file, index_col=0, header=None).squeeze("columns")
    return {
        "val_balanced_acc": metrics.get("evaluation/val/balanced_acc", 0.0),
        "test_balanced_acc": metrics.get("evaluation/test/balanced_acc", 0.0),
        "full_batch_mixing_entropy": metrics.get("evaluation/full/batch_mixing_entropy", 0.0),
    }


def csv_process(results_dir, checkpoint_path):
    logger.info("Parsing result directories to build metrics table...")
    data = []

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    paper_review_dir = os.path.abspath(
        os.path.join(results_dir, "..", "benchmark", "paper_review")
    )
    completed_entries = _load_completed_entries(paper_review_dir)

    if not completed_entries:
        logger.error(
            f"No completed runs CSVs found in {paper_review_dir}. "
            "Run 00_benchmark_sbatch_review.py first to generate them."
        )
        return pd.DataFrame()

    logger.info(f"Loaded {len(completed_entries)} completed runs from {paper_review_dir}.")
    for dataset, task, model, parameter in completed_entries:
        folder_name = _reconstruct_folder_name(dataset, task, model, parameter)
        metrics_file = os.path.join(results_dir, folder_name, "all_metrics.csv")
        if not os.path.isfile(metrics_file):
            logger.debug(f"Metrics file missing: {metrics_file}")
            continue
        try:
            data.append({
                "dataset_name": dataset,
                "task": f"task_{task}",
                "model": _MODEL_DISPLAY_NAME.get(model, model),
                "test_fold_nb": _extract_test_fold_nb(parameter),
                "parameter": parameter,
                **_read_metrics(metrics_file),
            })
        except Exception as e:
            logger.debug(f"Failed to process {folder_name}: {e}")

    runs_table_df = pd.DataFrame(data)
    logger.info(f"Finished parsing. Extracted {len(runs_table_df)} valid runs.")

    if not runs_table_df.empty:
        for t, group_df in runs_table_df.groupby("task"):
            sorted_df = group_df.sort_values(by=["dataset_name", "model", "test_fold_nb"])
            task_checkpoint_path = checkpoint_path.replace(".csv", f"_{t}.csv")
            logger.info(f"Saving checkpoint for {t} to {task_checkpoint_path}")
            sorted_df.to_csv(task_checkpoint_path, index=False)

    return runs_table_df
