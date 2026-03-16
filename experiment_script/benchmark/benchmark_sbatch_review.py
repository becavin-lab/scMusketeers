import os
import sys
import csv

MODELS = ["scMusk", "celltypist", "scmap_cells", "scmap_cluster", "pca_knn", "pca_svm", "uce"]

def get_expected_parameters(task):
    """Return the expected parameter list for a given task."""
    expected_params = []
    
    if task == "1":
        # task1: i from 0 to 2, j from 0 to 3 -> i_j
        for i in range(3):
            for j in range(4):
                expected_params.append(f"{i}_{j}")
    elif task == "2":
        # task2: i from [0.05, 0.1, 0.5, 0.9], j from [30, 31, 32, 33, 34, 35] -> i_j
        i_vals = ["0.05", "0.1", "0.5", "0.9"]
        j_vals = ["30", "31", "32", "33", "34", "35"]
        for i in i_vals:
            for j in j_vals:
                expected_params.append(f"{i}_{j}")
                
    return expected_params

def find_completed_runs(results_dir):
    """Search for folders containing all_metrics.csv and extract dataset/task info."""
    completed_runs = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, "all_metrics.csv")):
                if "_task_" in folder:
                    dataset_name, rest = folder.split("_task_", 1)
                    parts = rest.split("_", 1)
                    task = parts[0] if len(parts) >= 1 else "Unknown"
                    parameter = parts[1] if len(parts) >= 2 else "None"
                    
                    # Extract model from parameter
                    model_name = "scMusk"
                    if "_" in parameter:
                        first_val = parameter.split("_", 1)[0]
                        if first_val in ["celltypist", "scmap", "scanvi", "pca", "uce"]:
                            # It's an established baseline model, we need to extract the whole model name
                            # parameter format e.g: celltypist_0_0, pca_svm_0_0, scmap_cells_0_0
                            parts = parameter.split("_")
                            if first_val == "pca" and len(parts) > 1 and parts[1] in ["knn", "svm"]:
                                model_name = f"pca_{parts[1]}"
                            elif first_val == "scmap" and len(parts) > 1 and parts[1] in ["cells", "cluster"]:
                                model_name = f"scmap_{parts[1]}"
                            else:
                                model_name = first_val
                                
                    elif parameter in ["celltypist", "scanvi", "scmap", "pca", "uce"]:
                         model_name = parameter

                    completed_runs.append({
                        "dataset": dataset_name,
                        "task": task,
                        "parameter": parameter,
                        "model": model_name
                    })
                else:
                    completed_runs.append({
                        "dataset": folder,
                        "task": "Unknown",
                        "parameter": "Unknown",
                        "model": "Unknown"
                    })
    return completed_runs


def save_to_csv(completed_runs, output_dir):
    """Save the completed runs data to separate CSV files by task."""
    if not completed_runs:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Group runs by task
    runs_by_task = {}
    for r in completed_runs:
        task = r["task"]
        if task not in runs_by_task:
            runs_by_task[task] = []
        runs_by_task[task].append(r)
        
    for task, runs in runs_by_task.items():
        file_name = f"completed_benchmark_runs_task{task}.csv"
        csv_path = os.path.join(output_dir, file_name)
        
        # Sort runs alphabetically by dataset, then model, then parameters
        sorted_runs = sorted(runs, key=lambda x: (x["dataset"], x["model"], x["parameter"]))
        
        try:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Dataset Name", "Task", "Model", "Parameter"])
                for r in sorted_runs:
                    writer.writerow([r["dataset"], r["task"], r["model"], r["parameter"]])
            print(f"Successfully saved table to {csv_path}")
        except Exception as e:
            print(f"Failed to save CSV {csv_path}: {e}")

def find_missing_runs(completed_runs, output_dir):
    """Identify missing runs per dataset, task, and model, and save them to CSV files."""
    if not completed_runs:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize completed runs structure: {dataset: {task: {model: set(parameters)}}}
    datasets = set()
    tasks_found = set()
    completed_mapped = {}
    
    for r in completed_runs:
        ds = r["dataset"]
        task = r["task"]
        model = r["model"]
        param = r["parameter"]
        
        # We only really care about tasks 1 and 2 for missing calculation
        if task not in ["1", "2"]:
            continue
            
        datasets.add(ds)
        tasks_found.add(task)
        
        if ds not in completed_mapped:
            completed_mapped[ds] = {}
        if task not in completed_mapped[ds]:
            completed_mapped[ds][task] = {}
        if model not in completed_mapped[ds][task]:
            completed_mapped[ds][task][model] = set()
            
        completed_mapped[ds][task][model].add(param)
        
    for task in tasks_found:
        missing_runs = []
        expected_params = get_expected_parameters(task)
        
        for ds in sorted(datasets):
            for model in MODELS:
                # Need to account for the way we extracted model prefix
                # Parameter format was either "i_j" (scMusk) or "{model}_i_j"
                for expected_param in expected_params:
                    # Construct what the original parameter string would have looked like
                    if model == "scMusk":
                        search_param = expected_param
                    else:
                        search_param = f"{model}_{expected_param}"
                        
                    # Check if this run is in completed_mapped
                    is_completed = False
                    if ds in completed_mapped and task in completed_mapped[ds] and model in completed_mapped[ds][task]:
                        if search_param in completed_mapped[ds][task][model]:
                            is_completed = True
                            
                    if not is_completed:
                        missing_runs.append({
                            "dataset": ds,
                            "task": task,
                            "model": model,
                            "parameter": search_param
                        })
                        
        if missing_runs:
            file_name = f"missing_benchmark_runs_task{task}.csv"
            csv_path = os.path.join(output_dir, file_name)
            
            try:
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Dataset Name", "Task", "Model", "Missing Parameter"])
                    for r in missing_runs:
                        writer.writerow([r["dataset"], r["task"], r["model"], r["parameter"]])
                print(f"Successfully saved missing runs to {csv_path}")
            except Exception as e:
                print(f"Failed to save CSV {csv_path}: {e}")

def main():
    # Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
    csv_dir = os.path.join(script_dir, "paper_review")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # 1. Search for completed metrics
    completed_runs = find_completed_runs(results_dir)

    # 2. Save completed to CSV
    save_to_csv(completed_runs, csv_dir)
    
    # 3. Find missing and save to CSV
    find_missing_runs(completed_runs, csv_dir)

if __name__ == "__main__":
    main()
