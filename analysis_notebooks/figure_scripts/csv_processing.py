import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def csv_process(results_dir, checkpoint_path):
    logger.info("Parsing result directories to build metrics table... This may take a while.")
    data = []
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
        
    folders = os.listdir(results_dir)
    total_folders = len(folders)
    logger.info(f"Found {total_folders} total folders to process.")
    
    for i, folder in enumerate(folders):
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{total_folders} folders...")
            
        folder_path = os.path.join(results_dir, folder)
        metrics_file = os.path.join(folder_path, "all_metrics.csv")
        
        if not os.path.isdir(folder_path) or not os.path.exists(metrics_file):
            continue

        if "_task_" not in folder:
            continue

        try:
            dataset_name, rest = folder.split("_task_", 1)
            parts = rest.split("_", 1)
            task = "task_" + parts[0]
            
            # Extract parameter
            parameter = parts[1] if len(parts) >= 2 else "None"
            
            model_name = "scPermut_default" # Legacy alias for scMusketeers
            if parameter.startswith(("celltypist", "scmap", "scanvi", "pca", "uce", "harmony")):
                subparts = parameter.split("_")
                if parameter.startswith("pca") and len(subparts) > 1 and subparts[1] in ["knn", "svm"]:
                    model_name = f"pca_{subparts[1]}"
                    if len(subparts) > 2:
                        parameter = "_".join(subparts[2:])
                    else:
                        parameter = "None"
                elif parameter.startswith("scmap") and len(subparts) > 1 and subparts[1] in ["cells", "cluster"]:
                    model_name = f"scmap_{subparts[1]}"
                    if len(subparts) > 2:
                        parameter = "_".join(subparts[2:])
                    else:
                        parameter = "None"
                else:
                    model_name = subparts[0]
                    if len(subparts) > 1:
                        parameter = "_".join(subparts[1:])
                    else:
                        parameter = "None"
                        
            # Safely extract test_fold_nb 
            test_fold_nb = -1
            if parameter != "None" and "_" in parameter:
                try:
                    test_fold_str = parameter.split("_")[0]
                    if "." not in test_fold_str:
                        test_fold_nb = int(test_fold_str)
                except ValueError:
                    pass
                
            metrics = pd.read_csv(metrics_file, index_col=0, header=None).squeeze("columns")
            
            val_balanced_acc = metrics.get('evaluation/val/balanced_acc', 0.0)
            test_balanced_acc = metrics.get('evaluation/test/balanced_acc', 0.0)
            full_batch_mixing_entropy = metrics.get('evaluation/full/batch_mixing_entropy', 0.0)

            data.append({
                'dataset_name': dataset_name,
                'task': task,
                'model': model_name,
                'test_fold_nb': test_fold_nb,
                'parameter': parameter,
                'val_balanced_acc': val_balanced_acc,
                'test_balanced_acc': test_balanced_acc,
                'full_batch_mixing_entropy': full_batch_mixing_entropy
            })
        except Exception as e:
            logger.debug(f"Failed to process folder {folder}: {e}")
            continue
            
    runs_table_df = pd.DataFrame(data)
    logger.info(f"Finished parsing. Extracted {len(runs_table_df)} valid runs.")
    
    if not runs_table_df.empty:
        for t, group_df in runs_table_df.groupby('task'):
            # Sort the CSV logically
            sorted_df = group_df.sort_values(by=["dataset_name", "model", "test_fold_nb"])
            
            task_checkpoint_path = checkpoint_path.replace('.csv', f'_{t}.csv')
            logger.info(f"Saving checkpoint for {t} to {task_checkpoint_path}")
            sorted_df.to_csv(task_checkpoint_path, index=False)
            
    return runs_table_df
