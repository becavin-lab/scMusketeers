import os
import shutil
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys

# Add path to source
sys.path.insert(0, "/workspace/cell/scMusketeers")
from scmusketeers.hpoptim import metrics

def test_metrics_saving():
    save_dir = "/workspace/cell/scMusketeers/test_metrics_output"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Mock workflow
    workflow = MagicMock()
    workflow.run_file.log_neptune = False
    
    # Mock metric dictionaries
    workflow.batch_metrics_list = {"batch_mixing_entropy": lambda x, y: 0.5}
    workflow.pred_metrics_list = {"acc": lambda x, y: 0.8}
    workflow.pred_metrics_list_balanced = {"balanced_acc": lambda x, y: 0.7}
    workflow.clustering_metrics_list = {"db_score": lambda x, y: 1.2}

    # Dummy data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    group = "test_group"
    
    # Batch mixing
    batch_list = {"test_group": np.array([[1, 0], [0, 1], [1, 0], [0, 1]])}
    enc = np.random.rand(4, 10)
    batches = np.array([0, 1, 0, 1])
    
    print("Testing metric_batch_mixing...")
    metrics.metric_batch_mixing(workflow, batch_list, group, enc, batches, save_dir)
    
    # Classification
    sizes = {"large": [0, 1]}
    print("Testing metric_classification...")
    metrics.metric_classification(workflow, y_pred, y_true, group, sizes, save_dir)

    # Clustering
    print("Testing metric_clustering...")
    metrics.metric_clustering(workflow, y_pred, group, enc, save_dir)
    
    # Check result
    csv_path = os.path.join(save_dir, "metrics.csv")
    if os.path.exists(csv_path):
        print(f"SUCCESS: {csv_path} created.")
        df = pd.read_csv(csv_path)
        print(df)
    else:
        print(f"FAILURE: {csv_path} NOT created.")

if __name__ == "__main__":
    test_metrics_saving()
