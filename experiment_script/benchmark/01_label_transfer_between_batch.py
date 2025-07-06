import argparse
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import neptune
import logging
import os

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import torch
torch.set_float32_matmul_precision('medium') # Or 'high' if you're comfortable with more aggressive precision trade-offs

#WD_PATH = '/dhome/acollin/scPermut/'
#sys.path.append(WD_PATH)
sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log)
    from ..scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                                get_default_param, get_runfile)
    from ..scmusketeers.tools.utils import str2bool
    from ..scmusketeers.benchmark.benchmark import Workflow
except ImportError:
    from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log, add_custom_log)
    from scmusketeers.tools.utils import str2bool
    from scmusketeers.benchmark.benchmark import Workflow


logger = logging.getLogger("Sc-Musketeers")

model_list_cpu = ['pca_knn', 'pca_svm']#,'harmony_svm','uce','scmap_cells','scmap_cluster',]
model_list_gpu = ['celltypist'] #'scanvi'

def run_benchmark():
    parser = argparse.ArgumentParser()
    # Set up logging
    logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")
    logger.setLevel(getattr(logging, "DEBUG"))
    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'ajrccm_by_batch', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'manip', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=3000, default=0, help = "Number of hvg to use. If no tag, don't use hvg.")

    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--mode', type = str, default = 'entire_condition', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = None, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--test_obs', type = str,nargs='+', default = None, help ='batches from batch_key to use as test')
    parser.add_argument('--test_index_name', type = str,nargs='+', default = None, help ='indexes to be used as test. Overwrites test_obs')
    
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--neptune_name', type=str, nargs='?', default="scmusk-tasks" , help ='')
    parser.add_argument('--gpu_models', type=str2bool, nargs='?',const=False, default=True , help ='')
    parser.add_argument('--resume', type=str2bool, nargs='?',const=False, default=False , help ='')

    working_dir = "/data/analysis/data_becavin/scMusketeers-data"
    #working_dir = '/workspace/cell/scMusketeers'
    parser.add_argument('--working_dir', type=str, nargs='?', default=working_dir, help ='')
    # parser.add_argument('--working_dir', type=str, nargs='?',const='/workspace/cell/scMusketeers', default='/workspace/cell/scMusketeers', help ='')
    parser.add_argument(
        "--ref_path",
        type=str,
        help="Path of the referent adata file (example : data/ajrccm.h5ad",
        default=os.path.join(working_dir,"data","ajrccm_by_batch.h5ad"),
    )

    run_file = parser.parse_args()
    logger.info(f"{run_file.class_key} , {run_file.batch_key}")
    working_dir = run_file.working_dir
    logger.info(f'working directory : {working_dir}')

    project = neptune.init_project(
            project="becavin-lab/scmusk-tasks",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            mode="read-only",
            )# For checkpoint

    logger.info(f"Fetch neptune history - {run_file.neptune_name}")
    runs_table_df = project.fetch_runs_table().to_pandas()
    project.stop()
    
    if run_file.gpu_models :
        model_list = model_list_gpu
    else:
        model_list = model_list_cpu
    logger.info(f"Get Models to run - GPU={run_file.gpu_models} - {model_list}")
    

    logger.info("Setup Workflow")
    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset(model_list)

    experiment.mode = "entire_condition"

    random_seed = 2

    logger.info("Get all datasets settings (train/test/split)")
    TOTAL_SPLIT_TEST = 1
    TOTAL_SPLIT_VAL = 1
    
    X = experiment.dataset.adata.X
    classes = experiment.dataset.adata.obs[experiment.class_key]
    groups = experiment.dataset.adata.obs[experiment.batch_key]
    n_batches = len(groups.unique())
    nfold_test = max(1,round(n_batches/5)) # if less than 8 batches, this comes to 1 batch per fold, otherwise, 20% of the number of batches for test
    kf_test = GroupShuffleSplit(n_splits=TOTAL_SPLIT_TEST, test_size=nfold_test, random_state=random_seed)
    test_split_key = experiment.dataset.test_split_key

    logger.info(f"Loop on kf_test {kf_test}")
    for i, (train_index, test_index) in enumerate(kf_test.split(X, classes, groups)):
        logger.info(f" kf_test temp is {i}, ({train_index}, {test_index})")
        groups = groups        
        test_obs = list(groups.iloc[test_index].unique()) # the batches that go in the test set

        experiment.dataset.test_split(test_obs = test_obs) # splits the train and test dataset
        nfold_val = max(1,round((n_batches-len(test_obs))/5)) # represents 20% of the remaining train set
        kf_val = GroupShuffleSplit(n_splits=TOTAL_SPLIT_VAL, test_size=nfold_val, random_state=random_seed)

        logger.info(f"Setup train and value datasets")
        X_train_val = experiment.dataset.adata_train_extended.X
        classes_train_val = experiment.dataset.adata_train_extended.obs[experiment.class_key]
        groups_train_val = experiment.dataset.adata_train_extended.obs[experiment.batch_key]

        logger.info(f"Loop on kf_val {kf_val}")
        for j, (train_index, val_index) in enumerate(kf_val.split(X_train_val, classes_train_val, groups_train_val)):
            experiment.keep_obs = list(groups_train_val[train_index].unique()) # keeping only train idx
            val_obs = list(groups_train_val[val_index].unique())
            logger.debug(f"Fold {i,j}:")
            logger.debug(f"train = {list(groups_train_val.iloc[train_index].unique())}, len = {len(groups_train_val.iloc[train_index].unique())}")
            logger.debug(f"val = {list(groups_train_val.iloc[val_index].unique())}, len = {len(groups_train_val.iloc[val_index].unique())}")
            logger.debug(f"test = {list(groups.iloc[test_index].unique())}, len = {len(groups.iloc[test_index].unique())}")

            
            logger.debug(set(groups_train_val.iloc[train_index].unique()) & set(groups.iloc[test_index].unique()))
            logger.debug(set(groups_train_val.iloc[train_index].unique()) & set(groups_train_val.iloc[val_index].unique()))
            logger.debug(set(groups_train_val.iloc[val_index].unique()) & set(groups.iloc[test_index].unique()))
            
            experiment.split_train_test_val()
            logger.debug(experiment.dataset.adata.obs.loc[:,[experiment.test_split_key,experiment.batch_key]].drop_duplicates())
            
            logger.info(f"Loop on the models {model_list}")
            for model in model_list:
                logger.info(f"Running model: {model}")
                checkpoint={'parameters/dataset_name': experiment.dataset_name, 'parameters/task': 'task_1', 'parameters/use_hvg': experiment.use_hvg,
                    'parameters/model': model, 'parameters/test_fold_nb':i,'parameters/val_fold_nb':j}
                logger.debug("Compare checkpoint and Neptune dataframe")
                common_headers = []
                common_values = []
                for header in list(checkpoint.keys()):
                    if header in runs_table_df.columns:
                        common_headers.append(header)
                        common_values.append(checkpoint[header])
                logger.debug(f"Common: {common_headers}")
                result = runs_table_df[
                    runs_table_df[common_headers]
                    .eq(common_values)
                    .all(axis=1)
                ]
                logger.debug(f"Result {result}")

                logger.debug(f"Resume {run_file.resume}")
                if result.empty or run_file.resume == False:
                    logger.debug(f'Running model - {model}')
                    logger.debug(f"Checkpoint: {checkpoint}")
                    start_neptune_log(experiment)
                    trial_name = f"Task1_{model}_{run_file.dataset_name}_test_{i}_val_{j}"
                    logger.debug(f" -- {trial_name} -- ")
                    add_custom_log(experiment,"task", "task1")
                    add_custom_log(experiment,"trial_name", trial_name)
                    add_custom_log(experiment,'test_fold_nb',i)
                    add_custom_log(experiment,'val_fold_nb',j)
                    add_custom_log(experiment,'test_obs',test_obs)
                    add_custom_log(experiment,'val_obs',val_obs)
                    add_custom_log(experiment,'train_obs',experiment.keep_obs)
                    add_custom_log(experiment,'task','task_1')
                    add_custom_log(experiment,'deprecated_status','False')
                    logger.debug(f"Train model {model}")
                    experiment.train_model(model)
                    logger.debug(f"Compute Metrics for {model}")
                    experiment.compute_metrics()
                    stop_neptune_log(experiment)

if __name__ == '__main__':
    run_benchmark()