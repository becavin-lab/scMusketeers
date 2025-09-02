import argparse
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import logging
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
try:
    from ..scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
    from ..scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log)
    from ..scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                                get_default_param, get_runfile)
    from ..scmusketeers.tools.utils import str2bool
    from ..scmusketeers.benchmark.benchmark import Workflow
except ImportError:
    from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log, add_custom_log)
    from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
    from scmusketeers.tools.utils import str2bool
    from scmusketeers.benchmark.benchmark import Workflow


logger = logging.getLogger("Sc-Musketeers")

model_list_cpu = ['pca_knn', 'pca_svm']#,'harmony_svm','uce','scmap_cells','scmap_cluster',]
model_list_gpu = ['scanvi'] #'scanvi'celltypist
def run_benchmark():
    # Set up logging
    logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")
    logger.setLevel(getattr(logging, "DEBUG"))
    
    logger.debug("Getting benchmark parameters")
    run_file = get_runfile()
    if run_file.gpu_models == "True" :
        model_list = model_list_gpu
    else:
        model_list = model_list_cpu
    logger.info(f"Get Models to run - GPU={run_file.gpu_models} - {model_list}")
    
    logger.info("Setup Workflow")
    experiment = Workflow(run_file=run_file)
    experiment.process_dataset(model_list)
    experiment.mode = "entire_condition"
    random_seed = 2

    logger.info("Get all datasets settings (train/test/split)")
    TOTAL_SPLIT_TEST = 3
    TOTAL_SPLIT_VAL = 5
    
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
                
                logger.debug(f'Running model - {model}')
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