import os
import logging
import csv
import json
from importlib.metadata import version

from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                stop_neptune_log)
from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)

import scmusketeers.transfer.optimize_model as om
import scmusketeers.hpoptim.hyperparameters as hp

from scmusketeers.transfer.dataset_tf import process_dataset
from scmusketeers.hpoptim.run_workflow import run_workflow

logger = logging.getLogger("Sc-Musketeers")

def run_sc_musketeers():
    # Set up logging
    logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")
    logger.info(f"Sc-Musketeers {version("sc-musketeers")} started")

    # Get all arguments
    run_file = get_runfile()
    
    # Set logger level
    if run_file.debug:
        logger.setLevel(getattr(logging, "DEBUG"))
    else:
        logger.setLevel(getattr(logging, "INFO"))
    logger.debug(f"Program arguments: {run_file}")

    # set model parameters if provided
    logger.debug(f"Bestparam path: {run_file.bestparam_path}")

    if "none.csv" in run_file.bestparam_path.strip():
        run_file.bestparam_path=""
    
    if (not run_file.bestparam_path.strip()=="") and (not run_file.bestparam_path.strip()=="none.csv"):
        logger.info(f"Setting hyperparameters provided for the model in: {run_file.bestparam_path}")
        # Open the CSV file
        with open(run_file.bestparam_path, mode='r', newline='') as csv_file:
            # Create a DictReader object
            reader = csv.DictReader(csv_file)
            
            # Read the first data row into a dictionary
            # The next() function retrieves the next item from an iterator
            data_dict = next(reader)


        print(data_dict)
        if "use_hvg" in data_dict:
            run_file.use_hvg = int(data_dict["use_hvg"])
        if "batch_size" in data_dict:
            run_file.batch_size = int(data_dict["batch_size"])
        if "clas_w" in data_dict:
            run_file.clas_w = float(data_dict["clas_w"])
        if "dann_w" in data_dict:
            run_file.dann_w = float(data_dict["dann_w"])
        if "rec_w" in data_dict:
            run_file.rec_w = float(data_dict["rec_w"])
        if "ae_bottleneck_activation" in data_dict:
            run_file.ae_bottleneck_activation = data_dict["ae_bottleneck_activation"]
        if "clas_loss_name" in data_dict:
            run_file.clas_loss_name = data_dict["clas_loss_name"]
        if "size_factor" in data_dict:
            run_file.size_factor = data_dict["size_factor"]
        if "weight_decay" in data_dict:
            run_file.weight_decay = float(data_dict["weight_decay"])
        if "learning_rate" in data_dict:
            run_file.learning_rate = float(data_dict["learning_rate"])
        if "warmup_epoch" in data_dict:
            run_file.warmup_epoch = int(data_dict["warmup_epoch"])
        if "dropout" in data_dict:
            run_file.dropout = float(data_dict["dropout"])
        if "layer1" in data_dict:
            run_file.layer1 = int(data_dict["layer1"])
        if "layer2" in data_dict:
            run_file.layer2 = int(data_dict["layer2"])
        if "bottleneck" in data_dict:
            run_file.bottleneck = int(data_dict["bottleneck"])
        if "training_scheme" in data_dict:
            run_file.training_scheme = data_dict["training_scheme"]

    # Run transfer
    if run_file.process == PROCESS_TYPE[0]:
        logger.info("Run transfer")
        # Transfer data
        workflow = om.Workflow(run_file=run_file)
        start_neptune_log(workflow)
        process_dataset(workflow)
        workflow.dataset.train_val_split_transfer()
        workflow.dataset.create_inputs()
        adata_pred, model, history, X_scCER, query_pred = (
            workflow.make_experiment()
        )        
        stop_neptune_log(workflow)
        adata_pred_path = os.path.join(
            run_file.out_dir, f"{run_file.out_name}.h5ad"
        )
        logger.info((f"Save adata_pred to {adata_pred_path}"))
        adata_pred.write_h5ad(adata_pred_path)

    # Run hyperparameters optimization
    elif run_file.process == PROCESS_TYPE[1]:
        logger.info("Run hyperparameters optimization")
        run_workflow(run_file)
    # Run models benchmark
    elif run_file.process == PROCESS_TYPE[2]:
        logger.info("Run benchmark")
    # Run single HP trial
    elif run_file.process == PROCESS_TYPE[3]:
        logger.info("Run single HP trial")
        if not run_file.trial_params_path or not os.path.exists(run_file.trial_params_path):
            raise ValueError("trial_params_path must be provided and exist for hp_optim_single")
        
        with open(run_file.trial_params_path, 'r') as f:
            params = json.load(f)
                
        workflow = hp.Workflow(run_file=run_file)
        workflow.set_hyperparameters(params)
        
        start_neptune_log(workflow)
        # Add custom logs similar to experiment.py if needed, but Ax usually tracks this via param dict
        # We can add a tag or something if needed.
        
        workflow.process_dataset()
        workflow.dataset.test_split(
            test_obs=workflow.test_obs,
            test_index_name=workflow.test_index_name,
        )
        workflow.split_train_val()
        
        opt_metric = workflow.make_experiment()
        
        if run_file.trial_result_path:
             with open(run_file.trial_result_path, 'w') as f:
                 json.dump({"opt_metric": opt_metric}, f)
        
        stop_neptune_log(workflow)

    else:
        # No process
        logger.info("Process not recognized")


if __name__ == "__main__":
    run_sc_musketeers()
