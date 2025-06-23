import os
import logging

from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                stop_neptune_log)
from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
from scmusketeers.hpoptim.experiment import MakeExperiment
from scmusketeers.transfer.optimize_model import Workflow
from scmusketeers.hpoptim.run_workflow import run_workflow

logger = logging.getLogger("Sc-Musketeers")

def run_sc_musketeers():

    # Set up logging
    logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")

    # Get all arguments
    run_file = get_runfile()
    
    # Set logger level
    if run_file.debug:
        logger.setLevel(getattr(logging, "DEBUG"))
    else:
        logger.setLevel(getattr(logging, "INFO"))
    logger.debug(f"Program arguments: {run_file}")

    # Run transfer
    if run_file.process == PROCESS_TYPE[0]:
        logger.info("Run transfer")
        # Transfer data
        workflow = Workflow(run_file=run_file)
        start_neptune_log(workflow)
        workflow.process_dataset()
        workflow.train_val_split()
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
    else:
        # No process
        logger.info("Process not recognized")

if __name__ == "__main__":
    run_sc_musketeers()
