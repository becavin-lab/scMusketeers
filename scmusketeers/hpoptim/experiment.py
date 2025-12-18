### From run_hp.py
# import neptune
import pandas as pd
import sys
import os
import json
import neptune
import logging
import json
import gc

sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..arguments.neptune_log import NEPTUNE_INFO
    from ..arguments.neptune_log import (start_neptune_log, add_custom_log,
                                                stop_neptune_log)
    from ..arguments.runfile import set_hyperparameters
    from .dataset import Dataset, load_dataset
    from .hyperparameters import Workflow
    
except ImportError:
    from scmusketeers.arguments.neptune_log import NEPTUNE_INFO
    from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                stop_neptune_log)
    from scmusketeers.arguments.runfile import set_hyperparameters
    from scmusketeers.hpoptim.dataset import Dataset, load_dataset
    from scmusketeers.hpoptim.hyperparameters import Workflow

logger = logging.getLogger("Sc-Musketeers")

def load_json(json_path):
    """
    Load json with hyperparameters
    """
    with open(json_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


class MakeExperiment:
    def __init__(self, run_file, total_trial, random_seed):
        # super().__init__()
        self.run_file = run_file
        self.total_trial = total_trial
        self.random_seed = random_seed
        self.workflow = None
        self.neptune_name = run_file.neptune_name
        self.trial_count = 0
        
    def train(self, params):
        logger.info("Run the Experiment: experiment.train()")
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        # import tensorflow as tf

        logger.debug("Load checkpoint")
        self.trial_count += 1
        self.run_file.trial_name = f"HyperParam_Optim_{self.run_file.dataset_name}_{self.trial_count}_{self.total_trial}"
        
        # we run the trial
        self.workflow = Workflow(
            run_file=self.run_file
        )
        logger.debug("Set hyparameters")
        self.workflow.set_hyperparameters(params)
        start_neptune_log(self.workflow)
        logger.debug(f" Trial name -- {self.run_file.trial_name} -- ")
        add_custom_log(self.workflow,"task", self.run_file.task)
        add_custom_log(self.workflow,"trial_name", self.run_file.trial_name)
        add_custom_log(self.workflow,"total_trial", self.total_trial)
        add_custom_log(self.workflow,"hp_random_seed", self.random_seed)
        add_custom_log(self.workflow,"trial_count", self.trial_count)
        
        logger.debug("")
        self.workflow.process_dataset()
        logger.debug("")
        self.workflow.split_train_test()
        logger.debug("")
        self.workflow.split_train_val()
        logger.debug("Run the training")
        opt_metric = (
            self.workflow.make_experiment()
        )  
        logger.debug("Log the trial metrics on Neptune")
        add_custom_log(
            self.workflow,"opt_metric", self.run_file.opt_metric
        )
        stop_neptune_log(self.workflow)
        del self.workflow # Delete the large workflow object
        gc.collect() # Ask Python to free up memory

        return opt_metric

