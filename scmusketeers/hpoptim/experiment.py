### From run_hp.py
# import neptune
import pandas as pd
import sys
import os
import json
import neptune
import logging

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
        project = neptune.init_project(
            project="becavin-lab/"+run_file.neptune_name,
            api_token=NEPTUNE_INFO[run_file.neptune_name],
            mode="read-only",
        )  # For checkpoint
        self.runs_table_df = project.fetch_runs_table().to_pandas()
        project.stop()
        
    def train(self, params):
        logger.info("Run the Experiment: experiment.train()")
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        # import tensorflow as tf

        logger.debug("Load checkpoint")
        self.trial_count += 1
        trial_name = f"HyperParam_Optim_{self.run_file.dataset_name}_{self.trial_count}/{self.total_trial}"
        checkpoint = {"parameters/" + k: i for k, i in params.items()}
        checkpoint["parameters/dataset_name"] = self.run_file.dataset_name
        checkpoint["parameters/opt_metric"] = self.run_file.opt_metric
        checkpoint["parameters/task"] = trial_name
        #logger.debug(checkpoint)
        
        # common columns runs_table_df and checkpoint
        logger.debug("Compare checkpoint and Neptune dataframe")
        common_headers = []
        common_values = []
        for header in list(checkpoint.keys()):
            if header in self.runs_table_df.columns:
                common_headers.append(header)
                common_values.append(checkpoint[header])
        
        result = self.runs_table_df[
            self.runs_table_df[common_headers]
            .eq(common_values)
            .all(axis=1)
        ]
        #logger.debug(f"Neptune dataframe: {result}")

        
        split, metric = self.run_file.opt_metric.split("-")
        if result.empty or pd.isna(
            result.loc[:, f"evaluation/{split}/{metric}"].iloc[-1]
        ):  # we run the trial
            logger.debug(f"Trial {self.trial_count} does not exist, running trial")
            self.workflow = Workflow(
                run_file=self.run_file
            )
            logger.debug("Set hyparameters")
            self.workflow.set_hyperparameters(params)
            logger.debug("")
            start_neptune_log(self.workflow)
            logger.debug("Log the trial on Neptune")
            trial_name = f"HyperParam_Optim_{self.run_file.task}_{self.run_file.dataset_name}_{self.trial_count}/{self.total_trial}"
            logger.debug(f" -- {trial_name} -- ")
            add_custom_log(self.workflow,"task", trial_name)
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
            # del self.workflow  # Should not be necessary
            return opt_metric
        else:  # we return the already computed value
            logger.info(f"Trial {self.trial_count} already exists, retrieving value")
            return result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]
