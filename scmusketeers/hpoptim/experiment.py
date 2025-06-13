### From run_hp.py
# import neptune
import pandas as pd
import sys
import os
import neptune

sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..arguments.neptune_log import NEPTUNE_INFO
    from ..arguments.runfile import set_hyperparameters
    from .dataset import Dataset, load_dataset
    from .hyperparameters import Workflow
except ImportError:
    from scmusketeers.arguments.neptune_log import NEPTUNE_INFO
    from scmusketeers.arguments.runfile import set_hyperparameters
    from scmusketeers.hpoptim.dataset import Dataset, load_dataset
    from scmusketeers.hpoptim.hyperparameters import Workflow


def load_json(json_path):
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
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        # import tensorflow as tf

        self.trial_count += 1
        # print('params')
        # print(params)
        checkpoint = {"parameters/" + k: i for k, i in params.items()}

        # dataset_name


        checkpoint["parameters/dataset_name"] = self.run_file.dataset_name
        checkpoint["parameters/opt_metric"] = self.run_file.opt_metric
        checkpoint["parameters/task"] = "hp_optim_V2"
        print(checkpoint)
        print(self.runs_table_df)
        # checkpoint = {'parameters/dataset_name': self.run_file.dataset_name,
        #               'parameters/total_trial': total_trial, 'parameters/trial_count': self.trial_count,
        #               'parameters/opt_metric': self.opt_metric, 'parameters/hp_random_seed': random_seed}
        
        # common columns runs_table_df and checkpoint
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
        print(result)
        split, metric = self.run_file.opt_metric.split("-")
        if result.empty or pd.isna(
            result.loc[:, f"evaluation/{split}/{metric}"].iloc[-1]
        ):  # we run the trial
            print(f"Trial {self.trial_count} does not exist, running trial")
            self.workflow = Workflow(
                run_file=self.run_file
            )
            self.workflow.set_hyperparameters(params)
            self.workflow.start_neptune_log(self.run_file.neptune_name)
            self.workflow.process_dataset()
            self.workflow.split_train_test()
            self.workflow.split_train_val()
            opt_metric = (
                self.workflow.make_experiment()
            )  # This starts the logging
            self.workflow.add_custom_log("task", "hp_optim_V2")
            self.workflow.add_custom_log("total_trial", self.total_trial)
            self.workflow.add_custom_log("hp_random_seed", self.random_seed)
            self.workflow.add_custom_log("trial_count", self.trial_count)
            self.workflow.add_custom_log(
                "opt_metric", self.run_file.opt_metric
            )
            self.workflow.stop_neptune_log()
            # del self.workflow  # Should not be necessary
            return opt_metric
        else:  # we return the already computed value
            print(f"Trial {self.trial_count} already exists, retrieving value")
            return result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]
