import json
import os
import logging

try:
    from ..arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log)
    from ..arguments.runfile import (PROCESS_TYPE, create_argparser,
                                                get_default_param, get_runfile)
    from ..hpoptim.experiment import MakeExperiment
except ImportError:
    from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                    stop_neptune_log)
    from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                                get_default_param, get_runfile)
    from scmusketeers.hpoptim.experiment import MakeExperiment
    
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
except ImportError:
    print("Tried import scmusketeers.workflow but AX Platform not installed")
    print("Please consider installing AxPlatform for hyperparameters optimization")
    print("poetry install --with workflow")
    

# JSON_PATH_DEFAULT = '/home/acollin/scMusketeers/experiment_script/hp_ranges/'
JSON_PATH_DEFAULT = "/home/becavin/scMusketeers/experiment_script/hp_ranges/"

TOTAL_TRIAL = 20
RANDOM_SEED = 40

logger = logging.getLogger("Sc-Musketeers")

def load_json(hparam_path):
    """
    Load json file with hyperparameter ranges
    """
    with open(hparam_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


def run_workflow(run_file):
    """
    Run hyperparameter optimization

    - Create Experiment <br>
    - Load hyperparameters ranges<br>
    - Run AX Platform for hyperparameters optimization
    - Run all trials
    - Print/save best parameters

    """
    logger.info("Create Experiment")
    experiment = MakeExperiment(
        run_file=run_file,
        total_trial=TOTAL_TRIAL,
        random_seed=RANDOM_SEED
    )

    logger.info("Load hyperparameters ranges")
    if not run_file.hparam_path:
        logger.debug("hp ranges: ")
        hparam_path = JSON_PATH_DEFAULT + "generic_r1.json"
    else:
        hparam_path = run_file.hparam_path
    logger.info(f"hp ranges: {hparam_path}")
    hparams = load_json(hparam_path)

    ### Loop API
    # best_parameters, values, experiment, model = optimize(
    #     parameters=hparams,
    #     evaluation_function=experiment.train,
    #     objective_name=run_file.opt_metric,
    #     minimize=False,
    #     total_trials=experiment.total_trial,
    #     random_seed=experiment.random_seed,
    # )

    ### Service API
    logger.info("Run AX Platform for hyperparameters optimization")
    ax_client = AxClient()
    ax_client.create_experiment(
        name="scmusketeers",
        parameters=hparams,
        objectives={"opt_metric": ObjectiveProperties(minimize=False)},
    )
    for i in range(TOTAL_TRIAL):
        logger.info(f"Running trial {i}/{TOTAL_TRIAL}")
        parametrization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=experiment.train(parametrization),
        )

    logger.info("Hyperparam optimization finished")
    best_parameters, values = ax_client.get_best_parameters()
    logger.info(f" Best parameters {best_parameters}")
    ## TO DO : SAVE best parameters

    
