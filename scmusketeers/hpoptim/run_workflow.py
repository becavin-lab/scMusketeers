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
    
logger = logging.getLogger("Sc-Musketeers")

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
except ImportError:
    logger.warning("Tried import scmusketeers.workflow but AX Platform not installed")
    logger.warning("Please consider installing AxPlatform if you want to perform hyperparameters optimization")
    logger.warning("poetry install --with workflow")
    

JSON_PATH_DEFAULT = "experiment_script/hp_ranges/"

TOTAL_TRIAL = 8
RANDOM_SEED = 40


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
    logger.info("#HP_optim--- Create Experiment")
    experiment = MakeExperiment(
        run_file=run_file,
        total_trial=TOTAL_TRIAL,
        random_seed=RANDOM_SEED
    )

    logger.info("#HP_optim--- Load hyperparameters ranges")
    if not run_file.hparam_path:
        logger.debug("hp ranges: ")
        hparam_path = JSON_PATH_DEFAULT + "generic_r1.json"
    else:
        hparam_path = run_file.hparam_path
    logger.info(f"hp ranges: {hparam_path}")
    hparams = load_json(hparam_path)


    ## Service API
    logger.info("#HP_optim--- Run AX Platform for hyperparameters optimization")
    ax_client = AxClient()
    ax_client.create_experiment(
        name="scmusketeers",
        parameters=hparams,
        objectives={"opt_metric": ObjectiveProperties(minimize=False)},
    )
    for i in range(TOTAL_TRIAL):
        logger.info(f"#HP_optim--- Running trial {i}/{TOTAL_TRIAL}")
        parametrization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=experiment.train(parametrization),
        )

    logger.info("#HP_optim--- Hyperparam optimization finished")
    best_parameters, values = ax_client.get_best_parameters()
    best_parameters = {'use_hvg': 654, 'batch_size': 357, 'clas_w': 75.9059544539962, 'dann_w': 0.0006800634789046336, 'learning_rate': 0.007181802826807251, 'weight_decay': 3.457969993736587e-05, 'warmup_epoch': 1, 'dropout': 0.17059661448001862, 'bottleneck': 42, 'layer2': 153, 'layer1': 905}
    logger.debug(f"Best parameters {best_parameters}")
    path_bestparam = os.path.join(run_file.out_dir,run_file.out_name+run_file.dataset_name+"_best_hp.json")
    with open(path_bestparam, 'w') as json_file:
        json.dump(best_parameters, json_file, indent=4)
    logger.info(f"Best hyperparameters saved in {path_bestparam}")

    
