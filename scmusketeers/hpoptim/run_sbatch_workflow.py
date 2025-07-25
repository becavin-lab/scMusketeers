import json
import os
import logging
import time

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
    logger.warning("Tried import scmusketeers.workflow but AX Platform not installed")
    logger.warning("Please consider installing AxPlatform if you want to perform hyperparameters optimization")
    logger.warning("poetry install --with workflow")
    
logger = logging.getLogger("Sc-Musketeers")
JSON_PATH_DEFAULT = "experiment_script/hp_ranges/"
RANDOM_SEED = 40
MAX_CONCURRENT_TRIALS = 4 # Set this to the number of GPUs you want to use
RESULTS_DIR = "trial_results" # Directory to store intermediate results


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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("slurm_logs", exist_ok=True)
    experiment = MakeExperiment(
        run_file=run_file,
        total_trial=run_file.total_trial,
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
        name="scmusketeers_parallel",
        parameters=hparams,
        objectives={"opt_metric": ObjectiveProperties(minimize=False)},
    )
    # --- 2. Orchestration Loop ---
    running_trials = {} # {trial_index: parametrization}
    
    for i in range(run_file.total_trial):
        # A. Poll for completed trials before launching new ones
        completed_indices = []
        for trial_index, params in running_trials.items():
            result_path = os.path.join(RESULTS_DIR, f"result_{trial_index}.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    raw_data = json.load(f)
                logger.info(f"Completing trial {trial_index} with result: {raw_data}")
                ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                completed_indices.append(trial_index)

        for index in completed_indices:
            del running_trials[index]

        # B. Launch new trials if there is capacity
        while len(running_trials) < MAX_CONCURRENT_TRIALS:
            current_run_count = i + len(running_trials)
            if current_run_count >= run_file.total_trial:
                break # Don't schedule more trials than total_trial

            parametrization, trial_index = ax_client.get_next_trial()
            running_trials[trial_index] = parametrization
            logger.info(f"Launching trial {trial_index} with params: {parametrization}")

            # Use subprocess to launch the sbatch job
            params_str = json.dumps(parametrization)
            cmd = [
                "sbatch",
                "--parsable", # Makes sbatch output just the job ID
                "sbatch_template.sh",
                str(trial_index),
                params_str,
                run_file.path, # Pass the path to your run configuration
                RESULTS_DIR
            ]
            subprocess.run(cmd)

        # C. Wait before polling again to avoid spamming the filesystem
        logger.info(f"Status: {len(running_trials)} running trials. Waiting for results...")
        time.sleep(30) # Poll every 30 seconds

    # --- 3. Finalize and Save Best Parameters (same as before) ---
    logger.info("#HP_optim--- Hyperparam optimization finished")
    best_parameters, values = ax_client.get_best_parameters()
    logger.debug(f"Best parameters {best_parameters}")
    path_bestparam = os.path.join(run_file.out_dir,run_file.out_name+run_file.dataset_name+"_best_hp.json")
    with open(path_bestparam, 'w') as json_file:
        json.dump(best_parameters, json_file, indent=4)
    logger.info(f"Best hyperparameters saved in {path_bestparam}")

    
