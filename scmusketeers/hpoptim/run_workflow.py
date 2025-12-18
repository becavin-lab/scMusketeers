import json
import os
import logging
import sys
import subprocess
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
    
logger = logging.getLogger("Sc-Musketeers")

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
except ImportError:
    logger.warning("Tried import scmusketeers.workflow but AX Platform not installed")
    logger.warning("Please consider installing AxPlatform if you want to perform hyperparameters optimization")
    logger.warning("poetry install --with workflow")
    

JSON_PATH_DEFAULT = "experiment_script/hp_ranges/"

RANDOM_SEED = 40


def load_json(hparam_path):
    """
    Load json file with hyperparameter ranges
    """
    with open(hparam_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


def run_trial_sbatch(experiment, params, trial_index, run_file):
    """
    Run a single trial via sbatch
    """
    # 1. Prepare paths
    trial_dir = os.path.join(run_file.out_dir, "hp_trials")
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    filename_base = f"{run_file.dataset_name}_hp_trial_{trial_index}"
    params_path = os.path.join(trial_dir, f"{filename_base}_params.json")
    result_path = os.path.join(trial_dir, f"{filename_base}_result.json")
    
    # Check resume logic
    if run_file.hp_resume:
        if os.path.exists(result_path):
            logger.info(f"Trial {trial_index} already completed and hp_resume=True. Skipping submission.")
            return result_path, "-1"
    else:
        if os.path.exists(result_path):
            logger.info(f"Trial {trial_index} result exists but hp_resume=False. Deleting old result.")
            os.remove(result_path)
    
    params["trial_name"] = f"HyperParam_Optim_{run_file.dataset_name}_{trial_index}_{run_file.total_trial}"

    # Write params
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
        
    # 2. Construct command
    # Reconstruct command line arguments, replacing process type and adding trial args
    cmd_args = list(sys.argv)
    # cmd_args[0] is the script name/path
    # We assume the command structure is `sc-musketeers hp_optim ...`
    # We need to find 'hp_optim' and replace it with 'hp_optim_single'
    
    try:
        proc_idx = cmd_args.index("hp_optim")
        cmd_args[proc_idx] = "hp_optim_single"
    except ValueError:
        # Fallback if hp_optim not explicitly in argv (unlikely if called via entry point)
        logger.warning("Could not find 'hp_optim' in sys.argv, appending 'hp_optim_single'")
        cmd_args.append("hp_optim_single")
        
    # Append new args
    cmd_args.extend(["--trial_params_path", params_path])
    cmd_args.extend(["--trial_result_path", result_path])
    
    # Filter out args that might conflict or be redundant if needed
    # (e.g. --total_trial, --hparam_path are ignored by single runner anyway)
    
    full_cmd = " ".join(cmd_args)
    
    # 3. Create sbatch script
    log_dir = os.path.join(run_file.out_dir, "logs","sbatch")
    if not os.path.exists(log_dir):
        try:
             os.makedirs(log_dir)
        except OSError as e:
             logger.warning(f"Could not create log dir {log_dir}: {e}")
             # Fallback or proceed if already exists (race condition handled)
    
    sbatch_script_path = os.path.join(trial_dir, f"{filename_base}.sh")
    log_out = os.path.join(log_dir, f"{filename_base}.out")
    log_err = os.path.join(log_dir, f"{filename_base}.err")
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={trial_index}_{run_file.dataset_name}_scmusk_hp
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --account=cell
#SBATCH --partition=cpucourt
#SBATCH --mem=8G
#SBATCH --time=10:00:00

# Environment setup
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module load miniconda
source activate scmusk-dev

echo "Running trial {trial_index}"
{full_cmd}
"""
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_content)
        
    # 4. Submit job
    logger.info(f"Submitting sbatch job for trial {trial_index}")
    try:
        result = subprocess.run(
            ["sbatch", sbatch_script_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        #job_id = "-1"
        job_id = result.stdout.strip().split()[-1] # "Submitted batch job 12345"
        logger.info(f"Submitted job {job_id}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit sbatch job: {e.stderr}")
        raise e
        
    # Return result path and job id
    return result_path, job_id


def is_job_running(job_id):
    """
    Check if a slurm job is still running/pending using squeue
    """
    try:
        # Check if job exists in squeue
        # -h: no header, -j: job id, -o: format (state)
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def run_workflow(run_file):
    """
    Run hyperparameter optimization
    """
    logger.info("#HP_optim--- Create Experiment")
    logger.info(f"run_file: {run_file.out_dir}")
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
        name="scmusketeers",
        parameters=hparams,
        objectives={"opt_metric": ObjectiveProperties(minimize=False)},
    )
    
    submitted_trials = [] # List of dicts: {trial_index, result_path, job_id}
    started_trials_count = 0
    completed_trials_count = 0
    
    logger.info(f"#HP_optim--- Starting optimization loop for {run_file.total_trial} trials...")
    
    while completed_trials_count < run_file.total_trial:
        
        # 1. Collection Phase: Check running trials
        # Iterate backwards to safely remove items
        for i in range(len(submitted_trials) - 1, -1, -1):
            trial_info = submitted_trials[i]
            res_path = trial_info["result_path"]
            t_idx = trial_info["trial_index"]
            job_id = trial_info["job_id"]
            
            if os.path.exists(res_path):
                # Result found
                try:
                    with open(res_path, "r") as f:
                        res_data = json.load(f)
                    
                    metric = res_data["opt_metric"]
                    logger.info(f"Trial {t_idx} completed with metric: {metric}")
                    
                    ax_client.complete_trial(
                        trial_index=t_idx,
                        raw_data=metric,
                    )
                    completed_trials_count += 1
                    logger.info(f"#HP_optim--- Progress: {completed_trials_count}/{run_file.total_trial} completed")
                    
                except Exception as e:
                    logger.error(f"Error processing result for trial {t_idx}: {e}")
                    ax_client.log_trial_failure(trial_index=t_idx)
                    completed_trials_count += 1 # Count as completed (failed)
                
                # Remove from list
                submitted_trials.pop(i)
                
            else:
                # Result NOT found, check if job is still running/queued
                # If job is NOT in squeue, it finished (failed/cancelled) without writing result
                # Only check if job_id is valid (not "-1" from resume)
                if job_id != "-1" and not is_job_running(job_id):
                    logger.warning(f"Trial {t_idx} (Job {job_id}) is no longer running and no result file found. Marking as failed.")
                    ax_client.log_trial_failure(trial_index=t_idx)
                    # We treat failed jobs as completed trials to progress the loop, 
                    # unless we want to retry. Ax might suggest a new trial to replace it.
                    completed_trials_count += 1 
                    submitted_trials.pop(i)

        # 2. Submission Phase: Submit new trials if possible
        # We try to keep submitting until we reach total_trial limit
        # Ax will block or error if it needs more data (e.g. Sobol done, need results for BoTorch)
        
        if started_trials_count < run_file.total_trial:
            try:
                # Check if we should wait for data? 
                # AxClient.get_next_trial() might raise exception if generation strategy cannot generate.
                # Or it might just take a long time?
                # We wrap in try block.
                
                # Check max parallelism if needed? For now we trust Ax/User capacity.
                
                parametrization, trial_index = ax_client.get_next_trial()
                
                logger.info(f"#HP_optim--- Submitting trial {started_trials_count}/{run_file.total_trial} (Index {trial_index})")
                
                result_path, job_id = run_trial_sbatch(experiment, parametrization, trial_index, run_file)
                
                submitted_trials.append({
                    "trial_index": trial_index,
                    "result_path": result_path,
                    "job_id": job_id
                })
                started_trials_count += 1
                
            except Exception as e:
                # If Ax cannot generate a trial (e.g. waiting for data), we just log and continue waiting
                # We should be careful not to spam logs if it fails repeatedly.
                # logger.debug(f"Waiting for new trial generation: {e}")
                pass
        
        # 3. Wait before next poll
        # If we have running trials, we wait. If we finished everything, loop terminates.
        if completed_trials_count < run_file.total_trial:
            time.sleep(10)


    logger.info("#HP_optim--- Hyperparam optimization finished")
    best_parameters, values = ax_client.get_best_parameters()
    logger.debug(f"Best parameters {best_parameters}")
    path_bestparam = os.path.join(run_file.out_dir,run_file.out_name+run_file.dataset_name+"_best_hp.json")
    with open(path_bestparam, 'w') as json_file:
        json.dump(best_parameters, json_file, indent=4)
    logger.info(f"Best hyperparameters saved in {path_bestparam}")

    
