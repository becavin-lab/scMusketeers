# train_trial.py

import json
import argparse
import os

# You will need to import your experiment setup here
# Make sure these imports are available in the sbatch environment
from scmusketeers.arguments.runfile import get_runfile
from scmusketeers.hpoptim.experiment import MakeExperiment

def main():
    parser = argparse.ArgumentParser(description="Run a single Ax trial.")
    parser.add_argument("--trial_index", type=int, required=True, help="Trial index from Ax.")
    parser.add_argument("--params_json", type=str, required=True, help="JSON string of parameters.")
    parser.add_argument("--run_file_path", type=str, required=True, help="Path to the run file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    args = parser.parse_args()

    # Load the runfile and create the experiment object
    run_file = get_runfile(args.run_file_path)
    experiment = MakeExperiment(run_file=run_file, random_seed=40)

    # Decode the parameters
    parameters = json.loads(args.params_json)

    # --- Run the actual training ---
    # This calls your existing training logic with the given params
    metric_dict = experiment.train(parameters)

    # --- Save the result ---
    result_path = os.path.join(args.output_dir, f"result_{args.trial_index}.json")
    with open(result_path, "w") as f:
        json.dump(metric_dict, f)

    print(f"Trial {args.trial_index} finished. Metric: {metric_dict}. Result saved to {result_path}")

if __name__ == "__main__":
    main()