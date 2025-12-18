import sys
import os
import argparse
import json
from unittest.mock import MagicMock, patch

# Add workspace to path
sys.path.append("/workspace/cell/scMusketeers")

# Import the module to test
from scmusketeers.hpoptim.run_workflow import run_trial_sbatch

def test_sbatch_generation():
    print("Testing sbatch generation and run_trial_sbatch unit...")
    
    # Note: run_workflow logic has changed to loop dynamically.
    # We are testing run_trial_sbatch which is the core unit.

    # Mock run_file
    run_file = argparse.Namespace(
        out_dir="./test_output",
        out_name="test_run",
        dataset_name="test_dataset",
        total_trial=1,
        hparam_path=None
    )
    
    # Mock params
    params = {"learning_rate": 0.001, "batch_size": 32}
    trial_index = 0
    experiment = MagicMock()
    
    # Mock sys.argv
    sys.argv = ["sc-musketeers", "hp_optim", "--ref_path", "data.h5ad", "--other_arg", "value"]
    
    # Mock subprocess.run to avoid actual submission
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Submitted batch job 12345"
        
        # We also need to mock os.path.exists and open for the result file loop
        # But run_trial_sbatch loops until result exists.
        # We should patch time.sleep to throw an exception or something to break the loop,
        # OR just check that it submits and then writes the script.
        # We can't easily break the loop without modifying code or mocking exists to return True immediately.
        
        
        # Test 1: hp_resume=False (Default)
        # Result file shouldn't exist ideally, or if it does, it should be deleted.
        # But here we assume clean slate or we just check normal submission.
        
        # Test 1: hp_resume=False (Normal)
        
        print("\n--- Test 1: hp_resume=False (Normal) ---")
        run_file.hp_resume = False
        res_path, job_id = run_trial_sbatch(experiment, params, trial_index, run_file)
        print(f"Returned Job ID: {job_id}")
        if os.path.exists("./test_output/hp_trials/test_dataset_hp_trial_0.sh"):
             print("SUCCESS: Script generated (Normal)")
        else:
             print("FAILURE: Script not generated (Normal)")

        # Test 2: hp_resume=True with existing result
        print("\n--- Test 2: hp_resume=True (Skip) ---")
        run_file.hp_resume = True
        # Create dummy result
        dummy_result = "./test_output/hp_trials/trial_0_result.json"
        with open(dummy_result, "w") as f:
            json.dump({"opt_metric": 0.88}, f)
            
        script_path = "./test_output/hp_trials/test_dataset_hp_trial_0.sh"
        # Clear script to verify it's NOT regenerated
        if os.path.exists(script_path):
            os.remove(script_path)
            
        # Mocking return value for skip case logic? running run_trial_sbatch...
        # Wait, run_trial_sbatch returns just path if skipped? No, I need to check implementation.
        # Implementation returns "return result_path" if skipped. But "return result_path, job_id" at end.
        
        # Checking implementation of run_trial_sbatch again...
        # In my last edit:
        # if run_file.hp_resume: ... return result_path
        # Oh! I forgot to return job_id in the early return!
        # It needs to return a tuple there too or unpack will fail in run_workflow.
        
        # I should fix run_workflow.py first if bug exists.
        # But let's finish updating this script assuming I fix it to return (path, "-1") or ("path", "SKIPPED").
        
        try:
             res = run_trial_sbatch(experiment, params, trial_index, run_file)
             if isinstance(res, tuple):
                 print(f"SUCCESS: Returned tuple {res}")
             else:
                 print(f"FAILURE: Returned single value {res} (Expected tuple)")
        except Exception as e:
             print(f"FAILURE: Exception {e}")
        
        
        if os.path.exists(script_path):
            print("FAILURE: Script generated despite hp_resume=True")
        else:
            print("SUCCESS: Script skipped (hp_resume=True)")



if __name__ == "__main__":
    test_sbatch_generation()
