import os
import json
import pandas as pd
import argparse
import sys

def compile_results(results_dir, output_file):
    print(f"Searching for results in {results_dir}...")
    
    data = []
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    # Walk through directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_best_hp.json"):
                full_path = os.path.join(root, file)
                print(f"Found: {file}")
                
                try:
                    with open(full_path, 'r') as f:
                        params = json.load(f)
                    
                    # Add dataset name (inferred from filename or dir)
                    # filename format: [out_name][dataset_name]_best_hp.json
                    # We can use the dirname as dataset name if structure is results/dataset_name/
                    dataset_name = os.path.basename(root)
                    
                    entry = {"dataset": dataset_name}
                    entry.update(params)
                    data.append(entry)
                    
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    
    # Reorder columns to put dataset first
    cols = ["dataset"] + [c for c in df.columns if c != "dataset"]
    df = df[cols]
    
    print("\nCompiled Results:")
    print(df)
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved compiled results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile best hyperparameters.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing results")
    parser.add_argument("--output_file", type=str, default="best_hyperparameters_summary.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    compile_results(args.results_dir, args.output_file)
