#!/usr/bin/env python3

import os
import csv

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(base_dir), "results")
    output_dir = os.path.join(base_dir, "paper_review")
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "task1_expdesign.csv")
    
    results = []
    
    if os.path.isdir(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            
            if os.path.isdir(item_path) and "task_1" in item:
                run_id = item
                
                parts = item.split("_task_1_")
                dataset = parts[0] if len(parts) > 1 else ""
                
                rest = parts[1] if len(parts) > 1 else ""
                
                fold1 = ""
                fold2 = ""
                model = ""
                
                if rest:
                    rest_parts = rest.split("_")
                    # Check if the last two parts are digits
                    if len(rest_parts) >= 2 and rest_parts[-1].isdigit() and rest_parts[-2].isdigit():
                        fold1 = rest_parts[-2]
                        fold2 = rest_parts[-1]
                        model = "_".join(rest_parts[:-2])
                    elif len(rest_parts) >= 1 and rest_parts[-1].isdigit():
                        fold1 = rest_parts[-1]
                        model = "_".join(rest_parts[:-1])
                    else:
                        model = rest
                        
                results.append({
                    "Folder_Name": item,
                    "Run_ID": run_id,
                    "Dataset": dataset,
                    "Model": model,
                    "Fold1": fold1,
                    "Fold2": fold2
                })
    else:
        print(f"Results directory not found at {results_dir}")
        return
            
    # Sort results by Run_ID
    results = sorted(results, key=lambda x: x["Run_ID"])
            
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Folder_Name", "Run_ID", "Dataset", "Model", "Fold1", "Fold2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Extracted {len(results)} folder names and saved to {output_csv}")

if __name__ == "__main__":
    main()
