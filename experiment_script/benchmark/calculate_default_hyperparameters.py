import csv
import os

# Paths
base_dir = "/workspace/cell/scMusketeers/experiment_script"
best_params_path = os.path.join(base_dir, "hyperparam/all_datasets_best_hyperparameters.csv")
old_params_path = os.path.join(base_dir, "hyperparam/default_df_t10.csv")
new_params_path = os.path.join(base_dir, "hyperparam/default_df_t11.csv")
comparison_path = os.path.join(base_dir, "hyperparam/comparison_t11_vs_t10.txt")

# Numeric columns to average
numeric_cols = [
    'use_hvg', 'batch_size', 'clas_w', 'rec_w', 'dann_w', 
    'learning_rate', 'weight_decay', 'dropout', 
    'bottleneck', 'layer2', 'layer1'
]

# Columns that must be integers
int_cols = ['use_hvg', 'batch_size', 'bottleneck', 'layer1', 'layer2']

# Read best params
best_params = []
with open(best_params_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        best_params.append(row)

# Calculate means
means = {}
for col in numeric_cols:
    values = []
    for row in best_params:
        if col in row and row[col]:
            try:
                values.append(float(row[col]))
            except ValueError:
                pass
    if values:
        mean_val = sum(values) / len(values)
        if col in int_cols:
            means[col] = int(round(mean_val))
        else:
            means[col] = mean_val
    else:
        means[col] = 0

print("New Average Parameters:")
for col, val in means.items():
    print(f"{col}: {val}")

# Read old params
old_params = []
old_fieldnames = []
with open(old_params_path, 'r') as f:
    reader = csv.DictReader(f)
    old_fieldnames = reader.fieldnames
    for row in reader:
        old_params.append(row)

# Comparison
comparison_lines = []
comparison_lines.append(f"{'Parameter':<20} | {'New (t11)':<15} | {'Old (t10)':<15} | {'Difference':<15}")
comparison_lines.append("-" * 75)

print("\nComparison (New vs Old t10):")
# Assuming first row is the target
target_row = old_params[0]
for col in numeric_cols:
    if col in target_row:
        old_val_str = target_row[col]
        try:
             old_val = float(old_val_str)
             new_val = means.get(col, 0)
             diff = new_val - old_val
             
             if col in int_cols:
                 line = f"{col}: {new_val} vs {int(old_val)} ({int(diff):+d})"
                 table_line = f"{col:<20} | {new_val:<15} | {int(old_val):<15} | {int(diff):+d}"
             else:
                 line = f"{col}: {new_val:.6f} vs {old_val} ({diff:+.4f})"
                 table_line = f"{col:<20} | {new_val:<15.6f} | {old_val:<15} | {diff:+.4f}"
             
             print(line)
             comparison_lines.append(table_line)
        except ValueError:
             print(f"{col}: {means.get(col, 0)} vs {old_val_str}")
             comparison_lines.append(f"{col:<20} | {means.get(col, 0):<15} | {old_val_str:<15} | N/A")
    else:
        print(f"{col}: {means.get(col, 0)} vs N/A")
        comparison_lines.append(f"{col:<20} | {means.get(col, 0):<15} | N/A             | N/A")

# Create new params
new_param_row = target_row.copy()
for col in numeric_cols:
    if col in means:
        val = means[col]
        if col in int_cols:
            val = int(val) 
        new_param_row[col] = str(val)

# Write to new params file
with open(new_params_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=old_fieldnames)
    writer.writeheader()
    writer.writerow(new_param_row)

print(f"\nSaved new default parameters to {new_params_path}")

# Write comparison to file
with open(comparison_path, 'w') as f:
    for line in comparison_lines:
        f.write(line + "\n")

print(f"Saved comparison table to {comparison_path}")
