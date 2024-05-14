import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# Define the root directory and cross_val.txt file
root = os.getcwd()  # Change this to the appropriate directory
cross_val_file = os.path.join(root, 'cross_val.txt')

# Read the cross_val.txt file and count the number of files for each class
class_counts = defaultdict(int)
with open(cross_val_file, "r") as f:
    for line in f:
        class_label = line.strip().split()[0]  # Assuming class is the first word in each line
        class_counts[class_label] += 1

# Load all file names from cross_val.txt
with open(cross_val_file, "r") as f:
    file_names = [x.strip() for x in f.readlines()]

# Create StratifiedKFold object
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Determine the maximum number of validation files per fold
max_val_files = min(78, min(class_counts.values()))

# Save each fold's train and validation files as separate text files
for fold_idx, (train_indices, val_indices) in enumerate(skf.split(file_names, [class_counts[file_name.split()[0]] for file_name in file_names])):
    fold_dir = os.path.join(root, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Shuffle the validation indices and select up to max_val_files indices
    np.random.seed(42)  # Set random seed for reproducibility in shuffle (*******)
    np.random.shuffle(val_indices)
    val_indices = val_indices[:max_val_files]
    
    train_file = os.path.join(fold_dir, f'fold_{fold_idx + 1}_train.txt')
    with open(train_file, 'w') as f:
        for idx in train_indices:
            f.write(file_names[idx] + '\n')
            
    val_file = os.path.join(fold_dir, f'fold_{fold_idx + 1}_val.txt')
    with open(val_file, 'w') as f:
        for idx in val_indices:
            f.write(file_names[idx] + '\n')
