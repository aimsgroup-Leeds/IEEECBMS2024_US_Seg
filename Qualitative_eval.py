# Qualitative eval script

# Test_set_files

# Finding top 10 and bottom 10 images from cross_val_files

import os
import re
import heapq
import pandas as pd

# Directory containing the subdirectories

cross_val = False
dataset = 'US_Nerve'

base_dir = '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_pvt_v2_b5_lr0.01'

# Function to extract values from metrics.txt file
def extract_values(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        values_started = False
        values = []
        for line in lines:
            if 'Seg_Performance Image_indices_ranked' in line:
                values_started = True
                line_values = re.findall(r'\d+', line)
                values.extend(map(int, line_values))
            elif values_started and 'Average' in line:
                break
            elif values_started:
                line_values = re.findall(r'\d+', line)
                values.extend(map(int, line_values))
    return values

values_dict = {}

if cross_val:
    # Iterate through subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            metrics_file_path = os.path.join(subdir_path, 'Metrics.txt')
            if os.path.exists(metrics_file_path):
                values = extract_values(metrics_file_path)
                values_dict[subdir] = values
                
else:
    # Search for "Metrics.txt" files directly in the base directory
    for file_name in os.listdir(base_dir):
        if file_name == 'Metrics.txt':
            metrics_file_path = os.path.join(base_dir, file_name)
            values = extract_values(metrics_file_path)
            values_dict[base_dir] = values

if dataset == 'BUSI':
    # Remove indices 66 to 79 from all the lists in the values_dict
    for key, values in values_dict.items():
        values_dict[key] = [value for value in values if value < 66 or value > 79]
        
elif dataset == 'US_Nerve':
    
    masked_test_set_indices = []
    
    # Replace 'path_to_reference_sheet.xlsx' with the actual path to your reference Excel sheet
    reference_sheet_path = '/home/scs/Desktop/Eddy/US_Nerve/train_masks.csv'
    reference_data = pd.read_csv(reference_sheet_path)
    
    # Read the test.txt file
    test_txt_path = '/home/scs/Desktop/Eddy/US_Nerve/test.txt'
    test_entries = []
    with open(test_txt_path, 'r') as test_file:
        test_entries = test_file.read().splitlines()
        
    for i, entry in enumerate(test_entries):
        subject, img = entry.split('_')
        filtered_data = reference_data[(reference_data['subject'] == int(subject)) & (reference_data['img'] == int(img))]
        if not filtered_data.empty and not filtered_data['pixels'].isnull().all():
            masked_test_set_indices.append(i)

    # Update values_dict with valid test entries
    for key, values in values_dict.items():
        values_dict[key] = [value for value in values if value in masked_test_set_indices]

  
# Dictionary to store top 5 indices for each key
# top_indices_dict = {}

# Create a list of tuples where each tuple contains the last 5 values from each list
last_10_values = [(key, values[-10:]) for key, values in values_dict.items()]

# Create a dictionary to store the total count of each value across all lists
value_counts = {}

# Count the occurrences of each value
for _, values in last_10_values:
    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1

# Find the 5 values with the highest total counts
top_values = heapq.nlargest(10, value_counts.keys(), key=value_counts.get)

print("Top 10 performing File indices:")
for value in top_values:
    print(value)
    
# Create a list of tuples where each tuple contains the first 5 values from each list
first_10_values = [(key, values[:10]) for key, values in values_dict.items()]

# Create a dictionary to store the total count of each value across all lists
value_counts = {}

# Count the occurrences of each value
for _, values in first_10_values:
    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1

# Find the 5 values with the highest total counts
top_values_start = heapq.nlargest(10, value_counts.keys(), key=value_counts.get)

print("Worst 10 performing File indices:")
for value in top_values_start:
    print(value)


