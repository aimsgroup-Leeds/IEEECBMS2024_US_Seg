import os
import numpy as np
import shutil
import random
import csv

Dataset = 'Nerve'

if Dataset == 'BUSI':
    # root_dir = '/home/scs/Desktop/Eddy/BUSI/img'
    root_dir = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\BUSI\img'
elif Dataset == 'Nerve':
    # root_dir = '/home/scs/Desktop/Eddy/BUSI/Nerve/train'
    root_dir = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\Nerve\img'  # splitting all images in train folder as mask ground truth not given for test folder

else:
    ValueError('Dataset choice not recognised')

train_path = 'Nerve_train.txt'
val_path = 'Nerve_val.txt'
test_path = 'Nerve_test.txt'

# val_ratio = 0.8
# test_ratio = 0.9

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

allFileNames = os.listdir(root_dir)

# Split each category proportionally
train_ratio = 0.8
val_ratio = 0.1

## BUSI
if Dataset == 'BUSI':
    # Separate filenames based on categories (benign, malignant, normal)
    benign_files = [filename for filename in allFileNames if 'benign' in filename]
    malignant_files = [filename for filename in allFileNames if 'malignant' in filename]
    normal_files = [filename for filename in allFileNames if 'normal' in filename]

    # Shuffle each category separately
    random.shuffle(benign_files)
    random.shuffle(malignant_files)
    random.shuffle(normal_files)

    benign_train = benign_files[:int(len(benign_files) * train_ratio)]
    benign_val = benign_files[int(len(benign_files) * train_ratio):int(len(benign_files) * (train_ratio + val_ratio))]
    benign_test = benign_files[int(len(benign_files) * (train_ratio + val_ratio)):]

    malignant_train = malignant_files[:int(len(malignant_files) * train_ratio)]
    malignant_val = malignant_files[
                    int(len(malignant_files) * train_ratio):int(len(malignant_files) * (train_ratio + val_ratio))]
    malignant_test = malignant_files[int(len(malignant_files) * (train_ratio + val_ratio)):]

    normal_train = normal_files[:int(len(normal_files) * train_ratio)]
    normal_val = normal_files[int(len(normal_files) * train_ratio):int(len(normal_files) * (train_ratio + val_ratio))]
    normal_test = normal_files[int(len(normal_files) * (train_ratio + val_ratio)):]

    # Combine the splits of each category
    train_FileNames = np.concatenate((benign_train, malignant_train, normal_train))
    val_FileNames = np.concatenate((benign_val, malignant_val, normal_val))
    test_FileNames = np.concatenate((benign_test, malignant_test, normal_test))

    print(len(allFileNames), len(train_FileNames), len(val_FileNames), len(test_FileNames))
    #

## Nerve
elif Dataset == 'Nerve':

    BP_files = []  # BP stands for the files where the Brachial Plexus is present and therefore shown in mask
    No_BP_files = []

    csv_filename = 'train_masks.csv'
    csv_file = os.path.join(os.path.dirname(root_dir), csv_filename)

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            img = row['img']
            subject = row['subject']
            pixels = row['pixels']

            combined_value = f"{subject}_{img}"

            if combined_value + ".tif" in allFileNames:  # Assuming filenames include extensions
                if not pixels:
                    No_BP_files.append(combined_value)
                else:
                    BP_files.append(combined_value)

    print("Images with masks:")
    print(len(BP_files))

    print("\nImages without masks:")
    print(len(No_BP_files))

    random.shuffle(BP_files)
    random.shuffle(No_BP_files)

    BP_files_train = BP_files[:int(len(BP_files) * train_ratio)]
    BP_files_val = BP_files[int(len(BP_files) * train_ratio):int(len(BP_files) * (train_ratio + val_ratio))]
    BP_files_test = BP_files[int(len(BP_files) * (train_ratio + val_ratio)):]

    No_BP_files_train = No_BP_files[:int(len(No_BP_files) * train_ratio)]
    No_BP_files_val = No_BP_files[int(len(No_BP_files) * train_ratio):int(len(No_BP_files) * (train_ratio + val_ratio))]
    No_BP_files_test = No_BP_files[int(len(No_BP_files) * (train_ratio + val_ratio)):]

    # Combine the splits of each category
    train_FileNames = np.concatenate((BP_files_train, No_BP_files_train))
    val_FileNames = np.concatenate((BP_files_val, No_BP_files_val))
    test_FileNames = np.concatenate((BP_files_test, No_BP_files_test))

    print(len(allFileNames), len(train_FileNames), len(val_FileNames), len(test_FileNames))

with open(train_path, 'w') as train_file:
    for line in train_FileNames:
        line = line.split('.')[0]
        train_file.write(line)
        train_file.write('\n')

with open(val_path, 'w') as val_file:
    for line in val_FileNames:
        line = line.split('.')[0]
        val_file.write(line)
        val_file.write('\n')

with open(test_path, 'w') as test_file:
    for line in test_FileNames:
        line = line.split('.')[0]
        test_file.write(line)
        test_file.write('\n')
