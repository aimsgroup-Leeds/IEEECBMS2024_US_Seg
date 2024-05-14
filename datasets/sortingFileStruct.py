# This file is used as standalone to organise file structure of BUSI data following download from:
# https://academictorrents.com/details/d0b7b7ae40610bbeaea385aeb51658f527c86a16


import os
import zipfile
import shutil
from PIL import Image, ImageChops


def extract_zip_file(zip_file_path, extract_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder_path)

    # Get the list of extracted file names
    extracted_files = zip_ref.namelist()

    # Get the name of the first folder inside the zip file
    first_folder_name = extracted_files[0].split('/')[0]

    extracted_folder_path = os.path.join(extract_folder_path, first_folder_name)

    return extracted_folder_path


def restructuring_data(extract_folder_path, new_path_location):
    img_folder = os.path.join(new_path_location, 'img')
    mask_folder = os.path.join(new_path_location, 'mask_ground_truth')

    # Create the img and mask_ground_truth folders if they don't exist
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Traverse through all files and subdirectories in the extract_folder_path
    for root, dirs, files in os.walk(extract_folder_path):
        for file_name in files:
            if file_name.endswith('.png') or file_name.endswith('.tif'):
                if 'mask' in file_name:
                    # Move files with 'mask' in their filename to the mask_ground_truth folder
                    file_path = os.path.join(root, file_name)
                    new_file_path = os.path.join(mask_folder, file_name)
                    shutil.copy(file_path, new_file_path)
                else:
                    # Move other files to the img folder
                    file_path = os.path.join(root, file_name)
                    new_file_path = os.path.join(img_folder, file_name)
                    shutil.copy(file_path, new_file_path)


def Multiple_Masks(mask_folder, img_folder):
    img_filenames = os.listdir(img_folder)

    mask_files = {}
    parent_folder = os.path.dirname(mask_folder)
    combined_mask_dir = parent_folder + r'\combined_maskGT'

    for filename in os.listdir(mask_folder):
        mask_filename = filename.split('_mask')[0] + '.png'
        if mask_filename in img_filenames:
            if mask_filename in mask_files:
                mask_files[mask_filename].append(filename)
            else:
                mask_files[mask_filename] = [filename]

    for mask_filename, masks_list in mask_files.items():
        combined_mask = None

        for file in masks_list:
            mask_path = os.path.join(mask_folder, file)
            mask_img = Image.open(mask_path).convert("RGB").resize((256, 256), resample=0)

            if combined_mask is None:
                combined_mask = mask_img
            else:
                combined_mask = ImageChops.add(combined_mask, mask_img)

        if not os.path.exists(combined_mask_dir):
            os.mkdir(combined_mask_dir)
        combined_mask.save(os.path.join(combined_mask_dir, mask_filename))


# Block 1 - used to format data structure
Zip_file = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\ultrasound-nerve-segmentation.zip'
Dataset_location = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\Nerve\train'
## New_file_location = r'C:\Users\ekael\Mirror\PhD\Project\Datasets\Nerve'
## restructuring_data(Dataset_location, New_file_location)

Extracted_file = extract_zip_file(Zip_file, Dataset_location)
restructuring_data(Extracted_file, Dataset_location)
# end Block 1

# # Block 2 - Used to combine masks, if there are multiple masks per image
# img_folder = '/home/scs/Desktop/Eddy/BUSI/img'
# mask_folder = '/home/scs/Desktop/Eddy/BUSI/mask_ground_truth'
#
# Multiple_Masks(mask_folder, img_folder)


