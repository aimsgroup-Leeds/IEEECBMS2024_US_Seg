# Qualitative visualisation of top 5 and worst 5 images

# Top 5: 78, 74, 75, 76, 66
# Worst 5: 47, 64, 57, 48, 53
# no_backbone - using fold 5, B2 - using fold 4, B7 - using fold 5, PVT - using fold 3 - These cross_val folds are closest to avg val

import os
from PIL import Image, ImageFont, ImageDraw

# Directory containing the image files
image_dir = '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_None_lr0.001'

# Directories containing the overlay and prediction files for different models
model_dirs = [
    # '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/BUSI_None_lr0.01/fold_4',
    # '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/BUSI_efficientnet_b2_lr0.01/fold_3',
    # '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/BUSI_efficientnet_b7_lr0.01/fold_4',
    # '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/BUSI_pvt_v2_b5_lr0.001/fold_2'
    
    '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_None_lr0.001',
    '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_efficientnet_b2_lr0.01',
    '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_efficientnet_b7_lr0.01',
    '/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_pvt_v2_b5_lr0.01'
    
]

# Path to the text file listing filenames
filename_list_file = '/home/scs/Desktop/Eddy/US_Nerve/test.txt'

# Dimensions for resizing
image_width = 384
image_height = 384

# Indices of images
top_indices = [170, 71, 104, 91, 152]
worst_indices = [215, 86, 49, 15, 172]

def read_filenames_from_list(filename_list_file):
    with open(filename_list_file, 'r') as f:
        filenames = [line.strip() for line in f]
    return filenames

def generate_filenames(filenames, index_list):
    result_filenames = []
    for index in index_list:
        filename = filenames[index - 1]  # Adjust for 0-based index
        image_filename = f"{filename}_image.png"
        target_filename = f"{filename}_target.png"
        overlay_filename = f"{filename}_overlay.png"
        prediction_filename = f"{filename}_pred.png"
        result_filenames.append((image_filename, target_filename, overlay_filename, prediction_filename))
    return result_filenames

# Read filenames from the list file
filenames = read_filenames_from_list(filename_list_file)

# Generate filenames for top and worst indices
top_filenames = generate_filenames(filenames, top_indices)
worst_filenames = generate_filenames(filenames, worst_indices)

# Fonts for adding text
font_path = "/home/scs/Desktop/Eddy/arial/arial.ttf"  # Replace with the actual path to a TTF font file
# font_size = 28
font = ImageFont.truetype(font_path)

# Function to create a combined image for a given set of filenames
def create_combined_image_with_text(filenames, column_labels, text_height, scale_factor):
    combined_width = int(image_width * (len(model_dirs) + 2) * scale_factor)
    combined_height = int(image_height * len(filenames) * scale_factor)  # No extra row for text
    text_space_height = int(text_height)  # Height of the space reserved for text
    
    combined_image = Image.new('RGB', (combined_width, combined_height + text_space_height))
    draw = ImageDraw.Draw(combined_image)
       
    for row, (image_filename, target_filename, overlay_filename, prediction_filename) in enumerate(filenames):
        image_path = os.path.join(image_dir, image_filename)
        target_path = os.path.join(image_dir, target_filename)
        image = Image.open(image_path).resize((int(image_width * scale_factor), int(image_height * scale_factor)))
        target = Image.open(target_path).resize((int(image_width * scale_factor), int(image_height * scale_factor)))
        combined_image.paste(image, (0, row * int(image_height * scale_factor)))
        combined_image.paste(target, (int(image_width * scale_factor), row * int(image_height * scale_factor)))
        
        for col, model_dir in enumerate(model_dirs):
            overlay_path = os.path.join(model_dir, overlay_filename)
            overlay = Image.open(overlay_path).resize((int(image_width * scale_factor), int(image_height * scale_factor)))
            combined_image.paste(overlay, ((col + 2) * int(image_width * scale_factor), row * int(image_height * scale_factor)))
    
    # Add text under each column label
    for col, label in enumerate(column_labels):
        text = label
        # text_position = ((col) * int(image_width * scale_factor) + int(image_width * scale_factor) // 2 - int(fixed_font_size * len(text)) // 2,
        #                  combined_height + (text_space_height - int(fixed_font_size)) // 2)  # Center vertically in the text space
        text_position = (
        (col) * int(image_width * scale_factor) + int(image_width * scale_factor) // 2 - font.getsize(label)[0] // 2,
        combined_height + (text_space_height - font.getsize(label)[1]) // 2)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
    
    return combined_image

# Scale factor for resizing the combined image
text_height = 15  # Adjust the text height as needed
scale_factor = 0.15  # Adjust the scale factor as needed

# Column labels for each model
column_labels = [
    "Image",
    "GT",
    "Unet",
    "B2 U-Net",
    "B7 U-Net",
    "PVT U-Net"
]

# Create and display the combined images for top and worst indices with column labels
combined_top_with_text = create_combined_image_with_text(top_filenames, column_labels, text_height, scale_factor)
combined_top_with_text.show()
combined_top_with_text.save('/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_Qual_results_good_with_text.png') 

combined_worst_with_text = create_combined_image_with_text(worst_filenames, column_labels, text_height, scale_factor)
combined_worst_with_text.show()
combined_worst_with_text.save('/home/scs/Desktop/Eddy/MSC-US-Seg/US_Pred_test/US_Nerve_Qual_results_bad_with_text.png')