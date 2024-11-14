import os
import torch
from PIL import Image
from itertools import combinations
from foreground_feature_averaging3 import ForegroundFeatureAveraging  # Import the model class

# Path to the folder containing the images
folder_path = "CUTE/apple/instance_1/in_the_wild"

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForegroundFeatureAveraging(device)

# Load all images from the folder
image_list = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)
        image_list.append(img)

# Ensure at least 2 images are loaded
if len(image_list) < 2:
    raise ValueError("The folder must contain at least two images.")

# Compute cosine similarity between all pairs of images
image_pairs = list(combinations(range(len(image_list)), 2))  # Get all pairs (combinations) of image indices

for idx1, idx2 in image_pairs:
    img1 = image_list[idx1]
    img2 = image_list[idx2]

    # Forward pass through the model to compute cosine similarity between the two images
    similarity = model("Crop-Feat", [img1], [img2])
    
    print(f"Cosine similarity between image {idx1 + 1} and image {idx2 + 1}: {similarity:.4f}")
