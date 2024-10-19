import os
from PIL import Image
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from foreground_feature_averaging5 import ForegroundFeatureAveraging  # Import the updated model with MiDaS

# Initialize device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
foreground_model = ForegroundFeatureAveraging(device=device)  # Use the updated model with MiDaS
foreground_model.eval()  # Set the model to evaluation mode

def load_image(image_path):
    """Load a single image."""
    img = Image.open(image_path).convert("RGBA")  # Assuming RGBA images
    return img

def get_image_paths(folder_path):
    """Get all image paths in a folder."""
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add image extensions as needed
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

def compute_cosine_similarity(img1, img2):
    """Compute the cosine similarity between two images using embeddings from the model."""
    # Get feature vectors using the ForegroundFeatureAveraging model (with depth)
    features_img1 = foreground_model('Crop-Feat', [img1])  # Pass a list with the image
    features_img2 = foreground_model('Crop-Feat', [img2])  # Pass a list with the image

    # Extract feature vectors (assuming they are 1D tensors)
    features_img1 = features_img1.cpu().numpy().flatten()
    features_img2 = features_img2.cpu().numpy().flatten()

    # Compute cosine similarity
    similarity = cosine_similarity([features_img1], [features_img2])
    return similarity[0][0]

if __name__ == "__main__":
    folder_path = 'CUTE/apple/instance_1/in_the_wild/'  # Path to the folder containing the 4 images

    # Get all image paths in the folder
    image_paths = get_image_paths(folder_path)
    
    # Ensure we have exactly 4 images
    if len(image_paths) != 4:
        print("The folder should contain exactly 4 images.")
    else:
        # Load the images
        images = [load_image(img_path) for img_path in image_paths]

        # Iterate over all unique pairs of images and compute cosine similarity
        for (i, j) in combinations(range(4), 2):  # Generate all unique pairs (i, j)
            img1 = images[i]
            img2 = images[j]
            similarity = compute_cosine_similarity(img1, img2)
            print(f"Cosine similarity between image {i+1} and image {j+1}: {similarity:.4f}")
