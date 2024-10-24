import os
from PIL import Image
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Import the AttentionModel you defined earlier
from attention_model import AttentionModel  # Make sure to adjust the import path

# Initialize device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attention_model = AttentionModel().to(device)  # Use the AttentionModel
attention_model.eval()  # Set the model to evaluation mode

def load_image(image_path):
    """Load a single image."""
    img = Image.open(image_path).convert("RGB")  # Assuming RGB images
    return img

def get_image_paths(folder_path):
    """Get all image paths in a folder."""
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add image extensions as needed
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

def preprocess_image(img):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((336, 336)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)  # Add batch dimension and send to device

def compute_cosine_similarity(img1, img2):
    """Compute the cosine similarity between two images using embeddings from the model."""
    # Preprocess the images
    img1_tensor = preprocess_image(img1)
    img2_tensor = preprocess_image(img2)

    # Get similarity score from the attention model
    similarity = attention_model(img1_tensor, img2_tensor)

    return similarity.item()  # Return the similarity score as a float

if __name__ == "__main__":
    folder_path = 'CUTE/apple/instance_1/in_the_wild/'  # Path to the folder containing the images

    # Get all image paths in the folder
    image_paths = get_image_paths(folder_path)

    # Ensure we have at least 2 images to compare
    if len(image_paths) < 2:
        print("The folder should contain at least 2 images.")
    else:
        # Load the images
        images = [load_image(img_path) for img_path in image_paths]

        # Iterate over all unique pairs of images and compute cosine similarity
        for (i, j) in combinations(range(len(images)), 2):  # Generate all unique pairs (i, j)
            img1 = images[i]
            img2 = images[j]
            similarity = compute_cosine_similarity(img1, img2)
            print(f"Cosine similarity between image {i+1} and image {j+1}: {similarity:.4f}")
