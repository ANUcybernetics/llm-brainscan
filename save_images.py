import os

import matplotlib.pyplot as plt


def save_weight_image_matplot(weight_data, filename, directory='weight_images'):
    """
    Save weight data as an image using matplotlib

    Args:
        weight_data: The weight data to visualize and save
        filename: Name for the saved file
        directory: Directory to save the image in (default: 'weight_images')
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Create figure for the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(weight_data, cmap='viridis')
    plt.colorbar()
    plt.title(f'Weight Visualization: {filename}')
    plt.tight_layout()

    # Save the figure
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close()

    print(f"Weight image saved to {full_path}")

def save_weight_image_rawpng(weight_data, filename, directory='weight_images_raw'):
    """
    Save weight data as a raw grayscale PNG file

    Args:
        weight_data: The weight data to save as an image
        filename: Name for the saved file
        directory: Directory to save the image in (default: 'weight_images_raw')
    """
    import numpy as np
    from PIL import Image

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Normalize data to 0-255 range for grayscale image
    normalized_data = (weight_data - weight_data.min()) / (weight_data.max() - weight_data.min())
    img_data = (normalized_data * 255).astype(np.uint8)

    # Create image and save
    img = Image.fromarray(img_data, mode='L')  # 'L' mode is for grayscale

    # Save the image
    full_path = os.path.join(directory, filename)
    img.save(full_path)

    print(f"Raw weight image saved to {full_path}")
