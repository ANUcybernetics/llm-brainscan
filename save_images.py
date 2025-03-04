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
