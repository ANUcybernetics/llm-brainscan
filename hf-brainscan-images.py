# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "matplotlib", "numpy"]
# ///

import os

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM


def save_weight_image(weight, name, output_dir):
    """Save a tensor as a normalized image"""
    # Convert to numpy and normalize to [0, 1]
    weight_np = weight.float().cpu().numpy()

    # Normalize the weights
    vmin, vmax = np.min(weight_np), np.max(weight_np)
    normalized = (weight_np - vmin) / (vmax - vmin + 1e-6)

    # Create the figure with no axes and save
    plt.figure(figsize=(10, 10))
    plt.imshow(normalized, cmap='viridis')
    plt.colorbar(label="Weight Value")
    plt.title(f"{name} - min: {vmin:.4f}, max: {vmax:.4f}")
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    output_path = os.path.join(output_dir, f"{name.replace('/', '_').replace('.', '_')}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    # Directory to save the images
    output_dir = "mlp_weight_images"
    os.makedirs(output_dir, exist_ok=True)

    # Use TinyStories-33M model
    model_name = "roneneldan/TinyStories-33M"
    print(f"Loading model from {model_name}")

    # Load the model directly with transformers
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Process MLP layers
    mlp_counter = 0
    for name, param in model.named_parameters():
        if 'mlp' in name.lower() and 'weight' in name.lower():
            print(f"Processing MLP layer: {name}")
            mlp_counter += 1

            # Save the weight matrix as an image
            save_weight_image(param.data, f"mlp_{mlp_counter}_{name}", output_dir)

    print(f"Processed {mlp_counter} MLP layers")
    print(f"Done! Check the images in {output_dir} directory")


if __name__ == "__main__":
    main()
