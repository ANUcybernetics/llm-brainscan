# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "matplotlib", "numpy"]
# ///

import os

from transformers import AutoModelForCausalLM

# Import the image saving function from save_images.py
from save_images import save_weight_image_rawpng

# Import the weight transformation functions
from transform_weights import softmax_and_normalize_weights, transform_weights


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
        shape = param.data.shape
        dim_str = " × ".join(str(dim) for dim in shape)

        if 'mlp' in name.lower() and 'weight' in name.lower():
            print(f"Processing MLP layer: {name} ({dim_str})")
            mlp_counter += 1

            # Transform the weights before saving
            transformed_weights = transform_weights(param.data, softmax_and_normalize_weights)

            # Save the transformed weight matrix as an image
            layer_name = f"mlp_{mlp_counter}_{name}"
            save_weight_image_rawpng(transformed_weights, f"{layer_name}.png", output_dir)
        else:
            # Format dimensions as readable string
            print(f"Skipping visualization for {name} ({dim_str})")

    print(f"Processed {mlp_counter} MLP layers")
    print(f"Done! Check the images in {output_dir} directory")


if __name__ == "__main__":
    main()
