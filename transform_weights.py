import numpy as np


def normalize_weights(weight_np):
    """Transform weights by normalizing to [0, 1] range"""
    vmin, vmax = np.min(weight_np), np.max(weight_np)
    normalized = (weight_np - vmin) / (vmax - vmin + 1e-6)
    return normalized, vmin, vmax


def transform_weights(weight, transform_func=normalize_weights):
    """Apply transformation to weight tensor"""
    # Convert to numpy
    weight_np = weight.float().cpu().numpy()

    # Apply the transformation function
    transformed_data, *metadata = transform_func(weight_np)

    return transformed_data, metadata
