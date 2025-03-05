import numpy as np


def normalize_weights(weight_np):
    """Transform weights by normalizing to [0, 1] range"""
    vmin, vmax = np.min(weight_np), np.max(weight_np)
    normalized = (weight_np - vmin) / (vmax - vmin + 1e-6)
    return normalized

def enhance_contrast(weight_np):
    """Apply contrast enhancement by clipping outliers"""
    percentile=1
    low, high = np.percentile(weight_np, [percentile, 100-percentile])
    clipped = np.clip(weight_np, low, high)
    normalized = (clipped - low) / (high - low + 1e-6)
    return normalized


def softmax_and_normalize_weights(weight_np):
    """Transform weights by normalizing and applying softmax to enhance contrast"""
    temperature=10.0
    # First normalize to [0, 1]
    normalized = normalize_weights(weight_np)

    # Apply temperature scaling and softmax-like transformation
    # Higher temperature = sharper contrast
    scaled = np.exp(temperature * (normalized - 0.5))
    result = scaled / (1.0 + scaled)  # Sigmoid-like squashing to [0, 1]

    return result


def quantize2_weights(weight_np):
    """Transform weights by quantizing to binary values (0 or 1)"""
    # First normalize to [0, 1]
    normalized = normalize_weights(weight_np)

    # Threshold at 0.5 to get binary values
    binary = (normalized > 0.5).astype(float)

    return binary

def quantize3_weights(weight_np):
    """Transform weights by quantizing to three values (0, 0.5, or 1)"""
    # First normalize to [0, 1]
    normalized = normalize_weights(weight_np)

    # Threshold at 1/3 and 2/3 to get ternary values
    result = np.zeros_like(normalized)
    result[normalized > 2/3] = 1.0
    result[(normalized >= 1/3) & (normalized <= 2/3)] = 0.5

    return result


def transform_weights(weight, transform_func=normalize_weights):
    """Apply transformation to weight tensor"""
    # Convert to numpy
    weight_np = weight.float().cpu().numpy()

    # Apply the transformation function
    transformed_data = transform_func(weight_np)

    return transformed_data
