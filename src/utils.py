import numpy as np

def normalize_data(series):
    mean = np.mean(series)
    std = np.std(series)
    normalized = (series - mean) / std
    return normalized, {"mean": mean, "std": std}
