import numpy as np
import torch


def normalize(arr: np.array, min_val=None, max_val=None):
    if torch.is_tensor(arr):
        if min_val is None:
            min_val = torch.quantile(arr, .01)
        if max_val is None:
            max_val = torch.quantile(arr, .99)
    else:
        if min_val is None:
            min_val = np.quantile(arr, .01)
        if max_val is None:
            max_val = np.quantile(arr, .99)
    return (arr - min_val) / (max_val - min_val + 1e-7)