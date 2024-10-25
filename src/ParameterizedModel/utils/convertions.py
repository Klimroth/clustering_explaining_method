import numpy as np
import torch

def to_torch(x):
    if torch.isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)