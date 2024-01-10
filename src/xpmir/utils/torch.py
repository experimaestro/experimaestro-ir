import torch
from typing import Optional


def to_device(tensor: Optional[torch.Tensor], device: torch.device):
    """Move to device if not None"""
    if tensor is not None:
        return tensor.to(device)
    return tensor
