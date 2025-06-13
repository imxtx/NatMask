import torch
import torch.nn.functional as F
from torch import Tensor


def dodging_loss(f1: Tensor, f2: Tensor, single_esb: bool = False) -> Tensor:
    """Computes dodging loss as the mean cosine similarity between features.
    Args:
        f1: First feature tensor
        f2: Second feature tensor
    Returns:
        Mean cosine similarity between f1 and f2
    """
    if single_esb:
        return F.cosine_similarity(f1, f2)

    return torch.mean(F.cosine_similarity(f1, f2))


def impersonation_loss(f1: Tensor, f2: Tensor, single_esb: bool = False) -> Tensor:
    """Computes impersonation loss as 1 minus the mean cosine similarity.
    Args:
        f1: First feature tensor
        f2: Second feature tensor
    Returns:
        1 minus the mean cosine similarity between f1 and f2
    """
    if single_esb:
        return 1 - F.cosine_similarity(f1, f2)

    return torch.mean(1 - F.cosine_similarity(f1, f2))


def tv_loss(x: Tensor, weight: float = 1) -> Tensor:
    """Computes total variation loss for image smoothness.

    Args:
        x: Input tensor of shape [batch_size, channels, height, width]
        weight: Weight of TV loss
    """
    batch_size, c, h, w = x.size()
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return weight * (tv_h + tv_w) / batch_size / (w * h * c)
