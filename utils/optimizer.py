import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable


class NatMaskOptimizer(Optimizer):
    """Optimizer specilized for updating adversarial face mask"""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        momentum: float = 1.0,
        conv_grad: bool = False,
    ) -> None:
        """
        Args:
            params (list[Tensor]): (N, H, W, C) Batch of uv texture map
            lr (float, optional): Learning rate. Defaults to 1e-3.
            momentum (float, optional): Momentum. Defaults to 1.0.
            conv_grad (bool, optional): Convolution on grad. Defaults to False.
        """
        assert isinstance(params[0], torch.Tensor)
        super().__init__(params, defaults=dict(lr=lr, momentum=momentum))
        self.conv_grad = conv_grad
        self.kernel = get_kernel().to(params[0].device)

    @torch.no_grad()
    def step(self, closure: Callable = None) -> None:
        """Performs a single optimization step"""
        for group in self.param_groups:
            for p in group["params"]:
                # Boosting adversarial attacks with momentum
                # Paper: https://arxiv.org/abs/1710.06081
                if p not in self.state:
                    self.state[p] = dict(mom=torch.zeros_like(p.grad))
                grad = p.grad
                if self.conv_grad:
                    grad = self._conv_grad(p.grad)
                accumulated_grad = self.state[p]["mom"]
                accumulated_grad = group["momentum"] * accumulated_grad + grad
                self.state[p]["mom"] = accumulated_grad
                # update param
                p.data = p.data - group["lr"] * accumulated_grad.sign()

    def _conv_grad(self, grad: Tensor) -> Tensor:
        """
        # Paper: https://arxiv.org/abs/1904.02884
        # Based on: https://github.com/dongyp13/Translation-Invariant-Attacks
        Args:
            grad (Tensor): (N, H, W, C) Batch of gradients with respect to images
        """
        grad = torch.permute(grad, (0, 3, 1, 2))
        grad = F.conv2d(grad, self.kernel, groups=3, padding="same")
        grad = grad / torch.std(grad, dim=[1, 2, 3], keepdim=True)
        grad = torch.permute(grad, (0, 2, 3, 1))
        return grad


def get_kernel(kernlen: int = 15, nsig: int = 3) -> Tensor:
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    kernel = torch.from_numpy(stack_kernel)

    return kernel


if __name__ == "__main__":
    optimizer = NatMaskOptimizer([torch.randn(4, 4)])
    print(optimizer)
