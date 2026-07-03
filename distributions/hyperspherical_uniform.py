"""
Code directly taken from https://github.com/nicola-decao/s-vae-pytorch/
"""

import math
import torch


class HypersphericalUniform(torch.distributions.Distribution):
    """Uniform distribution over the surface of a `dim`-dimensional hypersphere."""

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        """
        Args:
            dim (int): Dimensionality of the ambient space; samples live on the
                `dim`-sphere embedded in `dim + 1` dimensions.
            validate_args (bool, optional): Passed to `torch.distributions.Distribution`.
            device (str or torch.device, optional): Device for sampling.
        """
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        """Draws samples by normalizing standard Gaussian vectors in `dim + 1` dimensions."""
        output = (
            torch.distributions.Normal(0, 1)
            .sample(
                (shape if isinstance(shape, torch.Size) else torch.Size([shape]))
                + torch.Size([self._dim + 1])
            )
            .to(self.device)
        )

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        """Returns the (constant) differential entropy, equal to the log surface area."""
        return self.__log_surface_area()

    def log_prob(self, x):
        """Returns the (constant) log density at `x`, equal to `-log_surface_area`."""
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        """Log surface area of the `dim`-sphere."""
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2], device=self.device)
            )
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma
