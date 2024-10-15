import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=2,
        latent_dim=2,
        encoder_layer=None,
        decoder_layer=None,
        activation=nn.ELU(),
        distribution="normal",
        batch_invariant=False,
        **kwargs
    ) -> None:
        super(Encoder, self).__init__()

        # Use a larger network for larger input dimensions
        if encoder_layer is None:
            if input_dim > 6500:
                encoder_layer = [4096, 2048, latent_dim]
            else:
                encoder_layer = [2048, latent_dim]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.distribution = distribution
        self.batch_invariant = batch_invariant

        # Encoder
        modules = []
        in_channels = input_dim

        # Build Encoder
        for h_dim in encoder_layer:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        return self.encode(input)
