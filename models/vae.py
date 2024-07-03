import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *
from distributions.von_mises_fisher import VonMisesFisher
from distributions.hyperspherical_uniform import HypersphericalUniform
from torch.distributions.normal import Normal
import numpy as np


class VAE(nn.Module):
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
        super(VAE, self).__init__()

        if encoder_layer is None:
            encoder_layer = [2048, 1024]
        if decoder_layer is None:
            decoder_layer = [1024, 2048]

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

        # Build Latent
        if self.distribution == "normal":
            self.fc_mu = nn.Linear(encoder_layer[-1], latent_dim)
            self.fc_logvar = nn.Linear(encoder_layer[-1], latent_dim)
        else:
            raise ValueError("Unknown distribution")

        # Build Decoder
        modules = []
        in_channels = latent_dim

        for h_dim in decoder_layer:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    self.activation,
                )
            )
            in_channels = h_dim
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, input_dim),
            )
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)

        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recon_input = self.decode(z)
        return recon_input, mu, logvar
