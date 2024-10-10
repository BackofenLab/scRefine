import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial


class VAE_ZINB(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_layer=None,
        decoder_layer=None,
        distribution="zinb",
        activation=nn.Softplus(),
    ):
        super(VAE_ZINB, self).__init__()

        # Use a larger network for larger input dimensions
        if encoder_layer is None:
            if input_dim > 6500:
                encoder_layer = [4096, 2048, 1024]
            else:
                encoder_layer = [2048, 1024]
        if decoder_layer is None:
            if input_dim > 6500:
                decoder_layer = [1024, 2048, 4096]
            else:
                decoder_layer = [1024, 2048]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.distribution = distribution

        # Encoder
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

        self.fc_mu = nn.Linear(encoder_layer[-1], latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(
            encoder_layer[-1], latent_dim
        )  # Log variance of latent space

        # Decoder
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

        # Outputs for ZINB distribution
        self.decoder_mu = nn.Linear(input_dim, input_dim)  # Mean (mu)
        self.decoder_drop_logits = nn.Linear(
            input_dim, input_dim
        )  # Inverse dispersion (theta)

        self.log_theta = torch.nn.Parameter(torch.randn(input_dim))

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)

        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        hidden_dec = self.decoder(z)
        mean = torch.exp(self.decoder_mu(hidden_dec))  # Positive values
        dropout_logits = self.decoder_drop_logits(hidden_dec)
        return mean, dropout_logits

    def reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        input = torch.log(1 + input)

        # Encode
        mu, logvar = self.encode(input)

        # Sample from latent space
        z = self.reparameterize(mu, logvar)

        # Decode
        de_mean, de_dropout = self.decode(z)
        return de_mean, de_dropout, mu, logvar

    def loss_function(self, x, mean, dropout_logits, mu, logvar, kld_weight=0.00025):
        # Recon loss
        recon_loss = self.recon_loss(x, mean, dropout_logits)

        # KL Divergence
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = -recon_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recon_loss": -recon_loss.detach(),
            "kld_loss": -kld_loss.detach(),
        }

    def recon_loss(self, x, mu, dropout_logits):
        """
        x: input data
        mu: output of decoder
        dropout_logits: dropout logits of zinb distribution
        """
        theta = self.log_theta.exp()

        nb_logits = (mu + 1e-5).log() - (theta + 1e-5).log()

        if self.distribution == "zinb":
            distribution = ZeroInflatedNegativeBinomial(
                total_count=theta,
                logits=nb_logits,
                gate_logits=dropout_logits,
                validate_args=False,
            )
        elif self.distribution == "nb":
            distribution = NegativeBinomial(
                total_count=theta, logits=nb_logits, validate_args=False
            )
        return distribution.log_prob(x).sum(-1).mean()
