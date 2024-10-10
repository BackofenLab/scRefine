import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


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

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=0.00025) -> dict:
        recon_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
        loss = recon_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "kld_loss": -kld_loss.detach(),
        }
