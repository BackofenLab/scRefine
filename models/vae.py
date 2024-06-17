import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *
from distributions.von_mises_fisher import VonMisesFisher
from distributions.hyperspherical_uniform import HypersphericalUniform
from torch.distributions.normal import Normal
import numpy as np


class SCPHERE(nn.Module):
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
        super(SCPHERE, self).__init__()

        if encoder_layer is None:
            encoder_layer = [128, 64, 32]
        if decoder_layer is None:
            decoder_layer = [32, 128]

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
            self.fc_var = nn.Linear(encoder_layer[-1], latent_dim)
        elif self.distribution == "vmf":
            self.fc_mu = nn.Linear(encoder_layer[-1], latent_dim)
            self.fc_var = nn.Linear(encoder_layer[-1], 1)
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

        self.decoder = nn.Sequential(*modules)
        self.fc_logits = nn.Linear(decoder_layer[-1], input_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        if self.distribution == "normal":
            mu = self.fc_mu(result)
            var = F.softplus(self.fc_var(result))
        elif self.distribution == "vmf":
            mu = self.fc_mu(result)
            mu = mu / mu.norm(p=2, dim=-1, keepdim=True)
            # the `+ 1` to prevent collapsing behaviors
            var = F.softplus(self.fc_var(result)) + 1
        else:
            raise NotImplemented

        return mu, var

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        result = self.fc_logits(result)
        return result

    def reparameterize(self, mu: Tensor, var: Tensor) -> Tensor:
        if self.distribution == "normal":
            q_z = Normal(mu, var)
            p_z = Normal(torch.zeros_like(mu), torch.ones_like(var))
        elif self.distribution == "vmf":
            q_z = VonMisesFisher(mu, var)
            p_z = HypersphericalUniform(self.latent_dim - 1, device=mu.device)
        else:
            raise NotImplemented

        return q_z, p_z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, var = self.encode(input)
        q_z, p_z = self.reparameterize(mu, var)
        z = q_z.rsample()
        return (mu, var), (q_z, p_z), z, self.decode(z)


def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    mu, var = model.encode(x.reshape(-1, 784))
    q_z, p_z = model.reparameterize(mu, var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == "normal":
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction="none")(
        x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))
    ).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == "normal":
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()
