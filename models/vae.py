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
        
class ZINB(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, mu, theta, pi):
        eps = self.eps
        softplus_theta = torch.nn.functional.softplus(theta)


        mu = torch.clamp(mu, min=eps, max=1e6)
        softplus_theta = torch.clamp(softplus_theta, min=eps, max=1e6)
        pi = torch.clamp(pi, min=eps, max=1 - eps)

        log_theta_eps = torch.log(softplus_theta + eps)
        log_mu_eps = torch.log(mu + eps)

        nb_case = (
            torch.lgamma(softplus_theta + x)
            - torch.lgamma(softplus_theta)
            - torch.lgamma(x + 1)
            + softplus_theta * (log_theta_eps + log_mu_eps - log_theta_eps)
            + x * (log_mu_eps - log_theta_eps + log_mu_eps)
        )
        log_nb_prob = nb_case - (softplus_theta + x) * torch.log1p(mu / softplus_theta + eps)


        log_nb_prob = torch.clamp(log_nb_prob, min=-50, max=50)

        log_zero_prob = torch.log(pi + (1.0 - pi) * torch.exp(log_nb_prob) + eps)
        log_nonzero_prob = torch.log(1.0 - pi + eps) + log_nb_prob
        
        result = torch.where(x < eps, log_zero_prob, log_nonzero_prob)


        return -torch.clamp(result, min=-50, max=50).sum(dim=-1).mean()

class Encoder2(nn.Module):
    def __init__(self, input_dim, latent_dim, n_hidden=128, n_layers=1, ln = False):
        super().__init__()
        if ln: 
            layers = [nn.LayerNorm(input_dim)]
            layers += [nn.Linear(input_dim, n_hidden), nn.ReLU()]
        else:
            layers = [nn.Linear(input_dim, n_hidden), nn.ReLU()]


        for _ in range(n_layers - 1):
            if ln: layers += [nn.LayerNorm(n_hidden)]
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]


        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(n_hidden, latent_dim)
        self.var = nn.Linear(n_hidden, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.mu(h), torch.exp(self.var(h))

class Decoder2(nn.Module):
    def __init__(self, latent_dim, output_dim, n_hidden=128, n_layers=1, ln = False):
        super().__init__()
        
        if ln:
            layers  = [nn.LayerNorm(latent_dim)]
            layers += [nn.Linear(latent_dim, n_hidden), nn.ReLU()]
        else:
            layers = [nn.Linear(latent_dim, n_hidden), nn.ReLU()]
            
        for _ in range(n_layers - 1):
            if ln: layers += [nn.LayerNorm(n_hidden)]
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]


        layers.append(nn.Linear(n_hidden, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z, library):
        return torch.exp(self.decoder(z) + library)

class VAE_simple(nn.Module):
    def __init__(self, n_input, n_latent=10, n_hidden=128, n_layers=1, encode_covariates=False, ln = False):
        super().__init__()
        self.encode_covariates = encode_covariates
        input_dim = n_input + 1 if encode_covariates else n_input
        
        self.z_encoder = Encoder2(input_dim, n_latent, n_hidden, n_layers, ln)
        self.l_encoder = Encoder2(input_dim, 1, n_hidden, 1, ln)  # Library size encoder
        self.decoder = Decoder2(n_latent, n_input, n_hidden, n_layers)
        
    def forward(self, x, covariates=None):
        if self.encode_covariates and covariates is not None:
            #print(x.size())
            #print(covariates.size())
            
            covariates = torch.unsqueeze(covariates, dim = -1)
            #print(covariates.size())
            x = torch.cat([x, covariates], dim=-1)
        
        z_mu, z_var = self.z_encoder(x)
        qz = Normal(z_mu, torch.sqrt(z_var))
        z = qz.rsample()
        
        l_mu, l_var = self.l_encoder(x)
        ql = Normal(l_mu, torch.sqrt(l_var))
        library = ql.rsample()
        
        px_rate = self.decoder(z, library)
        
        return {
            "z_mu": z_mu, "z": z, "qz": qz, "ql": ql, "library": library, "px_rate": px_rate
        }
    
    def loss(self, x, output, kl_weight = 0.001):
        qz, ql, px_rate = output["qz"], output["ql"], output["px_rate"]
        
        # KL divergence
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        pl = Normal(torch.zeros_like(ql.loc), torch.ones_like(ql.scale))
        kl_z = kl_divergence(qz, pz).sum(dim=-1)
        kl_l = kl_divergence(ql, pl).sum(dim=-1)
        
        reconst_loss = nn.MSELoss()(px_rate, x)

        loss = (reconst_loss + (kl_z + kl_l)*kl_weight).mean()
        # Total loss
        return {"loss": loss, "recon_loss": reconst_loss.item(), "kld_loss1": kl_z.mean() * kl_weight, "kld_loss2": kl_l.mean() * kl_weight, "distance": 0}
