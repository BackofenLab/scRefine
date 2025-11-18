import torch
from torch import nn
from torch.nn import functional as F
import torch
import inspect
from torch import Tensor
from typing import Literal, Callable
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Subset
from torch.utils.data import Dataset
from collections.abc import Sequence
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
import scipy.sparse as sp_sparse
from uuid import uuid4
from typing import NamedTuple
from collections import defaultdict
import logging
import warnings
from pathlib import Path
import os
from pandas.api.types import CategoricalDtype
from copy import deepcopy
import collections
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
import pytorch_lightning as pl
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal, kl_divergence
from sklearn.preprocessing import LabelEncoder

_identity = lambda x: x
        
        
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


def embedding_distance_loss(embeddings, indices, decay_factor=0.1):


    group_0 = embeddings[indices == 0]
    group_1 = embeddings[indices == 1]
    
    
    dists_01 = torch.cdist(group_0, group_1, p=2)  
    dists_10 = torch.cdist(group_1, group_0, p=2)  


    weights_01 = torch.exp(-decay_factor * dists_01)
    weights_10 = torch.exp(-decay_factor * dists_10)

    weights_01 = torch.clip(weights_01 , 0, 5)
    weights_10 = torch.clip(weights_10 , 0, 5)

    mean_dist_01 = torch.sum(dists_01 * weights_01) / torch.sum(weights_01)
    mean_dist_10 = torch.sum(dists_10 * weights_10) / torch.sum(weights_10)


    return (mean_dist_01 + mean_dist_10) / 2

def kl_divergence_gmm(q_pi, q_mu, q_sigma, p_mu, p_sigma, p_pi):
    
    batch_size = q_mu.shape[0]  

    p_mu = p_mu.unsqueeze(0).expand(batch_size, -1, -1)  
    p_sigma = p_sigma.unsqueeze(0).expand(batch_size, -1, -1)  

    q_dist = Normal(q_mu, q_sigma)  

    
    p_mu = p_mu.squeeze()
    p_sigma = p_sigma.squeeze()


    p_dist = Normal(p_mu, p_sigma)  # p(z | c)

    kl_z_per_component = kl_divergence(q_dist, p_dist).sum(dim=-1)  # (batch_size, num_components)
    kl_z_per_component = kl_z_per_component.unsqueeze(dim=-1)
    kl_z = (q_pi * kl_z_per_component).sum(dim=-1)  
    p_pi = p_pi.unsqueeze(0).expand(batch_size, -1)  
    kl_c = (q_pi * (q_pi.log() - p_pi.log())).sum(dim=-1)  # (batch_size,)

    return kl_z + kl_c  
    
class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_cont
        The dimensionality of the continuous covariates
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list = None,
        n_cont: int = 0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = n_cont + sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + self.n_cov * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond


    def forward(self, x: torch.Tensor, *cat_list: int, cont: torch.Tensor | None = None):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        cont
            tensor of continuous covariates with shape ``(n_cont,)``

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """

        one_hot_cat_list = []  # for generality in this list many idxs useless.
        cont_list = [cont] if cont is not None else []
        cat_list = cat_list or []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        cov_list = cont_list + one_hot_cat_list
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            if (
                                x.device.type == "mps"
                            ):  # TODO: remove this when MPS supports for loop.
                                x = torch.cat(
                                    [(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0
                                )
                            else:
                                x = torch.cat(
                                    [layer(slice_x).unsqueeze(0) for slice_x in x], dim=0
                                )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                cov_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in cov_list
                                ]
                            else:
                                cov_list_layer = cov_list
                            x = torch.cat((x, *cov_list_layer), dim=-1)
                        x = layer(x)
        return x

                    
                    
class EncoderHVAE(nn.Module):

    def __init__(
        self,
        n_input, n_output,
        n_cat_list = None,
        n_layers = 1,
        n_hidden = 128,
        dropout_rate = 0.1,
        distribution = "normal",
        var_eps = 1e-4,
        var_activation = None,
        return_dist = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):

        q = self.encoder(x, *cat_list)

        q_m = self.mean_encoder(q)

        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps

        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent

        return q_m, q_v, latent
        
class EncoderHVAE2(nn.Module):


    def __init__(self, n_input, n_output, n_cat_list = None, n_layers = 1, n_hidden = 128, dropout_rate = 0.1, distribution = "normal", var_eps = 1e-4, return_dist = False,
    var_activation = None):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.mean_encoder2 = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):

        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        q_pi = self.mean_encoder2(q)

        return torch.nn.functional.softmax(q_pi), q_m, q_v
        
class DecoderHVAE(nn.Module):

    def __init__(self, n_input, n_output, n_cat_list= None, n_layers = 1, n_hidden = 128, inject_covariates = True, use_batch_norm = False, use_layer_norm = False, scale_activation = "softmax"
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
        )

        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        covariates: torch.Tensor,
        *cat_list: int,
    ):

        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = torch.nn.functional.relu(self.px_dropout_decoder(px))
        px_rate = torch.nn.functional.relu(torch.exp(library) * px_scale) 
        px_r = torch.nn.functional.relu(self.px_r_decoder(px)) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
        
class LearnableGMMPrior(nn.Module):
    def __init__(self, latent_dim, num_components):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.pi_logits = nn.Parameter(torch.ones(num_components))
        self.mu = nn.Parameter(torch.randn(num_components, latent_dim))
        self.log_sigma = nn.Parameter(torch.zeros(num_components, latent_dim))

    def forward(self):
        pi = torch.softmax(self.pi_logits, dim=0) 
        sigma = torch.exp(self.log_sigma)
        return self.mu, sigma, pi


class DecoderHVAE3(nn.Module):

    def __init__(
        self, n_input, n_output, n_cat_list = None, n_layers = 1, n_hidden = 128, inject_covariates = True,
        use_batch_norm = False, use_layer_norm = False, scale_activation = "softmax",
    ):
        super().__init__()
        self.px_decoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
            n_hidden=n_hidden, dropout_rate=0, inject_covariates=inject_covariates, use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,)

        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        self.px_r_decoder = nn.Linear(n_hidden, n_output)
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str,z: torch.Tensor, library: torch.Tensor, covariates: torch.Tensor,*cat_list: int,):

        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = torch.nn.functional.relu(self.px_dropout_decoder(px))
        px_rate = torch.nn.functional.relu(library * px_scale)
        px_r =  torch.nn.functional.softplus(self.px_r_decoder(px)) + 1e-4 if dispersion == "gene-cell" else None


        return px_scale, px_r, px_rate, px_dropout
        
        
@torch.no_grad()
def _mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):
    def pdist2(a, b):
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).t()
        return a2 + b2 - 2.0 * a @ b.t()

    Kxx = 0.0; Kyy = 0.0; Kxy = 0.0
    dxx = pdist2(x, x); dyy = pdist2(y, y); dxy = pdist2(x, y)

    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2))
        Kxx += torch.exp(-gamma * dxx)
        Kyy += torch.exp(-gamma * dyy)
        Kxy += torch.exp(-gamma * dxy)

    m = max(2, x.size(0)); n = max(2, y.size(0))
    mmd = (Kxx.sum() - torch.diagonal(Kxx).sum()) / (m * (m - 1.0)) \
        + (Kyy.sum() - torch.diagonal(Kyy).sum()) / (n * (n - 1.0)) \
        - 2.0 * Kxy.mean()
    return mmd


def gmvae_kl_with_mmd(
    z_mu, z_logvar, logits_q_c,
    p_mu, p_logvar, p_pi,
    *, mmd_weight=0.1, mmd_sigmas=(0.5,1.0,2.0,4.0,8.0), reduction="mean",
    sample_for_mmd=True
):
    eps = 1e-8
    B, D = z_mu.shape
    K, Dp = p_mu.shape
    assert D == Dp

    q_pi = F.softmax(logits_q_c, dim=-1).clamp_min(eps)         # [B,K]
    p_pi = (p_pi / (p_pi.sum() + eps)).clamp_min(eps)           # [K]

    z_mu_exp = z_mu.unsqueeze(1).expand(B, K, D)
    z_logvar_exp = z_logvar.unsqueeze(1).expand(B, K, D)
    p_mu_exp = p_mu.unsqueeze(0).expand(B, K, D)
    p_logvar_exp = p_logvar.unsqueeze(0).expand(B, K, D)

    sigma_q2 = z_logvar_exp.exp()
    sigma_p2 = p_logvar_exp.exp()
    diff2 = (z_mu_exp - p_mu_exp).pow(2)

    kl_per_dim = (p_logvar_exp - z_logvar_exp) + (sigma_q2 + diff2) / (sigma_p2 + eps) - 1.0
    kl_qz_qpzc = 0.5 * kl_per_dim.sum(dim=-1)                   # [B,K]

    kl_z_given_c = (q_pi * kl_qz_qpzc).sum(dim=1)               # [B]
    kl_c = (q_pi * (q_pi.log() - torch.log(p_pi).unsqueeze(0))).sum(dim=1)  # [B]

    if sample_for_mmd:
        std = (0.5 * z_logvar).exp()
        z_sample = z_mu + torch.randn_like(std) * std
    else:
        z_sample = z_mu

    with torch.no_grad():
        c_idx = torch.multinomial(p_pi, num_samples=B, replacement=True)
    p_mu_pick = p_mu[c_idx]
    p_std_pick = (0.5 * p_logvar[c_idx]).exp()
    z_prior_sample = p_mu_pick + torch.randn_like(p_std_pick) * p_std_pick

    mmd = _mmd_rbf(z_sample, z_prior_sample, sigmas=mmd_sigmas)

    if reduction == "mean":
        kl_z_given_c = kl_z_given_c.mean()
        kl_c = kl_c.mean()
    elif reduction == "sum":
        kl_z_given_c = kl_z_given_c.sum()
        kl_c = kl_c.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    total = kl_z_given_c + kl_c + mmd_weight * mmd
    return total, {"kl_z_given_c": kl_z_given_c, "kl_c": kl_c, "mmd": mmd}


class LearnableGMMPriorZ(nn.Module):
    def __init__(self, z_dim, num_components):
        super().__init__()
        self.z_dim = z_dim
        self.K = num_components
        self.mu = nn.Parameter(torch.randn(self.K, z_dim) * 0.01)
        self.logvar = nn.Parameter(torch.zeros(self.K, z_dim))
        self.logits_pi = nn.Parameter(torch.zeros(self.K)) 

    def forward(self):
        p_mu = self.mu                          
        p_logvar = self.logvar                  
        p_pi = F.softmax(self.logits_pi, dim=0) 
        return p_mu, p_logvar, p_pi


class HVAE_Lightning(pl.LightningModule):
    def __init__(
        self, n_input, n_latent=10, n_hidden=128, n_layers=1,
        encode_covariates=False, ln=False,
        lr=1e-3,
        kl_weight=0.001,             # will be ANNEALED; this is starting value
        mmd_weight=0.1,
        n_batch=9, num_components=9,
        reconstruction_weight=1.0,
        gene_factor=0.05,
        prediction_genes=18,
        random_assignment=False,
        dispersion="gene-cell",           # "gene" or "gene-cell"
        gene_dropout_shared=False,   # if True, gate is also per-gene (optional)
        kl_anneal_warmup_epochs=20,  # linear warm-up length
        kl_anneal_max=0.5,           # plateau value after warm-up
    ):
        super().__init__()
        self.save_hyperparameters()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encode_covariates = encode_covariates
        input_dim = n_input + 20 if encode_covariates else n_input
        self.random_assignment = random_assignment
        self.dispersion = dispersion
        self.reconstruction_weight = reconstruction_weight
        self.gene_factor = gene_factor
        self.lr = lr

        self.z_encoder = EncoderHVAE(input_dim, n_latent, n_layers=n_layers)
        self.l_encoder = EncoderHVAE2(n_latent, num_components, n_layers=n_layers)
        self.decoder = DecoderHVAE3(n_latent, n_input, n_layers=n_layers, use_layer_norm=True)

        self.gmm_prior = LearnableGMMPrior(1, num_components=num_components).to(device)

        self.c_head = nn.Sequential(
            nn.Linear(n_latent, 256), nn.ReLU(),
            nn.Linear(256, num_components)
        )
        self.gmm_prior_z = LearnableGMMPriorZ(n_latent, num_components).to(device)
        self.mmd_weight = mmd_weight

        self.pred_genes = nn.Linear(n_latent, prediction_genes)

        self.cov_embedding = nn.Embedding(n_batch, 20, padding_idx=0, max_norm=2)
        nn.init.normal_(self.cov_embedding.weight, mean=0.0, std=0.01)

        if self.dispersion == "gene":
            self.r_gene_unconstrained = nn.Parameter(torch.zeros(n_input))
            if getattr(self, "gene_dropout_shared", gene_dropout_shared):
                self.pi_gene_logit = nn.Parameter(torch.full((n_input,), -1.0))
            else:
                self.pi_gene_logit = None


        self.kl_weight_start = kl_weight
        self.kl_anneal_warmup_epochs = kl_anneal_warmup_epochs
        self.kl_anneal_max = kl_anneal_max

    def one_hot_encode(self, labels, num_classes):
        batch_size = labels.shape[0]
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot

    def _current_kl_weight(self):
        """
        Linear warm-up from kl_weight_start 
        """
        e = max(0, int(self.current_epoch))
        if self.kl_anneal_warmup_epochs <= 0:
            return self.kl_anneal_max
        t = min(1.0, e / float(self.kl_anneal_warmup_epochs))
        return (1.0 - t) * self.kl_weight_start + t * self.kl_anneal_max
    

    def get_dataset_baseline(self, counts, batch_data, n_batch):

        lib_b = torch.empty(n_batch)
        for b in range(n_batch):
            s = torch.tensor(counts[np.array(batch_data) == b].sum(1)).clamp_min(1.) 
            lib_b[b] = torch.log(s).mean()        
        self.register_buffer("loglib_baseline_per_batch", lib_b)     # shape [n_batch]

        return

    # ---------- forward ----------
    def forward(self, x, gene, covariates=None):
        assert not torch.isinf(x).any(), "inf detected in input data!"
        assert not torch.isnan(x).any(), "NaN detected in input data!"
        assert covariates is None or not torch.isnan(covariates).any(), "NaN detected in covariates input!"

        batch_labels = covariates
        covariates = self.cov_embedding(covariates) if covariates is not None else None

        if self.encode_covariates and covariates is not None:
            x = torch.cat([x, covariates], dim=-1)

        z_mu, z_var, z = self.z_encoder(x)
        l_pi, l_mu, l_var = self.l_encoder(z_mu)

        c = torch.multinomial(l_pi, num_samples=1).squeeze()
        if self.random_assignment == False and batch_labels is not None:
            std = torch.sqrt(l_var + 1e-8)
            library = l_mu[torch.arange(len(batch_labels)), batch_labels] + \
                      std[torch.arange(len(batch_labels)), batch_labels] * torch.randn_like(l_var[:, 0])
            
            std = torch.sqrt(l_var + 1e-8)                           # [B,K]
            lib_sample = l_mu[torch.arange(len(batch_labels)), batch_labels] + \
                      std[torch.arange(len(batch_labels)), batch_labels] * torch.randn_like(l_var[:, 0])
            library = self.loglib_baseline_per_batch[batch_labels] + lib_sample

        else:
            std = torch.sqrt(l_var + 1e-8)                           # [B,K]
            lib_sample = l_mu[torch.arange(len(c)), c] + std[torch.arange(len(c)), c] * torch.randn_like(l_var[:, 0])
            library = self.loglib_baseline_per_batch[batch_labels] + lib_sample

        px_scale, px_r_dec, px_rate, px_dropout_dec = self.decoder(self.dispersion, z, library.unsqueeze(dim=-1), covariates)

        logits_q_c = self.c_head(z_mu)
        gene_pred = self.pred_genes(z_mu)

        return {
            "z_mu": z_mu,
            "z_var": z_var,
            "z": z,
            "ql": Normal(l_mu, torch.sqrt(l_var)),
            "l_pi": l_pi, "l_mu": l_mu, "l_var": l_var,
            "library": library,
            "px_scale": px_scale,
            "px_r_dec": px_r_dec,                 # decoder-produced (may ignore under 'gene' dispersion)
            "px_rate": px_rate,
            "px_dropout_dec": px_dropout_dec,     # decoder-produced gate
            "covariates": covariates,
            "logits_q_c": logits_q_c,
            "gene_pred": gene_pred
        }

    # ---------- loss ----------
    def compute_loss(self, x, output, gene):
        eps = 1e-5
        z_mu, z_var = output["z_mu"], output["z_var"]
        l_pi, l_mu, l_var = output["l_pi"], output["l_mu"], output["l_var"]
        px_rate = output["px_rate"]
        px_dropout_dec = output["px_dropout_dec"]
        gene_pred = output["gene_pred"]
        logits_q_c = output["logits_q_c"]


        p_mu_l, p_sigma_l, p_pi_l = self.gmm_prior()
        kl_l = kl_divergence_gmm(l_pi, l_mu, l_var, p_mu_l, p_sigma_l, p_pi_l).mean()

        p_mu_z, p_logvar_z, p_pi_z = self.gmm_prior_z()
        kl_z_total, pieces = gmvae_kl_with_mmd(
            z_mu=z_mu,
            z_logvar=torch.log(z_var + eps), 
            logits_q_c=logits_q_c,
            p_mu=p_mu_z,
            p_logvar=p_logvar_z,
            p_pi=p_pi_z,
            mmd_weight=self.mmd_weight,
            mmd_sigmas=(0.5,1.0,2.0,4.0,8.0),
            reduction="mean",
            sample_for_mmd=True
        )


        B, G = x.shape
        if self.dispersion == "gene":
            r_gene = F.softplus(self.r_gene_unconstrained)           
            px_r = r_gene.unsqueeze(0).expand(B, G)                  
            if self.pi_gene_logit is not None:
                pi_gene = torch.sigmoid(self.pi_gene_logit)          
                px_dropout = pi_gene.unsqueeze(0).expand(B, G)
            else:
                px_dropout = px_dropout_dec                         
        else:
            px_r = output["px_r_dec"]
            px_dropout = px_dropout_dec

        px_r = px_r.clamp(min=1e-2)
        px_rate = px_rate.clamp(min=1e-4)
        px_dropout = px_dropout.clamp(min=1e-2, max=1 - 1e-2)
        nb_probs = (px_r / (px_r + px_rate)).clamp(1e-5, 1 - 1e-5)

        zinb = ZeroInflatedNegativeBinomial(
            probs=nb_probs,
            total_count=px_r,
            gate=px_dropout,
            validate_args=False
        )

        x_nonneg = x
        zzn = -zinb.log_prob(x_nonneg).mean(dim=-1)  # [B]

        recon = zzn.mean() * self.reconstruction_weight

        ce_loss = torch.nn.MSELoss()
        ce_loss_val = ce_loss(torch.clip(gene_pred.float(), max=2), gene.float())
        kl_w = self._current_kl_weight()
        loss = recon + kl_w * (kl_z_total + kl_l) + self.gene_factor * ce_loss_val
        
        """
        with torch.no_grad():
            print("x_counts stats  min/max/mean:", x.min().item(), x.max().item(), x.float().mean().item())
            print("mu:", px_rate.min().item(), px_rate.max().item(), px_rate.float().mean().item())
            print("nb_probs:", nb_probs.min().item(), nb_probs.max().item(), nb_probs.float().mean().item())
            print("total counts:", px_r.min().item(), px_r.max().item(), px_r.float().mean().item())
            print("gate:", px_dropout.min().item(), px_dropout.max().item(), px_dropout.float().mean().item())
            print("#################################################################################")
        """

        logs = {
            "recon": recon.detach(),
            "kl_z_total": (kl_w * kl_z_total).detach(),
            "kl_l": (kl_w * kl_l).detach(),
            "gene_aux": ce_loss_val.detach(),
            "kl_weight_eff": torch.tensor(kl_w, device=x.device),
            "mmd": pieces["mmd"],
        }
        return {"train_loss": loss, **logs}


    def training_step(self, batch, batch_idx):
        x, _, covariates, gene = batch if self.encode_covariates else (batch, None)

        output = self(x, gene, covariates)
        loss_dict = self.compute_loss(x, output, gene)

        self.log("train_loss", loss_dict["train_loss"], prog_bar=True)
        self.log("train/recon", loss_dict["recon"], prog_bar=True)
        self.log("train/kl_z_total", loss_dict["kl_z_total"])
        self.log("train/kl_l", loss_dict["kl_l"])
        self.log("train/mmd", loss_dict["mmd"])
        self.log("train/gene_aux", loss_dict["gene_aux"])
        self.log("train/kl_weight_eff", loss_dict["kl_weight_eff"], prog_bar=True)
        return loss_dict["train_loss"]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "lr_scheduler": {"scheduler": scheduler}
        }
    
