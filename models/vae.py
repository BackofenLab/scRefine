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

    def __init__(
        self,
        n_in, n_out,
        n_cat_list: Iterable[int] = None, n_cont = 0, n_layers = 1,
        n_hidden = 128, dropout_rate = 0.1, use_activation = True, inject_covariates = True,
        activation_fn: nn.Module = nn.ReLU, use_layer_norm = False, use_batch_norm = True
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = n_cont + sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [(f"Layer {i}",nn.Sequential(nn.Linear(n_in + self.n_cov * self.inject_into_layer(i),n_out,bias=True,),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,),)
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num):
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond


                    
                    
                    
class EncoderHVAE(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
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

        # mean gamma
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
        px_rate = torch.clamp(torch.nn.functional.relu(torch.exp(library) * px_scale), max=12)   # torch.clamp( , max=12)
        px_r = torch.nn.functional.relu(self.px_r_decoder(px)) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
        
        
        
class HVAE_Lightning(pl.LightningModule):
    def __init__(self, n_input, n_latent=10, n_hidden=128, n_layers=1, encode_covariates=False, ln=False, kl_weight=0.1, lr=1e-3, n_batch = 9, num_components = 2):
    
        super().__init__()
        self.save_hyperparameters()
        self.encode_covariates = encode_covariates
        input_dim = n_input + 20 if encode_covariates else n_input
        self.z_encoder = EncoderHVAE(input_dim, n_latent,n_layers = n_layers)
        self.l_encoder = EncoderHVAE2(n_latent, num_components, n_layers = n_layers)
        self.decoder = DecoderHVAE(n_latent, n_input, n_layers = n_layers)
        self.gmm_prior = LearnableGMMPrior(1, num_components=num_components).to("cuda")
        self.kl_weight = kl_weight
        self.lr = lr
        self.dispersion = "gene-cell"
        self.cov_embedding =  nn.Embedding(n_batch, 20)
        
        
    def one_hot_encode(self,labels, num_classes):

        batch_size = labels.shape[0]
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
 
        return one_hot
        
    def forward(self, x, covariates=None):
    
        covariates = self.cov_embedding(covariates)
        if self.encode_covariates and covariates is not None:
            x = torch.cat([x, covariates], dim=-1)
        
        assert not torch.isnan(x).any(), "NaN detected in input data!"

        z_mu, z_var, z = self.z_encoder(x)
        l_pi, l_mu, l_var = self.l_encoder(z_mu)

        c = torch.multinomial(l_pi, num_samples=1).squeeze()
        library = l_mu[torch.arange(len(c)), c] + l_var[torch.arange(len(c)), c] * torch.randn_like(l_var[:, 0])
        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library.unsqueeze(dim=-1), covariates)
        
        return {
            "z_mu": z_mu, "z": z, "qz": Normal(z_mu, torch.sqrt(z_var)), 
            "ql": Normal(l_mu, torch.sqrt(l_var)), "library": library, 
            "px_scale": px_scale, "px_r": px_r, "px_rate": px_rate, "px_dropout": px_dropout, "l_pi": l_pi, "l_var": l_var, "l_mu":l_mu, "covariates":covariates
        }
    
    def compute_loss(self, x, output):
    
        qz, ql, l_pi, l_mu, l_var, px_rate, px_r, px_dropout, covariates = output["qz"], output["ql"], output["l_pi"], output["l_mu"], output["l_var"], output["px_rate"], output["px_r"], output["px_dropout"], output["covariates"]

        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        kl_z = kl_divergence(qz, pz).sum(dim=-1)       
        p_mu, p_sigma, p_pi = self.gmm_prior()
        kl_l = kl_divergence_gmm(l_pi, l_mu, l_var, p_mu, p_sigma, p_pi).mean()
        
        if self.dispersion == "gene-cell":

            px_dropout = px_dropout.clamp(min=1e-5, max=1-1e-5)
            
            zinb = ZeroInflatedNegativeBinomial(
            total_count=px_r.clamp(min=1e-5),  
            logits=torch.log(px_rate.clamp(min=1e-5)), )
            gate= px_dropout,
            validate_args=False
            )

        reconst_loss = -zinb.log_prob(x.round()).sum(dim=-1).mean()
        loss = (reconst_loss + (kl_z + kl_l) * self.kl_weight).mean()
        
        return {"loss": loss, "recon_loss": reconst_loss.item(), "kld_loss1": kl_z.mean() * self.kl_weight, "kld_loss2": kl_l.mean() * self.kl_weight}
    
    def training_step(self, batch, batch_idx):
    
        x,_,count, covariates = batch if self.encode_covariates else (batch, None)
        output = self(x, covariates)
        loss_dict = self.compute_loss(x, output)
        self.log("train_loss", loss_dict["loss"], prog_bar=True)
        self.log("train_recon_loss", loss_dict["recon_loss"], prog_bar=True)
        self.log("train_kld_loss1", loss_dict["kld_loss1"], prog_bar=True)
        self.log("train_kld_loss2", loss_dict["kld_loss2"], prog_bar=True)
        
        return loss_dict["loss"]

        
    def configure_optimizers(self):
    
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
        
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 5.0,  # Clip gradients to prevent explosions
            "lr_scheduler": {"scheduler": scheduler}
        }
        

        
        
class HVAE_Lightning7(pl.LightningModule):
    def __init__(self, n_input, n_latent=10, n_hidden=128, n_layers=1, encode_covariates=False, ln=False, kl_weight=0.001, lr=1e-3, n_batch = 9, num_components = 2, reconstruction_weight = 0.001, gene_factor = 0.5, prediction_genes = 18):
        super().__init__()
        self.save_hyperparameters()
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.encode_covariates = encode_covariates
        input_dim = n_input + 20 if encode_covariates else n_input
        
        self.z_encoder = EncoderHVAE(input_dim, n_latent,n_layers = n_layers)
        self.l_encoder = EncoderHVAE2(n_latent, num_components, n_layers = n_layers)
        self.decoder = DecoderHVAE3(n_latent, n_input, n_layers = n_layers, use_layer_norm = True)

        self.gmm_prior = LearnableGMMPrior(1, num_components=num_components).to(device)
        self.kl_weight = kl_weight
        self.lr = lr
        self.dispersion = "gene-cell"
        self.pred_genes = torch.nn.Linear(n_latent, prediction_genes)
        self.reconstruction_weight = reconstruction_weight
        self.gene_factor = gene_factor
        
        self.cov_embedding =  nn.Embedding(n_batch, 20, padding_idx = 0, max_norm = 2)
        nn.init.normal_(self.cov_embedding.weight, mean=0.0, std=0.01)

        
    def one_hot_encode(self,labels, num_classes):

        batch_size = labels.shape[0]
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
 
        return one_hot
        
    def forward(self, x, gene, covariates=None):
    
        assert not torch.isinf(x).any(), "inf detected in input data!"
        assert not torch.isnan(x).any(), "NaN detected in input data!"
        assert not torch.isnan(covariates).any(), "NaN detected in covariates inputdata!"
        
        covariates = self.cov_embedding(covariates)

        assert not torch.isnan(covariates).any(), "NaN detected in covariates data!"
        
        
        if self.encode_covariates and covariates is not None:
            x = torch.cat([x, covariates], dim=-1)
        
        assert not torch.isnan(x).any(), "NaN detected in input data!"

        z_mu, z_var, z = self.z_encoder(x)
        l_pi, l_mu, l_var = self.l_encoder(z_mu)
        
        c = torch.multinomial(l_pi, num_samples=1).squeeze()
        
        library = l_mu[torch.arange(len(c)), c] + l_var[torch.arange(len(c)), c] * torch.randn_like(l_var[:, 0])
        
        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library.unsqueeze(dim=-1), covariates)
        gene_pred = self.pred_genes(z_mu)
        
        return {
            "z_mu": z_mu, "z": z, "qz": Normal(z_mu, torch.sqrt(z_var)), 
            "ql": Normal(l_mu, torch.sqrt(l_var)), "library": library, 
            "px_scale": px_scale, "px_r": px_r, "px_rate": px_rate, "px_dropout": px_dropout, "l_pi": l_pi, "l_var": l_var, "l_mu":l_mu, "covariates":covariates,
            "gene_pred": gene_pred
        }
    
    def compute_loss(self, x, output, gene):
        qz, ql, l_pi, l_mu, l_var, px_rate, px_r, px_dropout, covariates, gene_pred = output["qz"], output["ql"], output["l_pi"], output["l_mu"], output["l_var"], output["px_rate"], output["px_r"], output["px_dropout"], output["covariates"], output["gene_pred"]

        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        kl_z = kl_divergence(qz, pz).sum(dim=-1)       

        p_mu, p_sigma, p_pi = self.gmm_prior()

        kl_l = kl_divergence_gmm(l_pi, l_mu, l_var, p_mu, p_sigma, p_pi).mean()
        
        
        if self.dispersion == "gene-cell":
            px_dropout = px_dropout.clamp(min=1e-5, max=1-1e-5)
            
            eps = 1e-5
            px_r = px_r.clamp(min=1e-2)
            px_rate = px_rate.clamp(min=1e-2)
            px_dropout = px_dropout.clamp(min=1e-2, max=1. - 1e-2)
            logits = torch.log(px_rate.clamp(min=1e-2))

            zinb = ZeroInflatedNegativeBinomial(
            total_count=px_r.clamp(min=1e-2),
            probs=px_rate.clamp(min=1e-4, max=1 - 1e-4),
            gate=px_dropout,
            validate_args=False
            )



        zz = -zinb.log_prob(torch.nn.functional.relu(x).round()).sum(dim=-1)
        assert not torch.isinf(zz).any(), "inf detected in zinb data!"
        assert not torch.isnan(zz).any(), "NaN detected in zinb data!"
        reconst_loss = -zinb.log_prob(torch.nn.functional.relu(x).round()).sum(dim=-1).mean() * self.reconstruction_weight
        
        ce_loss = torch.nn.MSELoss()
        ce_loss_val = ce_loss(torch.clip(gene_pred.float(), max=2), gene.float())
        
        loss = ((reconst_loss + (kl_z + kl_l) * self.kl_weight).mean()) *0.1 + self.gene_factor * ce_loss_val
        
        return {"loss": loss, "recon_loss": reconst_loss.item(), "kld_loss1": kl_z.mean() * self.kl_weight, "kld_loss2": kl_l.mean() * self.kl_weight, "gene": ce_loss_val}
    
    def training_step(self, batch, batch_idx):
    
        x,_,count, covariates, gene = batch if self.encode_covariates else (batch, None)
        x  = torch.clamp(x, max=10)
        output = self(x, gene, covariates)

        loss_dict = self.compute_loss(x, output, gene)
        self.log("train_loss", loss_dict["loss"], prog_bar=True)
        self.log("train_recon_loss", loss_dict["recon_loss"], prog_bar=True)
        self.log("train_kld_loss1", loss_dict["kld_loss1"], prog_bar=True)
        self.log("train_kld_loss2", loss_dict["kld_loss2"], prog_bar=True)
        self.log("gene loss", loss_dict["gene"], prog_bar=True)

        return loss_dict["loss"]

        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
        
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "lr_scheduler": {"scheduler": scheduler}
        }
