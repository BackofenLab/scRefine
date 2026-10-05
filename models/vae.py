import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal, kl_divergence
from torch.optim import Adam

_identity = lambda x: x
        
        
class ZINB(nn.Module):
    """Zero-inflated negative binomial negative log-likelihood loss."""

    def __init__(self, eps=1e-8):
        """Args: eps (float, optional): Numerical stability floor for clamps/logs."""
        super().__init__()
        self.eps = eps

    def forward(self, x, mu, theta, pi):
        """
        Args:
            x (Tensor): Observed counts.
            mu (Tensor): Negative binomial mean.
            theta (Tensor): Pre-softplus dispersion parameter.
            pi (Tensor): Zero-inflation (dropout) probability.

        Returns:
            Tensor: Scalar mean negative log-likelihood over the batch.
        """
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
    """
    Distance-weighted loss between two groups of embeddings (selected by `indices` in
    {0, 1}): the mean pairwise L2 distance between groups, weighted by
    `exp(-decay_factor * distance)` (clipped to [0, 5]) so nearby cross-group pairs
    contribute more.

    Args:
        embeddings (Tensor): Embeddings for all samples, shape (N, D).
        indices (Tensor): Group assignment per sample; only values 0 and 1 are used.
        decay_factor (float, optional): Controls how quickly the distance weight decays.

    Returns:
        Tensor: Scalar loss, the average of the two directional weighted mean distances.
    """
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


def _nt_xent_by_labels(z, labels, temperature=0.5):
    """
    NT-Xent (SimCLR-style) contrastive loss treating same-label samples within the batch
    as positive pairs; adapted from `train_simclr.py::nt_xent_loss` so it can be reused for
    arbitrary integer group ids (not just class labels).

    Args:
        z (Tensor): Embeddings, shape (batch, dim).
        labels (Tensor): Integer group id per sample, shape (batch,).
        temperature (float, optional): Softmax temperature.

    Returns:
        Tensor: Scalar loss.
    """
    batch_size = z.shape[0]

    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T) / temperature

    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)

    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

    positive_log_prob = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-10)

    return -positive_log_prob.mean()


def _nt_xent_weighted(z, values, batch_labels=None, temperature=0.5, bandwidth=None):
    """
    NT-Xent contrastive loss where "positive-ness" between a pair of samples is a
    continuous weight (a Gaussian kernel over `values`) rather than a binary same-label
    mask — used for contrastive learning directly on a continuous target (e.g. raw
    log-normalized marker-gene expression) instead of a discretized bin/label.

    Args:
        z (Tensor): Embeddings, shape (batch, dim).
        values (Tensor): Continuous per-sample value used to weight positive pairs, shape
            (batch,).
        batch_labels (Tensor, optional): Integer batch/domain id per sample, shape
            (batch,). When given, pairs from the same batch are excluded from both the
            softmax denominator and the positive weighting — only cross-batch pairs are
            compared, so a positive/negative pair is only ever "same expression, different
            batch" or "different expression, different batch", which is the signal that
            actually reflects batch-integration rather than within-batch similarity.
            When `None`, only the self-pair (diagonal) is excluded.
        temperature (float, optional): Softmax temperature for the embedding similarities.
        bandwidth (float, optional): Kernel bandwidth for the value-similarity weight.
            Defaults to `values.std()` (clamped away from 0) so the weighting adapts to the
            scale of each batch/gene.

    Returns:
        Tensor: Scalar loss.
    """
    batch_size = z.shape[0]

    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T) / temperature

    bw = bandwidth if bandwidth is not None else values.std().clamp_min(1e-4)
    diff = (values.unsqueeze(1) - values.unsqueeze(0)).abs()
    weight = torch.exp(-diff / bw)

    if batch_labels is not None:
        valid_mask = batch_labels.unsqueeze(1) != batch_labels.unsqueeze(0)
    else:
        valid_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)

    weight = weight * valid_mask

    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * valid_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

    weighted_log_prob = (weight * log_prob).sum(dim=1) / (weight.sum(dim=1) + 1e-10)

    return -weighted_log_prob.mean()


def gene_expression_contrastive_loss(z_proj, gene_values, batch_labels=None, temperature=0.5, bandwidth=None):
    """
    Contrastive alternative to the MSE-based marker-gene prediction head, operating
    directly on raw (log-normalized, unbinned) marker-gene expression: for each marker gene
    independently, pairs of cells are weighted as positives in proportion to how close
    their expression is (`_nt_xent_weighted`), and the per-gene losses are averaged.

    Args:
        z_proj (Tensor): Projected latent embeddings, shape (batch, dim).
        gene_values (Tensor): Log-normalized marker-gene expression, shape (batch, n_genes),
            as produced by the `AnnDataCollator` when `bin_marker_genes=False`.
        batch_labels (Tensor, optional): Integer batch/domain id per sample, shape (batch,),
            forwarded to `_nt_xent_weighted` so only cross-batch pairs are compared.
        temperature (float, optional): NT-Xent softmax temperature.
        bandwidth (float, optional): Kernel bandwidth passed to `_nt_xent_weighted`;
            defaults to each gene's per-batch std.

    Returns:
        Tensor: Scalar loss, averaged over marker genes.
    """
    n_genes = gene_values.shape[1]
    losses = [
        _nt_xent_weighted(
            z_proj, gene_values[:, g], batch_labels=batch_labels,
            temperature=temperature, bandwidth=bandwidth,
        )
        for g in range(n_genes)
    ]
    return torch.stack(losses).mean()


def kl_divergence_gmm(q_pi, q_mu, q_sigma, p_mu, p_sigma, p_pi):
    """
    KL divergence between a per-sample Gaussian-mixture posterior `q(z, c)` and a Gaussian-
    mixture prior `p(z, c)`, marginalized over the discrete component `c`.

    Args:
        q_pi (Tensor): Posterior component responsibilities, shape (batch, num_components).
        q_mu (Tensor): Posterior means per component, shape (batch, num_components, dim).
        q_sigma (Tensor): Posterior std devs per component, same shape as `q_mu`.
        p_mu (Tensor): Prior means per component, shape (num_components, dim) or (1, ...).
        p_sigma (Tensor): Prior std devs per component, same shape as `p_mu`.
        p_pi (Tensor): Prior component probabilities, shape (num_components,).

    Returns:
        Tensor: Per-sample KL divergence, shape (batch,).
    """
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


def kl_divergence_gmm_marginalized(q_pi, q_mu, q_sigma, p_mu, p_sigma, p_pi):
    """
    Component-marginalized KL between a per-sample Gaussian-mixture posterior `q(z, c)`
    and a Gaussian-mixture prior `p(z, c)`:
    `sum_k q_pi_k * KL(q(z|c=k) || p(z|c=k)) + KL(q(c) || p(c))`.

    Fixed version of `kl_divergence_gmm` (used when `library_kl="marginalized"`), which sums
    the per-component KLs over all components *before* weighting by `q_pi`, so the
    weighting cancels out (`q_pi` sums to 1) and every component's KL counts fully.

    Args:
        q_pi (Tensor): Posterior component responsibilities, shape (batch, K).
        q_mu (Tensor): Posterior mean per component, shape (batch, K).
        q_sigma (Tensor): Posterior std dev per component, shape (batch, K).
        p_mu (Tensor): Prior mean per component, shape (K, 1).
        p_sigma (Tensor): Prior std dev per component, shape (K, 1).
        p_pi (Tensor): Prior component probabilities, shape (K,).

    Returns:
        Tensor: Per-sample KL divergence, shape (batch,).
    """
    kl_per_component = kl_divergence(
        Normal(q_mu, q_sigma),
        Normal(p_mu.squeeze(-1).unsqueeze(0), p_sigma.squeeze(-1).unsqueeze(0)),
    )                                                            # (batch, K)
    kl_z = (q_pi * kl_per_component).sum(dim=-1)
    kl_c = (q_pi * (q_pi.log() - p_pi.log().unsqueeze(0))).sum(dim=-1)
    return kl_z + kl_c


def kl_divergence_batch_component(q_mu, q_sigma, p_mu, p_sigma, batch_labels):
    """
    Library-size KL restricted to each cell's own batch component:
    `KL(q(l | c=b) || p(l | c=b))` with `b` the cell's batch label (used when
    `library_kl="batch"`).

    This matches the generative path in `HVAE_Lightning.forward`, which samples the
    library size from the batch-label component (not from `q_pi`), so each cell's library
    Gaussian is pulled towards exactly the prior component it is sampled from. The
    component responsibilities `q_pi` and prior weights `p_pi` play no role here.

    Args:
        q_mu (Tensor): Posterior mean per component, shape (batch, K).
        q_sigma (Tensor): Posterior std dev per component, shape (batch, K).
        p_mu (Tensor): Prior mean per component, shape (K, 1).
        p_sigma (Tensor): Prior std dev per component, shape (K, 1).
        batch_labels (Tensor): Integer batch id per cell, shape (batch,).

    Returns:
        Tensor: Per-sample KL divergence, shape (batch,).
    """
    rows = torch.arange(len(batch_labels), device=q_mu.device)
    return kl_divergence(
        Normal(q_mu[rows, batch_labels], q_sigma[rows, batch_labels]),
        Normal(p_mu[batch_labels, 0], p_sigma[batch_labels, 0]),
    )


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network based on scVI github repository.

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
        """Returns whether covariates should be concatenated into hidden layer `layer_num`."""
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
        """
        Args:
            n_input (int): Input feature dimensionality.
            n_output (int): Latent dimensionality.
            n_cat_list (list[int], optional): Category counts for each categorical covariate.
            n_layers (int, optional): Number of hidden layers in the shared trunk.
            n_hidden (int, optional): Hidden layer width.
            dropout_rate (float, optional): Dropout applied in the shared trunk.
            distribution (str, optional): Latent distribution; `"ln"` applies a softmax to
                the sample (logistic-normal), otherwise the raw Gaussian sample is returned.
            var_eps (float, optional): Additive floor for the variance.
            var_activation (Callable, optional): Activation mapping raw variance output to a
                positive variance. Defaults to `torch.exp`.
            return_dist (bool, optional): If True, `forward` returns `(dist, latent)` instead
                of `(mean, var, latent)`.
            **kwargs: Additional keyword arguments forwarded to `FCLayers`.
        """
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
        """
        Args:
            x (Tensor): Input features, shape (batch, n_input).
            *cat_list (Tensor): Categorical covariate tensors forwarded to `FCLayers`.

        Returns:
            tuple: `(dist, latent)` if `return_dist=True`, else `(mean, variance, latent)`.
        """
        q = self.encoder(x, *cat_list)

        q_m = self.mean_encoder(q)

        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps

        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent

        return q_m, q_v, latent

class EncoderHVAE2(nn.Module):
    """
    Encodes a library-size/component distribution: per-GMM-component mean and variance for
    the library size, plus a softmax over components (`q_pi`).
    """

    def __init__(self, n_input, n_output, n_cat_list = None, n_layers = 1, n_hidden = 128, dropout_rate = 0.1, distribution = "normal", var_eps = 1e-4, return_dist = False,
    var_activation = None):
        """
        Args:
            n_input (int): Input feature dimensionality.
            n_output (int): Number of GMM components.
            n_cat_list (list[int], optional): Category counts for each categorical covariate.
            n_layers (int, optional): Number of hidden layers in the shared trunk.
            n_hidden (int, optional): Hidden layer width.
            dropout_rate (float, optional): Dropout applied in the shared trunk.
            distribution (str, optional): Unused by `forward` (kept for interface parity
                with `EncoderHVAE`).
            var_eps (float, optional): Additive floor for the variance.
            return_dist (bool, optional): Unused by `forward` (kept for interface parity).
            var_activation (Callable, optional): Activation mapping raw variance output to a
                positive variance. Defaults to `torch.exp`.
        """
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
        """
        Args:
            x (Tensor): Input features, shape (batch, n_input).
            *cat_list (Tensor): Categorical covariate tensors forwarded to `FCLayers`.

        Returns:
            tuple: `(q_pi, mean, variance)`, where `q_pi` is a softmax over GMM components.
        """
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        q_pi = self.mean_encoder2(q)

        return torch.nn.functional.softmax(q_pi), q_m, q_v

class LearnableGMMPrior(nn.Module):
    """A learnable Gaussian-mixture prior over a low-dimensional (e.g. library-size) space.

    With `shared_base=True`, each component's mean and log-std are a shared base plus a
    per-component (per-batch/dataset) deviation from it; with `shared_base=False`, every
    component has fully independent parameters.
    """

    def __init__(self, latent_dim, num_components, shared_base=True):
        """
        Args:
            latent_dim (int): Dimensionality of each Gaussian component.
            num_components (int): Number of mixture components.
            shared_base (bool, optional): Parameterize components as a shared base plus a
                per-component deviation (True) or independently (False).
        """
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.shared_base = shared_base
        self.pi_logits = nn.Parameter(torch.ones(num_components))
        if shared_base:
            self.base_mu = nn.Parameter(torch.randn(1, latent_dim))
            self.delta_mu = nn.Parameter(torch.randn(num_components, latent_dim) * 0.01)
            self.base_log_sigma = nn.Parameter(torch.zeros(1, latent_dim))
            self.delta_log_sigma = nn.Parameter(torch.zeros(num_components, latent_dim))
        else:
            self.mu = nn.Parameter(torch.randn(num_components, latent_dim))
            self.log_sigma = nn.Parameter(torch.zeros(num_components, latent_dim))

    def forward(self):
        """Returns: tuple: `(mu, sigma, pi)` — component means, std devs, and mixture weights."""
        pi = torch.softmax(self.pi_logits, dim=0)
        if self.shared_base:
            mu = self.base_mu + self.delta_mu
            sigma = torch.exp(self.base_log_sigma + self.delta_log_sigma)
        else:
            mu = self.mu
            sigma = torch.exp(self.log_sigma)
        return mu, sigma, pi


class DecoderHVAE3(nn.Module):
    """Decodes a latent `z` (plus library size and covariates) into ZINB parameters."""

    def __init__(
        self, n_input, n_output, n_cat_list = None, n_layers = 1, n_hidden = 128, inject_covariates = True,
        use_batch_norm = False, use_layer_norm = False, scale_activation = "softmax",
    ):
        """
        Args:
            n_input (int): Latent dimensionality.
            n_output (int): Number of genes to decode.
            n_cat_list (list[int], optional): Category counts for each categorical covariate.
            n_layers (int, optional): Number of hidden layers in the shared trunk.
            n_hidden (int, optional): Hidden layer width.
            inject_covariates (bool, optional): Whether covariates are concatenated into
                every hidden layer, or only the first.
            use_batch_norm (bool, optional): Use BatchNorm in the shared trunk.
            use_layer_norm (bool, optional): Use LayerNorm in the shared trunk.
            scale_activation (str, optional): Activation for `px_scale`: `"softmax"` or
                `"softplus"`.
        """
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
        """
        Args:
            dispersion (str): `"gene-cell"` to compute a per-gene-per-cell dispersion `px_r`
                from the decoder, or anything else to skip it (caller supplies `px_r`).
            z (Tensor): Latent representation, shape (batch, n_input).
            library (Tensor): Log library size, shape (batch, 1), used to scale `px_rate`.
            covariates (Tensor): Covariate tensor forwarded to the decoder trunk.
            *cat_list (Tensor): Categorical covariate tensors forwarded to `FCLayers`.

        Returns:
            tuple: `(px_scale, px_r, px_rate, px_dropout)` ZINB parameters.
        """
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = torch.nn.functional.relu(self.px_dropout_decoder(px))
        px_rate = torch.nn.functional.relu(torch.exp(library) * px_scale)
        px_r =  torch.nn.functional.softplus(self.px_r_decoder(px)) + 1e-4 if dispersion == "gene-cell" else None


        return px_scale, px_r, px_rate, px_dropout
        
        
@torch.no_grad()
def _mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """
    Unbiased multi-scale RBF-kernel maximum mean discrepancy between two sample sets.

    Args:
        x (Tensor): First sample set, shape (m, D).
        y (Tensor): Second sample set, shape (n, D).
        sigmas (tuple[float], optional): RBF kernel bandwidths to sum over.

    Returns:
        Tensor: Scalar MMD estimate.
    """
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
    """
    GMVAE-style latent loss: a component-marginalized Gaussian KL term (`kl_z_given_c` +
    `kl_c`) plus an MMD term between samples from `q(z)` and the GMM prior, which helps
    stabilize training compared to the analytic KL alone.

    Args:
        z_mu (Tensor): Posterior mean, shape (B, D).
        z_logvar (Tensor): Posterior log-variance, shape (B, D).
        logits_q_c (Tensor): Unnormalized posterior component logits, shape (B, K).
        p_mu (Tensor): Prior component means, shape (K, D).
        p_logvar (Tensor): Prior component log-variances, shape (K, D).
        p_pi (Tensor): Unnormalized prior component weights, shape (K,).
        mmd_weight (float, optional): Weight applied to the MMD term in the returned total.
        mmd_sigmas (tuple[float], optional): RBF kernel bandwidths for the MMD term.
        reduction (str, optional): `"mean"` or `"sum"` over the batch for the KL terms.
        sample_for_mmd (bool, optional): Sample `z` from `q(z)` for the MMD term (True), or
            use `z_mu` directly (False).

    Returns:
        tuple: `(total, pieces)` where `total = kl_z_given_c + kl_c + mmd_weight * mmd`
            (reduced per `reduction`), and `pieces` is a dict with the individual
            'kl_z_given_c', 'kl_c', and 'mmd' terms.
    """
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
    """A learnable Gaussian-mixture prior over the main latent space `z` (log-var parameterized).

    With `shared_base=True`, each component's mean and log-variance are a shared base plus
    a per-component (per-batch/dataset) deviation from it; with `shared_base=False`, every
    component has fully independent parameters.
    """

    def __init__(self, z_dim, num_components, shared_base=True):
        """
        Args:
            z_dim (int): Dimensionality of the latent space.
            num_components (int): Number of mixture components.
            shared_base (bool, optional): Parameterize components as a shared base plus a
                per-component deviation (True) or independently (False).
        """
        super().__init__()
        self.z_dim = z_dim
        self.K = num_components
        self.shared_base = shared_base
        if shared_base:
            self.base_mu = nn.Parameter(torch.randn(1, z_dim) * 0.01)
            self.delta_mu = nn.Parameter(torch.randn(self.K, z_dim) * 0.01)
            self.base_logvar = nn.Parameter(torch.zeros(1, z_dim))
            self.delta_logvar = nn.Parameter(torch.zeros(self.K, z_dim))
        else:
            self.mu = nn.Parameter(torch.randn(self.K, z_dim) * 0.01)
            self.logvar = nn.Parameter(torch.zeros(self.K, z_dim))
        self.logits_pi = nn.Parameter(torch.zeros(self.K))

    def forward(self):
        """Returns: tuple: `(mu, logvar, pi)` — component means, log-variances, and mixture weights."""
        if self.shared_base:
            p_mu = self.base_mu + self.delta_mu
            p_logvar = self.base_logvar + self.delta_logvar
        else:
            p_mu = self.mu
            p_logvar = self.logvar
        p_pi = F.softmax(self.logits_pi, dim=0)
        return p_mu, p_logvar, p_pi


class HVAE_Lightning(pl.LightningModule):
    """
    A hierarchical VAE (PyTorch Lightning module) for single-cell counts: a Gaussian latent
    `z` with a learnable GMM prior, a GMM-modeled library size, a ZINB decoder, and an
    auxiliary marker-gene prediction head, trained with an annealed KL weight and an MMD
    regularizer on the latent space.
    """

    def __init__(
        self, n_input, n_latent=10, n_hidden=128, n_layers=1,
        encode_covariates=False, ln=False,
        lr=1e-3,
        kl_weight=0.001,             # will be ANNEALED; this is starting value
        mmd_weight=0.1,
        n_batch=9, num_components=9,
        reconstruction_weight=1.0,
        gene_factor=50,
        prediction_genes=18,
        random_assignment=False,
        dispersion="gene-cell",           # "gene" or "gene-cell"
        gene_dropout_shared=False,   # if True, gate is also per-gene (optional)
        kl_anneal_warmup_epochs=20,  # linear warm-up length
        kl_anneal_max=0.5,           # plateau value after warm-up
        use_gene_contrastive=False,  # if True, replace the MSE marker-gene head with contrastive learning
        gene_contrastive_temperature=0.5,
        gene_contrastive_bandwidth=None,
        gene_contrastive_cross_batch_only=True,
        library_kl="original",       # "original", "marginalized" or "batch"
        shared_base_prior=True,      # shared-base + per-component-deviation GMM priors
    ):
        """
        Args:
            n_input (int): Number of genes.
            n_latent (int, optional): Dimensionality of the latent space `z`.
            n_hidden (int, optional): Unused directly here; hidden width is fixed per
                sub-module (kept for interface parity).
            n_layers (int, optional): Number of hidden layers in the encoder/decoder trunks.
            encode_covariates (bool, optional): Concatenate batch covariate embeddings into
                the encoder input.
            ln (bool, optional): Unused by this model (kept for interface/CLI parity).
            lr (float, optional): Adam learning rate.
            kl_weight (float, optional): Starting KL weight for annealing.
            mmd_weight (float, optional): Weight of the MMD term in the latent loss.
            n_batch (int, optional): Number of batches/domains, sized for the covariate
                embedding and per-batch library baseline.
            num_components (int, optional): Number of GMM components for the latent prior
                and the library-size distribution.
            reconstruction_weight (float, optional): Weight of the ZINB reconstruction loss.
            gene_factor (int, optional): Weight of the auxiliary marker-gene MSE loss.
            prediction_genes (int, optional): Number of marker genes predicted by the
                auxiliary head.
            random_assignment (bool, optional): Sample a random GMM component for the
                library size in `forward`, instead of using the observed batch label.
            dispersion (str, optional): `"gene-cell"` uses the decoder's per-cell dispersion;
                `"gene"` uses a learned per-gene dispersion parameter instead.
            gene_dropout_shared (bool, optional): When `dispersion="gene"`, also use a
                learned per-gene (rather than decoder-produced) dropout gate.
            kl_anneal_warmup_epochs (int, optional): Epochs over which the KL weight is
                linearly annealed from `kl_weight` to `kl_anneal_max`.
            kl_anneal_max (float, optional): KL weight after annealing completes.
            use_gene_contrastive (bool, optional): If True, replace the MSE-based
                marker-gene prediction head with an NT-Xent contrastive loss
                (`gene_expression_contrastive_loss`) computed directly on raw (unbinned)
                log-normalized marker-gene expression, using a learned projection head
                instead of a regression head. Requires the `AnnDataCollator` to be built
                with `bin_marker_genes=False` so `gene` targets are left unbinned. Only
                cross-batch pairs are compared (same-batch pairs are excluded from both
                positives and negatives), so the loss reflects batch-integration signal
                rather than within-batch similarity.
            gene_contrastive_temperature (float, optional): NT-Xent softmax temperature,
                only used when `use_gene_contrastive=True`.
            gene_contrastive_bandwidth (float, optional): Kernel bandwidth for the
                expression-similarity weighting; only used when `use_gene_contrastive=True`.
                Defaults to each gene's per-batch std (see `_nt_xent_weighted`).
            gene_contrastive_cross_batch_only (bool, optional): Only used when
                `use_gene_contrastive=True`. If True (default), the contrastive loss only
                compares cross-batch pairs (batch-integration signal). If False, all pairs
                are compared by marker-gene-expression similarity regardless of batch,
                favoring cell-type structure over batch alignment.
            library_kl (str, optional): Library-size KL variant. `"original"` uses
                `kl_divergence_gmm` (sums every component's KL, with the variance passed
                as std dev); `"marginalized"` weights each component's KL by `q_pi`
                (`kl_divergence_gmm_marginalized`); `"batch"` only uses the KL of each
                cell's batch-label component, matching how `forward` samples the library
                (`kl_divergence_batch_component`).
            shared_base_prior (bool, optional): Parameterize the library and `z` GMM
                priors as a shared base plus per-component deviations (True) or with
                independent per-component parameters (False).
        """
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

        if library_kl not in ("original", "marginalized", "batch"):
            raise ValueError(f"library_kl must be 'original', 'marginalized' or 'batch', got {library_kl!r}")
        self.library_kl = library_kl

        self.gmm_prior = LearnableGMMPrior(1, num_components=num_components,
                                           shared_base=shared_base_prior).to(device)

        self.c_head = nn.Sequential(
            nn.Linear(n_latent, 256), nn.ReLU(),
            nn.Linear(256, num_components)
        )
        self.gmm_prior_z = LearnableGMMPriorZ(n_latent, num_components,
                                              shared_base=shared_base_prior).to(device)
        self.mmd_weight = mmd_weight

        self.use_gene_contrastive = use_gene_contrastive
        self.gene_contrastive_temperature = gene_contrastive_temperature
        self.gene_contrastive_bandwidth = gene_contrastive_bandwidth
        self.gene_contrastive_cross_batch_only = gene_contrastive_cross_batch_only

        if self.use_gene_contrastive:
            self.gene_contrastive_head = nn.Sequential(
                nn.Linear(n_latent, n_latent), nn.ReLU(),
                nn.Linear(n_latent, n_latent),
            )
        else:
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
        """
        Args:
            labels (Tensor): Integer class labels, shape (batch,).
            num_classes (int): Number of classes.

        Returns:
            Tensor: One-hot encoding, shape (batch, num_classes).
        """
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


    def get_dataset_baseline(self, lib_sizes, batch_data, n_batch):
        """
        Computes and registers the per-batch mean log library size as a buffer
        (`loglib_baseline_per_batch`), used as an offset for the sampled library size
        in `forward`.

        Args:
            lib_sizes (array-like): Per-cell library size (total raw counts), shape
                (n_cells,).
            batch_data (array-like): Integer batch id per cell, shape (n_cells,).
            n_batch (int): Total number of batches.
        """
        lib_sizes = np.asarray(lib_sizes)
        batch_data = np.asarray(batch_data)
        lib_b = torch.empty(n_batch)
        for b in range(n_batch):
            s = torch.tensor(lib_sizes[batch_data == b]).clamp_min(1.)
            lib_b[b] = torch.log(s).mean()
        self.register_buffer("loglib_baseline_per_batch", lib_b)     # shape [n_batch]

        return

    # ---------- forward ----------
    def forward(self, x, gene, covariates=None):
        """
        Encodes `x` to a latent `z` and library size, then decodes ZINB parameters and an
        auxiliary marker-gene prediction.

        Args:
            x (Tensor): Input counts, shape (batch, n_input).
            gene (array-like): Unused by `forward` itself; kept for call-site parity with
                `compute_loss`, which takes the marker-gene targets separately.
            covariates (Tensor, optional): Integer batch labels, shape (batch,). Used both
                as the covariate embedding input and to select/offset the library size.

        Returns:
            dict: Model outputs, including latent stats (`z_mu`, `z_var`, `z`), library-size
                stats (`l_pi`, `l_mu`, `l_var`, `library`, `ql`), decoder outputs (`px_scale`,
                `px_r_dec`, `px_rate`, `px_dropout_dec`), the component logits `logits_q_c`,
                and the auxiliary `gene_pred` (marker-gene bin regression, or a projection
                embedding for contrastive learning when `use_gene_contrastive=True`).
        """
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
        gene_pred = (
            self.gene_contrastive_head(z_mu)
            if self.use_gene_contrastive
            else self.pred_genes(z_mu)
        )

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
    def compute_loss(self, x, output, gene, batch_labels=None):
        """
        Computes the total training loss: a ZINB reconstruction term, an annealed KL term
        (latent `z` + library size, via `gmvae_kl_with_mmd` and the `library_kl` variant), and an
        auxiliary marker-gene term — an MSE regression loss against binned marker-gene
        expression by default, or an NT-Xent contrastive loss directly on raw
        log-normalized marker-gene expression when `self.use_gene_contrastive=True`.

        Args:
            x (Tensor): Input counts, shape (batch, n_input).
            output (dict): The dict returned by `forward`.
            gene (array-like): Marker-gene targets, shape (batch, prediction_genes).
            batch_labels (Tensor, optional): Integer batch/domain id per sample, shape
                (batch,). Required when `self.library_kl="batch"`, to select each cell's
                library component. Otherwise only used when `self.use_gene_contrastive=True`
                and `self.gene_contrastive_cross_batch_only=True`, to restrict the
                contrastive loss to cross-batch pairs (see `gene_expression_contrastive_loss`).

        Returns:
            dict: `{"train_loss": ..., "recon": ..., "kl_z_total": ..., "kl_l": ...,
                "gene_aux": ..., "kl_weight_eff": ..., "mmd": ...}`.
        """
        eps = 1e-5
        z_mu, z_var = output["z_mu"], output["z_var"]
        l_pi, l_mu, l_var = output["l_pi"], output["l_mu"], output["l_var"]
        px_rate = output["px_rate"]
        px_dropout_dec = output["px_dropout_dec"]
        gene_pred = output["gene_pred"]
        logits_q_c = output["logits_q_c"]
        gene = torch.as_tensor(gene, device=x.device)


        p_mu_l, p_sigma_l, p_pi_l = self.gmm_prior()
        if self.library_kl == "marginalized":
            kl_l = kl_divergence_gmm_marginalized(
                l_pi, l_mu, torch.sqrt(l_var), p_mu_l, p_sigma_l, p_pi_l
            ).mean()
        elif self.library_kl == "batch":
            kl_l = kl_divergence_batch_component(
                l_mu, torch.sqrt(l_var), p_mu_l, p_sigma_l, batch_labels
            ).mean()
        else:
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


        if self.use_gene_contrastive:
            gene_aux_loss = gene_expression_contrastive_loss(
                gene_pred, gene.float(),
                batch_labels=batch_labels if self.gene_contrastive_cross_batch_only else None,
                temperature=self.gene_contrastive_temperature,
                bandwidth=self.gene_contrastive_bandwidth,
            )
        else:
            ce_loss = torch.nn.MSELoss()
            gene_aux_loss = ce_loss(torch.clip(gene_pred.float(), max=2), gene.float())

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

        zzn = -zinb.log_prob(x).mean(dim=-1)  # [B]

        recon = zzn.mean() * self.reconstruction_weight

        kl_w = self._current_kl_weight()
        loss = (recon + kl_w * (kl_z_total + kl_l)) * 0.1 + self.gene_factor * gene_aux_loss

        logs = {
            "recon": recon.detach(),
            "kl_z_total": (kl_w * kl_z_total).detach(),
            "kl_l": (kl_w * kl_l).detach(),
            "gene_aux": gene_aux_loss.detach(),
            "kl_weight_eff": torch.tensor(kl_w, device=x.device),
            "mmd": pieces["mmd"],
        }
        return {"train_loss": loss, **logs}


    def training_step(self, batch, batch_idx):
        """
        Lightning training step: runs `forward` + `compute_loss` on `batch` and logs the
        loss components.

        Args:
            batch (tuple): `(x, class_labels, covariates, gene)` as produced by the
                `AnnDataCollator`.
            batch_idx (int): Batch index (unused; required by the Lightning interface).

        Returns:
            Tensor: The scalar training loss for this batch.
        """
        x, _, covariates, gene = batch if self.encode_covariates else (batch, None)

        output = self(x, gene, covariates)
        loss_dict = self.compute_loss(x, output, gene, batch_labels=covariates)

        self.log("train_loss", loss_dict["train_loss"], prog_bar=True)
        self.log("train/recon", loss_dict["recon"], prog_bar=True)
        self.log("train/kl_z_total", loss_dict["kl_z_total"])
        self.log("train/kl_l", loss_dict["kl_l"])
        self.log("train/mmd", loss_dict["mmd"])
        self.log("train/gene_aux", loss_dict["gene_aux"])
        self.log("train/kl_weight_eff", loss_dict["kl_weight_eff"], prog_bar=True)
        return loss_dict["train_loss"]

    def configure_optimizers(self):
        """Lightning hook: Adam optimizer with a step-decay LR schedule (halves every 7 epochs)."""
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "lr_scheduler": {"scheduler": scheduler}
        }

