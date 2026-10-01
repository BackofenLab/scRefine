from collections import namedtuple
import os

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset

from .sampler import DomainBatchSampler


def create_bins(data, n_bins=10):
    """
    Bin a 1D array into equal-width bins spanning [min(data), max(data)].

    Bins are returned as values scaled to [0, 1).
    """
    data = np.asarray(data)

    if data.size == 0:
        return np.array([], dtype=float)

    # Avoid unstable binning when all values are identical.
    if np.min(data) == np.max(data):
        return np.zeros(data.shape, dtype=float)

    bin_edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
    bin_indices = np.digitize(data, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bin_indices / n_bins


def create_dataset_specific_bins(data, dataset_labels, n_bins=10):
    """
    Create bins independently within each dataset.

    Args:
        data (np.ndarray): One-dimensional values to bin.
        dataset_labels (np.ndarray): Dataset label for every value.
        n_bins (int): Number of bins per dataset.

    Returns:
        np.ndarray: Dataset-specific bin assignments in the original cell order.
    """
    data = np.asarray(data)
    dataset_labels = np.asarray(dataset_labels)

    bin_indices = np.zeros(data.shape, dtype=float)

    for dataset_label in np.unique(dataset_labels):
        dataset_mask = dataset_labels == dataset_label

        bin_indices[dataset_mask] = create_bins(
            data[dataset_mask],
            n_bins=n_bins,
        )

    return bin_indices


def _print_bin_statistics(gene_symbol, binned_expression, dataset_labels, n_bins):
    """Prints, per batch, how many cells fall into each bin for one marker gene."""
    bin_ids = np.round(binned_expression * n_bins).astype(int)

    print(f"Bin counts for marker gene '{gene_symbol}':")
    for dataset_label in np.unique(dataset_labels):
        mask = dataset_labels == dataset_label
        counts = np.bincount(bin_ids[mask], minlength=n_bins)
        counts_str = ", ".join(
            f"bin {bin_idx}: {count}" for bin_idx, count in enumerate(counts)
        )
        print(f"  batch '{dataset_label}': {counts_str}")


class IndexDataset(Dataset):
    """
    Dataset over the cells of an AnnData object that yields only the cell index; the
    `AnnDataCollator` turns a list of indices into a batch, so the expression matrix can
    stay sparse and is sliced once per batch instead of once per cell.
    """

    def __init__(self, n_cells):
        self.n_cells = n_cells

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return idx


class AnnDataCollator:
    """
    `collate_fn` that turns a list of cell indices into a training batch. Pickleable, so
    it works with multiple DataLoader workers.

    Precomputes the marker-gene targets once: per-dataset binned log1p expression when
    `bin_marker_genes=True` (for the MSE regression head), or raw log1p expression when
    False (for the marker-gene contrastive head). Only the marker-gene columns are
    densified; the full expression matrix stays sparse and is sliced per batch.
    """

    def __init__(self, adata, batch_ids, batch_labels, marker_genes, bin_marker_genes=True, n_bins=10):
        """
        Args:
            adata (AnnData): Input data with raw counts in `.X`.
            batch_ids (np.ndarray): Integer batch/domain id per cell, shape (n_cells,).
            batch_labels (np.ndarray): Batch/domain label per cell, shape (n_cells,); used
                to bin per dataset and to label the printed bin statistics.
            marker_genes (list): Marker gene symbols; symbols not in `adata.var_names` are
                skipped (and reported).
            bin_marker_genes (bool, optional): Bin the marker-gene targets per dataset
                instead of leaving them as raw log1p expression.
            n_bins (int, optional): Number of bins per dataset when binning.
        """
        self.X = adata.X
        self.batch_ids = torch.as_tensor(batch_ids, dtype=torch.long)

        # First occurrence of each gene symbol, matching `list.index` semantics.
        gene_position = {}
        for i, gene in enumerate(adata.var_names):
            gene_position.setdefault(gene, i)

        valid_genes = []
        for gene in marker_genes:
            if gene not in gene_position:
                print(f"skipped: {gene}")
                continue
            valid_genes.append(gene)
        if not valid_genes:
            raise ValueError("None of the marker genes are present in adata.var_names.")

        # Slice and densify only the marker-gene columns, not the whole matrix.
        marker_expression = self.X[:, [gene_position[g] for g in valid_genes]]
        if issparse(marker_expression):
            marker_expression = marker_expression.toarray()
        marker_expression = np.log1p(np.asarray(marker_expression))

        if bin_marker_genes:
            columns = []
            for col, gene in enumerate(valid_genes):
                binned = create_dataset_specific_bins(
                    marker_expression[:, col], batch_labels, n_bins=n_bins
                )
                _print_bin_statistics(gene, binned, batch_labels, n_bins=n_bins)
                columns.append(binned)
            self.gene_targets = np.stack(columns, axis=-1)
        else:
            self.gene_targets = marker_expression

        self.n_marker_genes = len(valid_genes)

    def __call__(self, batch_indices):
        """
        Args:
            batch_indices (list[int]): Cell indices selected for this batch.

        Returns:
            tuple: `(X, None, batch_ids, gene_targets)` - dense float32 counts, an unused
                class-label slot (kept for the model's batch layout), integer batch ids,
                and the marker-gene targets as a NumPy array.
        """
        idx = np.asarray(batch_indices, dtype=np.int64)

        X_batch = self.X[idx]
        if issparse(X_batch):
            X_batch = X_batch.toarray()
        X_tensor = torch.as_tensor(X_batch, dtype=torch.float32)

        return X_tensor, None, self.batch_ids[idx], self.gene_targets[idx]


DataInfo = namedtuple("DataInfo", ["batch_ids", "lib_sizes", "n_genes", "n_batches", "n_marker_genes"])


def build_dataloaders(adata, domain_key, marker_genes, batch_size=33, p_intra_domain=0.7,
                      bin_marker_genes=True, num_workers=None, device=None):
    """
    Builds the training and embedding DataLoaders for the HVAE.

    The training loader draws domain-aware minibatches with `DomainBatchSampler` (roughly
    `p_intra_domain` of each minibatch from one batch/domain, the rest from the other
    domains). The embedding loader iterates over all cells in their original order, so its
    output can be joined back to `adata.obs_names`.

    Args:
        adata (AnnData): Input data with raw counts in `.X`.
        domain_key (str): Column in `adata.obs` holding the batch/domain label.
        marker_genes (list): Marker gene symbols used as auxiliary targets.
        batch_size (int, optional): Minibatch size for both loaders.
        p_intra_domain (float, optional): Fraction of each training minibatch drawn from
            the same domain.
        bin_marker_genes (bool, optional): See `AnnDataCollator`.
        num_workers (int, optional): Training-loader workers; defaults to
            `min(4, os.cpu_count())`.
        device (torch.device, optional): Device for the sampler's domain-id tensor.

    Returns:
        tuple: `(train_loader, embed_loader, info)`, where `info` is a `DataInfo` with the
            integer batch id and raw library size of every cell, the number of genes, the
            number of batches, and the number of marker genes found in the data.
    """
    # Sorted unique labels -> 0..n_batches-1; the same ids are used by the sampler, the
    # model's batch embedding and the per-batch library-size baseline.
    cell_labels = adata.obs[domain_key].to_numpy()
    batch_names, batch_ids = np.unique(cell_labels, return_inverse=True)

    collator = AnnDataCollator(adata, batch_ids, cell_labels, marker_genes,
                               bin_marker_genes=bin_marker_genes)
    dataset = IndexDataset(adata.n_obs)

    sampler = DomainBatchSampler(
        domain_ids=torch.as_tensor(batch_ids, dtype=torch.long).to(device),
        batch_size=batch_size,
        p_intra_domain=p_intra_domain,
        device=device,
    )

    if num_workers is None:
        num_workers = min(4, os.cpu_count())

    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                              collate_fn=collator, pin_memory=True)
    embed_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    info = DataInfo(
        batch_ids=batch_ids,
        lib_sizes=np.asarray(adata.X.sum(axis=1)).ravel(),
        n_genes=adata.n_vars,
        n_batches=len(batch_names),
        n_marker_genes=collator.n_marker_genes,
    )
    return train_loader, embed_loader, info
