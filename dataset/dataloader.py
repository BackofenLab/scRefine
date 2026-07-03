import torch
from .sampler import ConcordSampler
from .anndataset import AnnDataset
from .knn import Neighborhood
from .anndata_utils import get_adata_basis
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc
import os
import logging
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def create_bins(data, n_bins = 10):
    """
    Bin a 1D array of values into `n_bins` equal-width bins spanning [min(data), max(data)].

    Args:
        data (np.ndarray): 1D array of values to bin.
        n_bins (int, optional): Number of bins. Defaults to 10.

    Returns:
        np.ndarray: Bin index for each value, scaled to [0, 1) by dividing by 10.
    """
    bin_edges = np.linspace(np.min(data), np.max(data), n_bins + 1)
    bin_indices = np.digitize(data, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bin_indices/10

class AnnDataCollator:
    """
    A callable class to act as the collate_fn for the DataLoader.
    This is pickleable and works with multiple workers.
    """
    def __init__(self, dataset,
                 dataset_key,
                 marker_genes):
        """
        Precomputes per-cell data needed for batching: dense log1p counts, integer-encoded
        batch labels, and (if `marker_genes` is non-empty) binned expression for each marker
        gene, used downstream as auxiliary prediction targets.

        Args:
            dataset (AnnDataset): The dataset whose `adata` provides the source data.
            dataset_key (str): Column in `adata.obs` identifying the batch/dataset of origin.
            marker_genes (list): Gene symbols to bin for auxiliary supervision. May be empty.
        """
        self.dataset = dataset

        data_save = self.dataset.adata.X.toarray()
        batch_save = self.dataset.adata.obs[dataset_key]
        self.count_data = np.log1p(data_save)
        self.batch_dict = list(np.unique(batch_save))
        self.batches = [self.batch_dict.index(a) for a in batch_save]
        self.marker_genes = marker_genes
        gene_data = self.dataset.adata.var_names
        bin_indices = []
        self.prediction_genes = 0

        gene_names = self.marker_genes
        self.data_save = data_save

        self.bin_indices = None

        if len(self.marker_genes) != 0:
        
            for gene_sym in gene_names:
                try:
                    relevant_int = list(gene_data).index(gene_sym)
                    relevant_data = self.count_data[:,relevant_int]
                    bin_data = np.expand_dims(create_bins(relevant_data, n_bins = 10),axis=-1)
                    bin_indices.append(bin_data)
                    self.prediction_genes = self.prediction_genes + 1
                except:
                    print("skipped: " + str(gene_sym))

            self.bin_indices = np.concat(bin_indices, axis = -1)

    def __call__(self, batch_indices):
        """
        Collates a list of dataset indices into a batch of tensors.

        Args:
            batch_indices (list[int]): Sample indices selected for this batch.

        Returns:
            tuple: `(X_tensor, class_tensor, domain_tensor, bins)`, where `X_tensor` is the
                dense input features, `class_tensor` is the class labels (or None),
                `domain_tensor` is the domain/batch labels, and `bins` is the binned marker
                gene expression (or None if no marker genes were configured).
        """
        # batch_indices is a list of indices from __getitem__
        idx = np.array(batch_indices, dtype=np.int64)

        # 1. Efficiently slice the sparse matrix for the whole batch
        X_batch = self.dataset.adata.X[idx]

        bins = None
        if len(self.marker_genes) != 0:
            bins = self.bin_indices[idx]

        # 2. Convert the entire batch slice to dense at once
        if issparse(X_batch):
            X_batch = X_batch.toarray()

        # 3. Convert to tensor
        X_tensor = torch.as_tensor(X_batch, dtype=torch.float32)

        # Prepare other items in the batch
        domain_tensor = self.dataset.domain_labels[idx]

        class_tensor = None
        if self.dataset.class_labels is not None:
            class_tensor = self.dataset.class_labels[idx]

        return (
            X_tensor,
            class_tensor,
            domain_tensor,
            bins,
        )


class DataLoaderManager:
    """
    Manages data loading for CONCORD, including optional preprocessing and sampling.
    This class handles the standard workflow of total-count normalization and log1p transformation.
    """
    def __init__(self, 
                 domain_key, 
                 class_key=None, 
                 covariate_keys=None,
                 feature_list=None,
                 normalize_total=True,  # Simplified parameter
                 log1p=True,            # Simplified parameter
                 batch_size=32, 
                 train_frac=0.9,
                 use_sampler=True,
                 sampler_knn=300, 
                 sampler_emb=None,
                 sampler_domain_minibatch_strategy='proportional',
                 domain_coverage=None,
                 p_intra_knn=0.3, 
                 p_intra_domain=1.0,
                 dist_metric='euclidean',
                 use_faiss=True, 
                 use_ivf=False,
                 ivf_nprobe=8,
                 preload_dense=False,
                 num_workers=None,
                 device=None):
        """
        Configures how AnnData objects are turned into train/val DataLoaders: which obs
        columns to use as domain/class/covariate labels, whether to normalize/log1p the
        data, and the sampling strategy (domain- and k-NN-aware `ConcordSampler`, or plain
        batching when `use_sampler=False`).

        Args:
            domain_key (str): Column in `adata.obs` giving the domain/batch label used for
                sampler-aware batching.
            class_key (str, optional): Column in `adata.obs` giving class labels.
            covariate_keys (list, optional): Columns in `adata.obs` treated as covariates.
            feature_list (list, optional): If given, subsets `adata` to these var names.
            normalize_total (bool, optional): Apply `sc.pp.normalize_total` before use.
            log1p (bool, optional): Apply `sc.pp.log1p` before use.
            batch_size (int, optional): Samples per batch.
            train_frac (float, optional): Fraction of data used for training; the remainder
                is held out for validation (or 1.0 to use everything for both, see
                `anndata_to_dataloader`).
            use_sampler (bool, optional): Whether to build a `ConcordSampler` for
                domain/knn-aware batching, versus plain sequential batching.
            sampler_knn (int, optional): Number of neighbors for the k-NN graph used by the
                sampler.
            sampler_emb (str, optional): Embedding key used to compute the k-NN graph.
            sampler_domain_minibatch_strategy (str, optional): How many minibatches to draw
                per domain: 'proportional', 'equal', or 'coverage'.
            domain_coverage (dict, optional): Per-domain coverage weights, required when
                `sampler_domain_minibatch_strategy='coverage'`.
            p_intra_knn (float, optional): Probability of sampling from k-NN neighborhoods.
            p_intra_domain (float, optional): Probability of sampling within the same domain.
            dist_metric (str, optional): Distance metric for the k-NN index.
            use_faiss (bool, optional): Use FAISS for k-NN search when available.
            use_ivf (bool, optional): Use a FAISS IVF index instead of flat search.
            ivf_nprobe (int, optional): Number of probes for the FAISS IVF index.
            preload_dense (bool, optional): Load the full dense matrix into memory upfront
                instead of using the memory-efficient `AnnDataCollator`.
            num_workers (int, optional): DataLoader worker processes. Defaults to 0 when
                `preload_dense=True`, else `min(4, os.cpu_count())`.
            device (torch.device, optional): Device used for domain-id tensors and sampler.
        """
        self.domain_key = domain_key
        self.class_key = class_key
        self.covariate_keys = covariate_keys
        self.feature_list = feature_list
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.use_sampler = use_sampler
        self.sampler_emb = sampler_emb
        self.sampler_knn = sampler_knn
        self.sampler_domain_minibatch_strategy = sampler_domain_minibatch_strategy
        self.domain_coverage = domain_coverage
        self.p_intra_knn = p_intra_knn
        self.p_intra_domain = p_intra_domain
        self.use_faiss = use_faiss
        self.use_ivf = use_ivf
        self.ivf_nprobe = ivf_nprobe
        self.dist_metric = dist_metric
        
        self.preload_dense = preload_dense
        if num_workers is None:
            # Use 0 worker for in-memory, otherwise use available cores
            self.num_workers = 0 if preload_dense else min(4, os.cpu_count())
        else:
            # Allow user to override
            self.num_workers = num_workers
        logger.info(f"Using {self.num_workers} DataLoader workers.")
        self.device = device


        # Dynamically set based on adata
        self.adata = None
        self.data_structure = None
        self.neighborhood = None
        self.train_sampler = None
        self.val_sampler = None

        self.data_structure = self._get_data_structure()

    
    def _get_data_structure(self):
        """
        Determines the structure of the data to be returned by the dataset.
        This logic is now owned by the manager, not the dataset.
        """
        structure = ['input']
        if self.domain_key is not None:
            structure.append('domain')
        if self.class_key is not None:
            structure.append('class')
        if self.covariate_keys:
            structure.extend(self.covariate_keys)
        structure.append('idx')
        return structure
    

    def compute_neighborhood(self, adata, emb_key):
        """
        Builds and stores a k-NN `Neighborhood` index over the given embedding, for use by
        the sampler when `p_intra_knn > 0`.

        Args:
            adata (AnnData): AnnData object containing the embedding.
            emb_key (str): Key identifying the embedding in `adata.obsm`/`adata.layers`, or
                `'X'` to use `adata.X` directly.
        """
        logger.info(f"Computing k-NN graph using embedding from '{emb_key}'...")
        emb = get_adata_basis(adata, basis=emb_key)
        self.neighborhood = Neighborhood(
            emb=emb, k=self.sampler_knn, use_faiss=self.use_faiss,
            use_ivf=self.use_ivf, ivf_nprobe=self.ivf_nprobe, metric=self.dist_metric
        )


    def build_sampler(self, SamplerClass, indices=None, neighborhood=None):
        """
        Instantiates a batch sampler over the manager's domain ids.

        Args:
            SamplerClass (type): Sampler class to construct (e.g. `ConcordSampler`).
            indices (np.ndarray, optional): Subset of sample indices to restrict the sampler
                to (used for train/val splits). Defaults to using all samples.
            neighborhood (Neighborhood, optional): Precomputed k-NN index to pass to the
                sampler for knn-aware batching.

        Returns:
            Sampler: The constructed sampler instance.
        """
        domain_ids = self.domain_ids if indices is None else self.domain_ids[indices]
        sampler = SamplerClass(
            batch_size=self.batch_size, 
            domain_ids=domain_ids, 
            p_intra_knn=self.p_intra_knn, 
            p_intra_domain=self.p_intra_domain,
            domain_minibatch_strategy=self.sampler_domain_minibatch_strategy,
            domain_coverage=self.domain_coverage,
            neighborhood=neighborhood,
            device=self.device
        )
        return sampler


    def anndata_to_dataloader(self, adata,dataset_key, marker_genes, split_val = False
                                ):
        """
        Converts an AnnData object into PyTorch DataLoaders, applying the configured
        normalization/log1p/feature-subsetting steps and building the domain-aware sampler
        (if enabled) and collator.

        Args:
            adata (AnnData): The input AnnData object.
            dataset_key (str): Column in `adata.obs` identifying the batch/dataset of origin,
                used by the collator to encode domain labels.
            marker_genes (list): Marker gene symbols used for auxiliary binned-expression
                targets in the collator. May be empty.
            split_val (bool, optional): When `train_frac < 1.0`, whether to hold out a
                distinct validation subset (True) or reuse the full dataset for validation
                (False). Defaults to False.

        Returns:
            tuple: `(train_dataloader, val_dataloader, data_structure, data_infos)`, where
                `data_structure` lists the fields returned per sample and `data_infos` is
                `[batches, data, input_size, num_batches, num_marker_genes]` derived from the
                training collator (empty list when `use_sampler=False`).
        """
        self.adata = adata

        if self.normalize_total:
            logger.info("Normalizing total counts per cell...")
            sc.pp.normalize_total(self.adata, target_sum=1e4, inplace=True)
        
        if self.log1p:
            logger.info("Log1p transforming data...")
            sc.pp.log1p(self.adata)

        # Subset features if provided
        if self.feature_list:
            logger.info(f"Filtering features with provided list ({len(self.feature_list)} features)...")
            self.adata = self.adata[:, self.feature_list].copy()     # <-- copy!
            if issparse(self.adata.X):
                self.adata.X.sort_indices()

        self.domain_labels = self.adata.obs[self.domain_key]
        self.domain_ids = torch.tensor(self.domain_labels.cat.codes.values, dtype=torch.long).to(self.device)
        
        dataset = AnnDataset(self.adata, 
                             domain_key=self.domain_key, 
                             class_key=self.class_key, 
                             covariate_keys=self.covariate_keys,
                             preload_dense=self.preload_dense)

        if self.preload_dense:
            logger.info("Loading all data into memory for fast access. This may consume a lot of RAM. If you run out of memory, please set `preload_dense=False`.")
        else:
            logger.info("Using AnnData collator for batching. This is more memory efficient but may slightly reduce speed. Set `preload_dense=True` to load all data into memory for faster access.")
        data_infos = []
        if self.use_sampler:
            if self.train_frac == 1.0:
                if self.p_intra_knn > 0.0:
                    self.compute_neighborhood(self.adata, self.sampler_emb)

                if self.preload_dense:
                    train_collate = None
                else:
                    train_collate = AnnDataCollator(dataset, dataset_key=dataset_key, marker_genes=marker_genes,
                                                    )
                self.train_sampler = self.build_sampler(ConcordSampler, neighborhood=self.neighborhood)
                
                train_dataloader = DataLoader(dataset, batch_sampler=self.train_sampler,
                                              num_workers=self.num_workers,
                                              collate_fn=train_collate,
                                              pin_memory=True)

                batches_out = train_collate.batches
                data_out = train_collate.data_save
                size_ = np.shape(data_out)[1]
                num_batches = len(train_collate.batch_dict)
                number_marker_genes = train_collate.prediction_genes

                data_infos = [batches_out,data_out,size_, num_batches, number_marker_genes]


                val_dataloader = DataLoader(dataset,batch_size = self.batch_size,
                                            shuffle = False, collate_fn=train_collate)

            else:
                train_size = int(self.train_frac * len(dataset))
                indices = np.arange(len(dataset))
                np.random.shuffle(indices)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                train_dataset = dataset.subset(train_indices)
                if split_val == True: val_dataset = dataset.subset(val_indices)
                else: val_dataset = dataset

                self.train_sampler = self.build_sampler(ConcordSampler, indices=train_indices, neighborhood=None)
                if self.preload_dense:
                    train_collate = None
                    val_collate   = None
                else:
                    train_collate = AnnDataCollator(train_dataset, dataset_key=dataset_key, marker_genes=marker_genes,
                                                    )
                    val_collate   = AnnDataCollator(val_dataset, dataset_key=dataset_key, marker_genes=marker_genes,
                                                    )
                train_dataloader = DataLoader(train_dataset, batch_sampler=self.train_sampler,
                                              num_workers=self.num_workers,
                                              collate_fn=train_collate,
                                              pin_memory=True)

                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,       
                    num_workers=self.num_workers,
                    collate_fn=val_collate,
                    pin_memory=True,
                ) # Uniform sampler instead of ConcordSampler for validation

                batches_out = train_collate.batches
                data_out = train_collate.data_save
                size_ = np.shape(data_out)[1]
                num_batches = len(train_collate.batch_dict)
                number_marker_genes = train_collate.prediction_genes

                data_infos = [batches_out,data_out,size_, num_batches, number_marker_genes]


        else: 
            if self.preload_dense:
                train_collate = None
            else:
                train_collate = AnnDataCollator(dataset, dataset_key=dataset_key, marker_genes=marker_genes,
                                                    )
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                          num_workers=self.num_workers,  
                                           collate_fn=train_collate,
                                           pin_memory=True)
            val_dataloader = None

        return train_dataloader, val_dataloader, self.data_structure, data_infos




