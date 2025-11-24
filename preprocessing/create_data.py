import os
import tempfile
import numpy as np
import scanpy as sc
import seaborn as sns
import anndata as ad
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def convert_object_to_int_with_nan(adata):
    """
    Converts all columns in .var of an AnnData object that have dtype 'object'
    but contain boolean values (with possible NaN) into integers (0 and 1),
    preserving NaN values.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object to process.

    Returns:
    --------
    AnnData
        The modified AnnData object with converted columns.
    """
    import numpy as np

    # Identify object dtype columns
    object_columns = adata.var.select_dtypes(include=["object"]).columns

    # Iterate through object columns and check if they are boolean or NaN
    for col in object_columns:
        non_nan_values = adata.var[col].dropna()
        if non_nan_values.isin([True, False]).all():
            # Convert boolean to integer, preserving NaN
            adata.var[col] = adata.var[col].apply(lambda x: int(x) if pd.notna(x) else np.nan)
            print(f"Converted column '{col}' from boolean (object) to int, preserving NaN.")
        else:
            print(f"Column '{col}' has non-boolean values. Skipped.")

    return adata
    


def ensembl_to_hugo_in_adata(
    adata,
    source="var_names",     # "var_names" or a column name in adata.var
    target_col="gene_symbol",
    species="human",
    strip_version=True,
    overwrite_var_names=False
):
    import mygene
    
    mg = mygene.MyGeneInfo()


    if source == "var_names":
        ensembl_ids = pd.Index(adata.var_names)
    else:
        if source not in adata.var.columns:
            raise ValueError(f"source='{source}' not found in adata.var.columns.")
        ensembl_ids = adata.var[source].astype(str)

    if strip_version:
        base_ids = ensembl_ids.to_series().str.replace(r"\.\d+$", "", regex=True)
    else:
        base_ids = ensembl_ids.to_series()


    query_res = mg.querymany(
        base_ids.tolist(),
        scopes="ensembl.gene",
        fields="symbol",
        species=species,
        as_dataframe=True
    )


    symbol_map = {}
    for idx, row in query_res.iterrows():
        if isinstance(row, pd.Series):
            symbol = row.get("symbol", None)
        else:
            symbol = None
        if isinstance(symbol, str):
            symbol_map[idx] = symbol


    mapped_symbols = base_ids.map(symbol_map).astype("object")
    adata.var[target_col] = mapped_symbols.values


    if overwrite_var_names:

        new_var_names = adata.var[target_col].where(
            ~adata.var[target_col].isna(),
            other=ensembl_ids
        )

        new_var_names = pd.Index(new_var_names)
        if not new_var_names.is_unique:
            counts = {}
            unique_names = []
            for name in new_var_names:
                if name not in counts:
                    counts[name] = 0
                    unique_names.append(name)
                else:
                    counts[name] += 1
                    unique_names.append(f"{name}.dup{counts[name]}")
            new_var_names = pd.Index(unique_names)

        adata.var_names = new_var_names

    return adata

def preprocessing(marker_gene_list, dataset, full_name, samples_to_remove = None, test_size = 0.1, n_top_genes=10000, batch_key="batch",layer = "counts", flavor='seurat', no_hvg = False):

    """
    Preprocess single-cell RNA-seq data for downstream analysis.


    Parameters
    ----------
    marker_gene_list : str
        Path to a file containing a list of marker gene names (one per line).
    dataset : str
        Filename of the input dataset (must be readable by Scanpy, e.g., .h5ad).
    full_name : str
        Filename for saving the full processed dataset.
    ref_name : str
        Filename for saving the reference (training) dataset.
    query_name : str
        Filename for saving the query (test) dataset.
    samples_to_remove : list of str, optional
        List of sample IDs to exclude from the dataset (default: None).
    test_size : float, optional
        Fraction of samples to hold out as the test/query dataset (default: 0.1).
    n_top_genes : int, optional
        Number of highly variable genes to select (default: 10000).
    batch_key : str, optional
        Column in `adata.obs` used as batch annotation when finding HVGs (default: "dataset").
    layer : str, optional
        Name of the layer in `adata.layers` to use for HVG selection (default: "counts").

    Outputs
    -------
    Writes three AnnData objects:
        - ref_name   : reference (train) dataset
        - query_name : query (test) dataset
        - full_name  : full processed dataset (HVGs + marker genes)

    """
    

    gene_symbol = [f[:-1] for f in open(marker_gene_list).readlines()]
    adata = sc.read(os.path.join(dataset))

    
    if samples_to_remove != None: 
        adata = adata[~adata.obs['SampleID'].isin(samples_to_remove)]
        
    X_train, X_test = train_test_split(range(0,adata.shape[0]), test_size = test_size)

    if no_hvg == False:
        subset_indices = [i for i, gene in enumerate(adata.var_names) if gene in set(gene_symbol)]
        adata_selected = adata[:, subset_indices]


        try:
            sc.pp.highly_variable_genes(adata, flavor = flavor, layer = layer, n_top_genes=n_top_genes, batch_key=batch_key, subset=False)
        except:
            sc.pp.highly_variable_genes(adata, flavor = flavor, n_top_genes=n_top_genes, batch_key=batch_key, subset=False)

        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        adata_hvg = adata_hvg.copy()
        adata_selected = adata_selected.copy()

        adata_selected.obs = adata_hvg.obs.copy()
        adata = ad.concat([adata_hvg, adata_selected], axis=1)
        adata.obs = adata_hvg.obs.copy()

        duplicates = adata.var_names.duplicated()
        adata = adata[:, ~duplicates].copy()

        assert len(adata.var_names) == len(list(set(adata.var_names)))
        

    adata_ref = adata[X_train].copy()
    adata_query = adata[X_test].copy()
    
    adata_ref.write(filename= full_name+ "_ref.h5ad")
    adata_query.write(filename= full_name+ "_query.h5ad")
    adata.write(filename=  full_name + ".h5ad")

    return


