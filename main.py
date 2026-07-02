import argparse
import os
import random
import numpy as np
import torch

from clean_data import clean_up
from preprocessing.create_data import preprocessing
from train_hvae import main as train_hvae_main

def print_banner(message: str) -> None:
    print(f"#################################### {message} #########################################")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

def load_marker_genes(marker_gene_path: str) -> list[str]:
    """
    Load marker genes.

    Kept the original line[:-1] behavior to avoid changing functionality.
    You may want to replace this with line.strip() later.
    """
    with open(marker_gene_path) as handle:
        return [line[:-1] for line in handle.readlines()]

def get_obs_fields(obs_fields: list[str], dataset_symbol: str) -> list[str]:
    """
    Add the dataset/batch symbol to obs fields and remove duplicates.

    Kept list(set(...)) behavior from the original code.
    """
    obs_fields = obs_fields.copy()
    obs_fields.append(dataset_symbol)
    return list(set(obs_fields))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Handles data preprocessing, training and cleanup."
    )

    preprocessing_args = parser.add_argument_group("preprocessing")
    preprocessing_args.add_argument("--dataset_path", type=str, help="path to dataset")
    preprocessing_args.add_argument(
        "--marker_genes",
        type=str,
        default="marker_genes.txt",
        help="path to marker genes",
    )
    preprocessing_args.add_argument(
        "--output_h5ad",
        type=str,
        default="output",
        help="preprocessing output h5ad",
    )
    preprocessing_args.add_argument(
        "--npy_dataset",
        type=str,
        default="output.npz",
        help="output npy dataset",
    )
    preprocessing_args.add_argument(
        "--dataset_symbol",
        type=str,
        default="batch",
        help="key to be used for batch correction",
    )
    preprocessing_args.add_argument(
        "--no_hvg",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    preprocessing_args.add_argument(
        "--n_top_genes",
        type=int,
        default=10000,
        help="number of genes if using hvg",
    )
    preprocessing_args.add_argument(
        "--split_data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    preprocessing_args.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="default test split if split_data is True",
    )

    training_args = parser.add_argument_group("training")
    training_args.add_argument(
        "--label_symbols",
        type=str,
        default="batch",
        help="labels used for plotting",
    )
    training_args.add_argument(
        "--data_symbol",
        type=str,
        default="data",
        help="data field",
    )
    training_args.add_argument(
        "--gene_symbol",
        type=str,
        default="gene_names",
        help="gene names",
    )
    training_args.add_argument(
        "--count_symbol",
        type=str,
        default=None,
        help="normalized data",
    )
    training_args.add_argument(
        "--output",
        type=str,
        default="embedding.npz",
        help="output embedding",
    )
    training_args.add_argument(
        "--model",
        type=str,
        default="model.ckpt",
        help="output model",
    )
    training_args.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs",
    )
    training_args.add_argument(
        "--latent_dimension",
        type=int,
        default=90,
        help="size of latent dim",
    )
    training_args.add_argument(
        "--hidden_dimension",
        type=int,
        default=128,
        help="size of hidden dim",
    )
    training_args.add_argument(
        "--layers",
        type=int,
        default=1,
        help="number of layers",
    )
    training_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    training_args.add_argument(
        "--batch_size",
        type=int,
        default=255,
        help="batch size",
    )
    training_args.add_argument(
        "--patience",
        type=int,
        default=15,
        help="patience",
    )
    training_args.add_argument(
        "--gene_factor",
        type=int,
        default=50,
        help="weight for marker genes",
    )
    training_args.add_argument(
        "--encode_covariates",
        type=bool,
        default=True,
        help="encoding covariates",
    )
    training_args.add_argument(
        "--ln",
        type=bool,
        default=False,
        help="layer normalization",
    )
    training_args.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="alternative validation data to plot",
    )
    training_args.add_argument(
        "--random_assignment",
        type=bool,
        default=False,
        help="random assignment of clusters",
    )
    training_args.add_argument(
        "--kl_weight",
        type=float,
        default=0.001,
        help="initial KL weight",
    )
    training_args.add_argument(
        "--mmd_weight",
        type=float,
        default=0.1,
        help="initial MMD weight",
    )
    training_args.add_argument(
        "--reconstruction_weight",
        type=float,
        default=1.0,
        help="initial reconstruction weight",
    )
    training_args.add_argument(
        "--kl_anneal_warmup_epochs",
        type=int,
        default=20,
        help="initial warmup epochs",
    )
    training_args.add_argument(
        "--kl_anneal_max",
        type=float,
        default=0.05,
        help="maximum KL annealing weight",
    )

    cleanup_args = parser.add_argument_group("cleanup")
    cleanup_args.add_argument(
        "--filtered_data_output",
        type=str,
        default="filtered_data.npz",
        help="cleaned up data",
    )
    cleanup_args.add_argument(
        "--conf_data",
        type=str,
        default="conf_data.npz",
        help="data with confidence scores",
    )
    cleanup_args.add_argument(
        "--plots",
        type=str,
        default="data",
        help="baseline plot name",
    )
    cleanup_args.add_argument(
        "--labels",
        type=str,
        default=None,
        help="label used for cleanup/leiden if None",
    )
    cleanup_args.add_argument(
        "--cluster_min",
        type=float,
        default=0.2,
        help="min confidence for per cluster cutoff",
    )
    cleanup_args.add_argument(
        "--quantil",
        type=float,
        default=0.2,
        help="confidence quantil",
    )
    cleanup_args.add_argument(
        "--quantil_max",
        type=float,
        default=0.3,
        help="max confidence for per cluster cutoff",
    )
    cleanup_args.add_argument(
        "--no_cleanup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    additional_fields_args = parser.add_argument_group("additional load fields")
    additional_fields_args.add_argument(
        "--obs_fields",
        nargs="+",
        type=str,
        default=[],
        help="list of observation fields",
    )
    additional_fields_args.add_argument(
        "--layer_fields",
        nargs="+",
        type=str,
        default=[],
        help="list of layer fields",
    )

    return parser


def run_preprocessing(args: argparse.Namespace) -> None:
    print_banner("running pre-processing")

    preprocessing(
        marker_gene_list=args.marker_genes,
        dataset=args.dataset_path,
        full_name=args.output_h5ad,
        batch_key=args.dataset_symbol,
        no_hvg=args.no_hvg,
        n_top_genes=args.n_top_genes,
        split_data=args.split_data,
        test_size=args.test_size,
    )


def run_training(args: argparse.Namespace) -> None:
    print_banner("training")

    gene_names = load_marker_genes(args.marker_genes)

    if args.no_hvg == True and args.split_data == False:
        train_data = args.dataset_path
    elif args.split_data == False and args.no_hvg == False:
        train_data = args.output_h5ad
    elif args.split_data == True:
        train_data = args.output_h5ad + "_ref.h5ad"

    train_hvae_main(
        train_data_name=train_data,
        gene_names=gene_names,
        dataset_symbol=args.dataset_symbol,
        label_symbols=args.label_symbols,
        data_symbol=args.data_symbol,
        gene_symbol=args.gene_symbol,
        val_data_path=args.val_data_path,
        num_epochs=args.epochs,
        latent_dim=args.latent_dimension,
        hidden_dim=args.hidden_dimension,
        layers=args.layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        encode_covariates=args.encode_covariates,
        ln=args.ln,
        gene_factor=args.gene_factor,
        kl_weight=args.kl_weight,
        mmd_weight=args.mmd_weight,
        reconstruction_weight=args.reconstruction_weight,
        save_embedding=args.output,
        kl_anneal_warmup_epochs=args.kl_anneal_warmup_epochs,
        kl_anneal_max=args.kl_anneal_max,
        model_checkpoint=args.model,
        count_symbol=args.count_symbol,
        random_assignment=args.random_assignment,
    )

def run_cleanup(args: argparse.Namespace) -> None:
    print_banner("data cleanup")

    if not args.no_cleanup:
        clean_up(
            args.output,
            args.filtered_data_output,
            args.conf_data,
            plots=args.plots,
            labels=args.labels,
            cluster_min=args.cluster_min,
            quantil=args.quantil,
            quantil_max=args.quantil_max,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed()

    run_preprocessing(args)
    run_training(args)
    run_cleanup(args)

if __name__ == "__main__":
    main()