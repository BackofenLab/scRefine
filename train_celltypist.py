import scanpy as sc
import celltypist
import celltypist.annotate
import sklearn

celltypist.models.models_path = "/home/lagandua/projects/sven/celltypist/.cache"
import numpy as np
import os
from time import time


def train():
    # Seed
    np.random.seed(42)

    # Load the data
    adata = sc.read("/home/lagandua/projects/sven/pineapple/data/global.h5ad")

    # Filter genes
    ct_model = celltypist.models.Model.load("Immune_All_Low.pkl")
    chosen_genes = np.array(
        list(set(ct_model.classifier.features).intersection(set(adata.var_names)))
    )
    keep = np.isin(adata.var_names, chosen_genes)

    # Normalize data
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Split the data
    val_size = 0.1  # 10%
    true_count = int(adata.shape[0] * val_size)
    false_count = adata.shape[0] - true_count

    val_flag = np.concatenate((np.full(true_count, True), np.full(false_count, False)))
    np.random.shuffle(val_flag)

    # Train the model
    model = celltypist.train(
        adata[~val_flag, keep],
        labels="Predicted_labels_CellTypist",
        n_jobs=30,
        feature_selection=True,
        check_expression=False,
        use_SGD=True,
        use_GPU=False,
    )
    # Save the model
    root_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = f"{root_dir}/exp/cell_typist/{str(int(time()))}"
    os.makedirs(experiment_dir, exist_ok=True)
    model.write(f"{experiment_dir}/model_global.pkl")

    ############# Training Set #############
    # Train Accuracy
    predictions = celltypist.annotate(
        adata[~val_flag, keep],
        model=model,
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
        use_GPU=True,
    )
    adata_pred = predictions.to_adata(insert_prob=True)
    # adata_pred.write(f"{experiment_dir}/train_global.h5ad")
    accuracy = np.mean(
        np.array(adata_pred.obs["majority_voting"])
        == np.array(adata[~val_flag].obs["Predicted_labels_CellTypist"])
    )
    print("Train Accuracy: ", accuracy)

    # Train F1-Score
    f1_score = sklearn.metrics.f1_score(
        np.array(adata[~val_flag].obs["Predicted_labels_CellTypist"]),
        np.array(adata_pred.obs["majority_voting"]),
        average="weighted",
    )
    print("Train Weighted F1-Score: ", f1_score)

    ########## Validation Set ##########
    # Evaluate the model on validation set
    vdata = adata[val_flag, keep]
    predictions = celltypist.annotate(
        vdata,
        model=model,
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
        use_GPU=True,
    )
    vdata_pred = predictions.to_adata(insert_prob=True)
    # vdata_pred.write(f"{experiment_dir}/val_global.h5ad")
    accuracy = np.mean(
        np.array(vdata_pred.obs["majority_voting"])
        == np.array(adata[val_flag].obs["Predicted_labels_CellTypist"])
    )
    print("Validation Accuracy: ", accuracy)

    # Validation F1 - Score
    f1_score = sklearn.metrics.f1_score(
        np.array(adata[val_flag].obs["Predicted_labels_CellTypist"]),
        np.array(vdata_pred.obs["majority_voting"]),
        average="weighted",
    )
    print("Validation Weighted F1-Score: ", f1_score)


if __name__ == "__main__":
    train()
