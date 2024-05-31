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

    # Load class mappings
    old_to_new_class_map = {}
    with open("/home/lagandua/projects/sven/pineapple/cell_mapping2.txt", "r") as file:
        for line in file:
            line = line.strip()
            split_line = line.split(":")
            old_classes = split_line[1].split(",")
            # print(old_classes)
            new_class = split_line[0]
            # print(new_class)
            for c in old_classes:
                old_to_new_class_map[c] = new_class
    # Create new class labels
    adata.obs["sven_classes"] = [
        old_to_new_class_map[c] for c in adata.obs["Predicted_labels_CellTypist"]
    ]

    # Normalize data
    adata.var_names_make_unique()
    # sc.pp.highly_variable_genes(adata)
    # keep = adata.var.highly_variable
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # Split the data
    val_size = 0.00
    true_count = int(adata.shape[0] * val_size)
    false_count = adata.shape[0] - true_count

    val_flag = np.concatenate((np.full(true_count, True), np.full(false_count, False)))
    np.random.shuffle(val_flag)

    # Train the model
    labels_name = "sven_classes"
    model = celltypist.train(
        adata[~val_flag, keep],
        labels=labels_name,
        n_jobs=30,
        feature_selection=False,
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
    adata_pred.write(f"{experiment_dir}/pred_global.h5ad")

    accuracy = np.mean(
        np.array(adata_pred.obs["majority_voting"])
        == np.array(adata[~val_flag].obs[labels_name])
    )
    print("Train Accuracy: ", accuracy)

    # Train F1-Score
    f1_score = sklearn.metrics.f1_score(
        np.array(adata[~val_flag].obs[labels_name]),
        np.array(adata_pred.obs["majority_voting"]),
        average="weighted",
    )
    print("Train Weighted F1-Score: ", f1_score)

    ########## Validation Set ##########
    # Evaluate the model on validation set
    # vdata = adata[val_flag, keep]
    # predictions = celltypist.annotate(
    #     vdata,
    #     model=model,
    #     majority_voting=True,
    #     mode="prob match",
    #     p_thres=0.5,
    #     use_GPU=True,
    # )
    # vdata_pred = predictions.to_adata(insert_prob=True)
    # # vdata_pred.write(f"{experiment_dir}/val_global.h5ad")
    # accuracy = np.mean(
    #     np.array(vdata_pred.obs["majority_voting"])
    #     == np.array(adata[val_flag].obs["Predicted_labels_CellTypist"])
    # )
    # print("Validation Accuracy: ", accuracy)

    # # Validation F1 - Score
    # f1_score = sklearn.metrics.f1_score(
    #     np.array(adata[val_flag].obs["Predicted_labels_CellTypist"]),
    #     np.array(vdata_pred.obs["majority_voting"]),
    #     average="weighted",
    # )
    # print("Validation Weighted F1-Score: ", f1_score)


if __name__ == "__main__":
    train()
