import scanpy as sc
import celltypist
import sklearn

celltypist.models.models_path = (
    "/home/lagandua/projects/sven/pineapple/celltypist/.cache"
)
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
    labels_name = "sven_classes"
    adata.obs[labels_name] = [
        old_to_new_class_map[c] for c in adata.obs["Predicted_labels_CellTypist"]
    ]

    # Train the model
    model = celltypist.train(
        adata[:, keep],
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

    ############ Training Set #############
    # Train Accuracy
    predictions = celltypist.annotate(
        adata[:, keep],
        model=model,
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
        use_GPU=True,
        over_cluster_resolution=len(set(adata.obs[labels_name])),
    )
    adata_pred = predictions.to_adata(insert_prob=True)
    adata_pred.write(f"{experiment_dir}/pred_global.h5ad")

    adata_pred = sc.read(f"{experiment_dir}/pred_global.h5ad")
    accuracy = np.mean(
        np.array(adata_pred.obs["majority_voting"]) == np.array(adata.obs[labels_name])
    )
    print("Train Accuracy: ", accuracy)

    # Train F1-Score
    f1_score = sklearn.metrics.f1_score(
        np.array(adata.obs[labels_name]),
        np.array(adata_pred.obs["majority_voting"]),
        average="weighted",
    )
    print("Train Weighted F1-Score: ", f1_score)

    ######## Retrain on the subset of the data ########
    pred_classes = set(np.array(adata_pred.obs["majority_voting"]))
    # Only choose rows with classes that are in the predicted classes
    subset_flag = np.array(
        [c in pred_classes for c in np.array(adata.obs[labels_name])]
    )
    model = celltypist.train(
        adata[subset_flag, keep],
        labels=labels_name,
        n_jobs=30,
        feature_selection=False,
        check_expression=False,
        use_SGD=True,
        use_GPU=False,
    )
    # Save the subset model
    model.write(f"{experiment_dir}/subset_model_global.pkl")

    ############# Subset Training Set #############
    # Train Accuracy
    predictions = celltypist.annotate(
        adata[subset_flag, keep],
        model=model,
        majority_voting=True,
        mode="prob match",
        p_thres=0.5,
        use_GPU=True,
        over_cluster_resolution=len(set(adata[subset_flag].obs[labels_name])),
    )
    adata_pred_subset = predictions.to_adata(insert_prob=True)
    adata_pred_subset.write(f"{experiment_dir}/subset_pred_global.h5ad")

    adata_pred_subset = sc.read(f"{experiment_dir}/subset_pred_global.h5ad")
    accuracy = np.mean(
        np.array(adata_pred_subset.obs["majority_voting"])
        == np.array(adata[subset_flag].obs[labels_name])
    )
    print("Subset Train Accuracy: ", accuracy)

    # Train F1-Score
    f1_score = sklearn.metrics.f1_score(
        np.array(adata[subset_flag].obs[labels_name]),
        np.array(adata_pred_subset.obs["majority_voting"]),
        average="weighted",
    )
    print("Subset Train Weighted F1-Score: ", f1_score)

    # Save classification report
    report = sklearn.metrics.classification_report(
        np.array(adata[subset_flag].obs[labels_name]),
        np.array(adata_pred_subset.obs["majority_voting"]),
        output_dict=True,
    )
    print("Subset Classification Report: ", report)

    with open(f"{experiment_dir}/subset_classification_report_global.txt", "w") as file:
        file.write(str(report))


if __name__ == "__main__":
    train()
