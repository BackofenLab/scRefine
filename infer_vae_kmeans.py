# A script to infer the VAE model and perform KMeans clustering on the latent space
# with different number of clusters and different number of labels per cluster.
# %%
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset.singlecell import SingleCellDataset
from models.vae import VAE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import time


# 256 Full
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/256/2024-07-26/19-37-24"

# 512 Full
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/512/2024-07-26/19-36-52"

# 1024 High level
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/1024/2024-07-30/14-49-42"
# 1024 min2K
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/1024/2024-07-03/13-58-28"
# 1024 Full
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/1024/2024-07-26/18-38-05"
# 1024 100% High
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/1024/2024-07-31/09-09-54"
# 1024 100% Low
exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/1024/2024-07-31/09-10-50"

# 2048 min2K
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/2048/2024-07-23/16-15-30"
# 2048 Full
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/2048/2024-07-23/16-10-06"

# 4096 min2K
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/4096/2024-07-23/17-26-46"
# 4096 Full
# exp_dir = "/home/lagandua/projects/sven/pineapple/exp/vae/4096/2024-07-23/16-12-03"

project_dir = "/home/lagandua/projects/sven/pineapple"
model_path = f"{exp_dir}/model_49.pth"
label_to_idx_path = f"{exp_dir}/label_to_index.json"
data_name = "data_low"
# data_path = f"{project_dir}/data/sven_classes/{data_name}.npz"
data_path = f"{project_dir}/data/Predicted_labels_CellTypist/{data_name}.npz"
out_dir = f"{project_dir}/exp/plots/vae-kmeans/"
os.makedirs(out_dir, exist_ok=True)
comment = "kldiv-w-25e-5"
device = "cuda:1"
latent_dim = int(exp_dir.split("/")[-3])

# %%
label_to_index = json.load(open(label_to_idx_path))

# %%
val_dataset = SingleCellDataset(data_path)
val_dataset.label_to_index = label_to_index
val_dataset.label_indices = [
    val_dataset.label_to_index[label] for label in val_dataset.labels
]
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True)

# %%
input_size = val_dataset.data.shape[-1]
num_classes = len(val_dataset.label_to_index)
latent_dim = latent_dim
model = VAE(input_size, latent_dim=latent_dim).to(device).float()
model.load_state_dict(torch.load(model_path))
model.eval()

# %%
val_reps = []
val_recons = []
val_inputs = []
val_labels = []
print("Generating representations")
with torch.no_grad():
    for val_input, val_target in val_dataloader:
        val_input = val_input.to(device).float()
        _, z, _ = model(val_input)

        val_labels.extend(val_target.cpu().numpy().tolist())
        val_reps.extend(z.cpu().numpy().tolist())
        # val_recons.extend(recon.cpu().numpy().tolist())
        # val_inputs.extend(val_input.cpu().numpy().tolist())
        _ = None

# %%
index_to_label = {
    v: k for k, v in val_dataset.label_to_index.items()
}  # Useful for plot legend

# %%
import numpy as np

val_reps = np.array(val_reps)
count = len(val_reps)

# Embeddings
embeddings = np.copy(val_reps[:count])

# Labels
# labels = np.copy(val_dataset.label_indices[:count])
labels = np.array(val_labels)

# %%
# Perform KMeans clustering with sklearn

for num_clusters in [
    num_classes,
    num_classes * 2,
    num_classes * 4,
    num_classes * 8,
    num_classes * 16,
]:
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    start_time = time.time()
    # Fit and predict the embeddings
    kmeans_results = kmeans.fit_predict(embeddings)
    sil_score_before = silhouette_score(embeddings, kmeans_results)

    print(f"KMeans took {time.time() - start_time} seconds")

    for labels_per_cluster in [50]:
        # %%
        class_to_cluster = {}
        cluster_to_class = {}
        exemplar_count = labels_per_cluster
        exemplar_count_temp = exemplar_count
        label_counter = 0
        for c_id in range(len(set(kmeans_results))):
            cluster_indices = np.where(kmeans_results == c_id)[0]
            if len(cluster_indices) < exemplar_count:
                exemplar_count_temp = len(cluster_indices)
            else:
                exemplar_count_temp = exemplar_count
            exemplar_indices = np.random.choice(
                cluster_indices, exemplar_count_temp, replace=False
            )
            exemplar_reps = embeddings[exemplar_indices]
            exemplar_labels = labels[exemplar_indices]
            exemplar_class = np.argmax(np.bincount(exemplar_labels))
            class_to_cluster[exemplar_class] = c_id
            cluster_to_class[c_id] = exemplar_class
            label_counter += exemplar_count_temp
        kmeans_results_new = np.array(
            [cluster_to_class[c_id] for c_id in kmeans_results]
        )
        print(f"Total exemplars labeled: {label_counter}")
        sil_score_after = silhouette_score(embeddings, kmeans_results_new)

        # %%
        # Perform k-NN based classification
        # Initialize KNN model
        knn = KNeighborsClassifier(n_neighbors=30)

        start_time = time.time()
        # Fit the KNN model
        knn.fit(embeddings, kmeans_results_new)
        print(f"KNN took {time.time() - start_time} seconds")

        # Predict the clusters
        knn_results = knn.predict(embeddings)

        # Calculate the accuracy
        accuracy = accuracy_score(labels, knn_results)
        # print("k-NN Classifier Accuracy:", accuracy)

        # F1 Score

        f1 = f1_score(labels, knn_results, average="weighted")
        # print("k-NN Classifier F1 Score:", f1)

        print("num_clusters:", num_clusters)
        print("labels_per_cluster:", labels_per_cluster)
        print("total labeled:", label_counter)
        print("accuracy:", accuracy)
        print("f1:", f1)
        print(
            "silhouette_score (before):",
            sil_score_before,
        )
        print(
            "silhouette_score (after):",
            sil_score_after,
        )
        print("-----------------")
