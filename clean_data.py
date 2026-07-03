import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import argparse
from scanpy.tl import leiden
from scanpy.pp import neighbors
import anndata as ad
from plotting import plot_tsne


def train_knn(data_x, data_y, train_fraction=0.4, confidence_threshold=0.5, random_state=42):
    """
    Fits a radius-neighbors classifier and flags low-confidence predictions.

    Args:
        data_x (array-like): Feature matrix (e.g. TSNE embeddings).
        data_y (array-like): Integer class labels.
        train_fraction (float, optional): Fraction of data used for the stratified
            train/test split (the split itself is unused downstream; kept for parity with
            the fitted model's usage). Defaults to 0.4.
        confidence_threshold (float, optional): Predictions with max class probability below
            this value are flagged as low-confidence. Defaults to 0.5.
        random_state (int, optional): Random seed for the train/test split. Defaults to 42.

    Returns:
        tuple: `(indices_of_interest, model, data_x, data_y, confidences, confidences2, probs)`
            where `indices_of_interest` are indices with confidence below the threshold,
            `confidences` is the predicted probability of each sample's true class, and
            `confidences2` is the max predicted probability per sample.
    """

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(
        data_x, data_y, train_size=train_fraction, stratify=data_y, random_state=random_state
    )

    model = RadiusNeighborsClassifier(radius = 0.1)
    model.fit(data_x, data_y)
    probs = model.predict_proba(data_x)

    preds = np.argmax(probs, axis=1)

    confidences = np.array(probs)[np.arange(len(probs)), data_y]
    confidences2 = np.max(np.array(probs), axis = 1)

    indices_of_interest = []
    for i, (true, pred, conf) in enumerate(zip(data_y, preds, confidences)):
        if conf < confidence_threshold:
            indices_of_interest.append(i)

    return indices_of_interest, model, data_x, data_y, confidences, confidences2, probs
    
def clean_up(data, filtered_data_output, conf_data, plots, labels = None, cluster_min = 0.2, quantil = 0.2, quantil_max = 0.3):
    """
    Filters low-confidence embeddings out of a saved embedding file, per cluster.

    Loads embeddings from `data`, clusters them (Leiden on a k-NN graph if `labels` is None,
    otherwise a saved label array), TSNE-projects them, fits a k-NN confidence model, and
    for each cluster keeps only samples whose confidence is above a per-cluster quantile
    threshold (clamped to `[cluster_min, quantil_max]`). Saves confidence and filtered
    results to disk, and writes TSNE scatter plots at each stage.

    Args:
        data (str): Path to an `.npz` file containing an 'embedding' array (and optionally
            the label array named by `labels`).
        filtered_data_output (str): Output `.npz` path for the filtered embeddings/labels.
        conf_data (str): Output `.npz` path for the unfiltered embeddings with confidences.
        plots (str): Prefix used for the output TSNE plot HTML files.
        labels (str, optional): Key in `data` holding labels; if None, Leiden clustering is
            used to derive labels. Defaults to None.
        cluster_min (float, optional): Minimum confidence threshold per cluster.
        quantil (float, optional): Quantile of per-cluster confidence used as the cutoff.
        quantil_max (float, optional): Maximum confidence threshold per cluster.
    """
    loaded = np.load(data)
    embeddings = loaded["embedding"]

    if labels == None:
        adata = ad.AnnData(X=embeddings)
        neighbors(adata, knn=True, n_neighbors = 50, use_rep='X')
        leiden(adata, resolution=0.5)
        labels = [int(l) for l in adata.obs["leiden"]]
        labels_unique = list(np.unique(labels))
        labels_int = [labels_unique.index(l) for l in labels]
    else:
        labels = loaded[labels]
        labels_unique = list(np.unique(labels))
        labels_int = [labels_unique.index(l) for l in labels]

    tsne = TSNE(n_components=2, random_state=0)
    train_embeddings = tsne.fit_transform(embeddings)
    scaler = StandardScaler()
    scaler.fit(train_embeddings)
    train_embeddings = scaler.transform(train_embeddings)
        
    inds_knn, model_knn, data_x_test, data_y_test, conf_knn, conf2_knn, probs = train_knn(train_embeddings, np.array(labels_int), train_fraction=0.4, confidence_threshold=0.2, random_state=42)

    np.savez(conf_data, embeddings = embeddings, labels = labels, confidence = conf_knn, tsne_emb = train_embeddings)
    
    plot_tsne(train_embeddings, labels, plots + "_labels.html")
    plot_tsne(train_embeddings, conf_knn, plots + "_conf.html")

    data_x_tsne = train_embeddings
    filtered_data_added = []
    filtered_conf_added = []
    filtered_labels_list = []
    filtered_data_tsne = []
        

    for cluster_id in np.unique(labels):
            

        selected_conf = np.array(conf_knn)[np.array(labels) == cluster_id]
        data_x_selected = np.array(embeddings)[np.array(labels) == cluster_id]
        data_x_selected_emb = np.array(data_x_tsne)[np.array(labels) == cluster_id]
        selected_conf_knn = np.array(conf_knn)[np.array(labels) == cluster_id]
        selected_tsne = np.array(train_embeddings)[np.array(labels) == cluster_id]
        labels_selected = np.array(labels)[np.array(labels) == cluster_id]

        
            
        q1 = max(min(np.quantile(selected_conf_knn, quantil), quantil_max), cluster_min)
            

        mask = np.array(selected_conf_knn) >= q1
        filtered_data = np.array(data_x_selected_emb)[mask]
        filtered_conf = np.array(selected_conf_knn)[mask]
        filtered_labels = np.array(labels_selected)[mask]
        filtered_tsne = np.array(selected_tsne)[mask]

        filtered_data_tsne.extend(filtered_tsne)
        filtered_data_added.extend(filtered_data)
        filtered_conf_added.extend(filtered_conf)
        filtered_labels_list.extend(filtered_labels)
            
    np.savez(filtered_data_output, data = filtered_data_added, labels = filtered_labels_list, confidence = filtered_conf_added, tsne = filtered_data_tsne)
    
    plot_tsne(filtered_data_tsne, filtered_conf_added, plots + "_clean.html")
    plot_tsne(filtered_data_tsne, filtered_labels_list, plots + "_clean_labels.html")
    
    return


