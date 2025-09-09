import numpy as np
import logging
import time
import os
import json
from sklearn.manifold import TSNE
import time
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import glob
import random
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import anndata as ad


def train_knn(data_x, data_y, train_fraction=0.4, confidence_threshold=0.5, random_state=42):

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
    
    

if __name__ == "__main__":


        loaded = np.load(embedding_dataset)
        filtered_data = "filtered.npz"
        embeddings = loaded["embedding"]
        labels = loaded["labels"]
        

        tsne = TSNE(n_components=2, random_state=0)
        train_embeddings = tsne.fit_transform(embeddings)
        scaler = StandardScaler()
        scaler.fit(train_embeddings)
        train_embeddings = scaler.transform(train_embeddings)
        
        inds_knn, model_knn, data_x_test, data_y_test, conf_knn, conf2_knn, probs = train_knn(train_embeddings, np.array(labels_int), train_fraction=0.4, confidence_threshold=0.2, random_state=42)
        
        data_x_tsne = train_embeddings
        filtered_data_added = []
        filtered_conf_added = []
        filtered_labels_list = []
        

        for cluster_id in np.unique(labels):
            

            selected_conf = np.array(conf_knn)[np.array(labels) == cluster_id]
            data_x_selected = np.array(embeddings)[np.array(labels) == cluster_id]
            data_x_selected_emb = np.array(data_x_tsne)[np.array(labels) == cluster_id]
            selected_conf_knn = np.array(conf_knn)[np.array(labels) == cluster_id]
            labels_selected = labels[np.array(labels) == cluster_id]
            

            q1 = max(min(np.quantile(selected_conf_knn, 0.2), 0.3), 0.2)
            
            mask = np.array(selected_conf_knn) >= q1
            filtered_data = np.array(data_x_selected_emb)[mask]
            filtered_conf = np.array(selected_conf_knn)[mask]
            filtered_labels = np.array(labels_selected)[mask]
            
            filtered_data_added.extend(filtered_data)
            filtered_conf_added.extend(filtered_conf)
            filtered_labels_list.extend(filtered_labels)
            
        np.savez(filtered_data, data = filtered_data_added, labels = filtered_labels_list, confidence = filtered_conf_added)
