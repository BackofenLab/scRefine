import numpy as np
import logging
import time
import os
import json
from sklearn.manifold import TSNE
import time
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
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import argparse
from scanpy.tl import leiden
from scanpy.pp import neighbors
import anndata as ad
import plotly.express as px

def plot_tsne(projections, batches, name):
    

    fig = px.scatter(
        projections, x=0, y=1,
        color=batches, labels={'color': 'batches'}
    )
    fig.write_html(name) 
    return

    
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
    
def clean_up(data, filtered_data_output, conf_data, plots, labels = None, cluster_min = 0.2, quantil = 0.2, quantil_max = 0.3):

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
        

    for cluster_id in np.unique(labels):
            

        selected_conf = np.array(conf_knn)[np.array(labels) == cluster_id]
        data_x_selected = np.array(embeddings)[np.array(labels) == cluster_id]
        data_x_selected_emb = np.array(data_x_tsne)[np.array(labels) == cluster_id]
        selected_conf_knn = np.array(conf_knn)[np.array(labels) == cluster_id]
        labels_selected = labels[np.array(labels) == cluster_id]
            
        q1 = max(min(np.quantile(selected_conf_knn, quantil), quantil_max), cluster_min)
            

        mask = np.array(selected_conf_knn) >= q1
        filtered_data = np.array(data_x_selected_emb)[mask]
        filtered_conf = np.array(selected_conf_knn)[mask]
        filtered_labels = np.array(labels_selected)[mask]
            
        filtered_data_added.extend(filtered_data)
        filtered_conf_added.extend(filtered_conf)
        filtered_labels_list.extend(filtered_labels)
            
    np.savez(filtered_data_output, data = filtered_data_added, labels = filtered_labels_list, confidence = filtered_conf_added)
    
    plot_tsne(filtered_data_output, filtered_conf_added, plots + "_clean.html")
    plot_tsne(filtered_data_output, filtered_labels_list, plots + "_clean_labels.html")
    
    return


if __name__ == "__main__":

    cmdline_parser = argparse.ArgumentParser('clean up data')


    cmdline_parser.add_argument('-d', '--data',
                                help='data',
                                default = 'embedding.npz',
                                type=str)
    cmdline_parser.add_argument('-f', '--filtered_data_output',
                                help='data',
                                default = 'filtered_data.npz',
                                type=str)
    cmdline_parser.add_argument('-c', '--conf_data',
                                help='data',
                                default = 'conf_data.npz',
                                type=str)
    cmdline_parser.add_argument('-p', '--plots',
                                help='plot',
                                default = 'data',
                                type=str)
    cmdline_parser.add_argument('-p', '--labels',
                                help='label used for cleanup/leiden if None',
                                default = None,
                                type=str)
    cmdline_parser.add_argument('-p', '--cluster_min',
                                help='plot',
                                default = 0.2,
                                type=float)
    cmdline_parser.add_argument('-p', '--quantil',
                                help='plot',
                                default = 0.2,
                                type=float)
    cmdline_parser.add_argument('-p', '--quantil_max',
                                help='plot',
                                default = 0.3,
                                type=float)
                                
    args, unknowns = cmdline_parser.parse_known_args()
    clean_up(args.data, args.filtered_data_output, args.conf_data, plots=args.plots, labels = args.labels, cluster_min = args.cluster_min, quantil = args.quantil, quantil_max = args.quantil_max)
    
