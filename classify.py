import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import time
from models.classifier import Classifier
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import argparse
        

def load_embedding_data(data_path):

    data = np.load(data_path, allow_pickle = True)
    print(data)
    embeddings = data["embedding"]
    labels = data["labels"]
    
    return embeddings, labels



def train_classifier(data_path, classification_output, model_path):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    embeddings, labels = load_embedding_data(data_path)
    
    label_set =  list(np.unique(labels))
    labels = np.array([label_set.index(l) for l in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    

    num_classes = len(label_set)
    classifier_model = Classifier(np.shape(embeddings)[1], 250, len(label_set)).to(device)
    classifier_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)

    val_reps = []
    val_recons = []
    val_inputs = []
    val_labels = []
    epochs = 100
    eval_every = 1

    for epoch in range(epochs):
        loss_added = 0
        for data, label in train_dataloader:
            data = data.to(device).float()

            classification = classifier_model(data)
            loss = criterion(classification, label)

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_added = loss_added + loss.item()
        
        print(f"train loss: {loss_added/len(train_dataset)}")

        val_loss = 0
        correct = 0
        total = 0
        classifier_model.eval()
        pred_labels = []
        true_labels = []
        val_labels = []
        val_reps = []
        label_prediction = []
        classifier_out = []
            
        with torch.no_grad():
            
            for data, label in val_dataloader:

                data = data.to(device).float()



                val_outputs = classifier_model(data)
                val_loss += criterion(val_outputs, label)
                _, predicted = torch.max(val_outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pred_labels.extend(predicted.cpu().numpy().tolist())
                true_labels.extend(label.cpu().numpy().tolist())
                    
                    
                val_labels.extend(label.cpu().numpy().tolist())
                val_reps.extend(data.cpu().numpy().tolist())
                label_prediction.extend(predicted.cpu().numpy().tolist())
                classifier_out.extend(val_outputs.cpu().numpy().tolist())
                _ = None


            val_loss /= len(val_dataset)
            print(
                    f"Val. Loss: {val_loss.item():.4f}, Val. Accuracy: {100*correct/total:.2f}%"
                )

        classifier_model.train()

    torch.save(classifier_model.state_dict(), model_path)
    np.savez(classification_output, labels = val_labels, predictions = label_prediction, output = classifier_out)
    
    return

def predict_classifier(data, model_path, output_path):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    pred_labels = []
    embeddings, labels = load_embedding_data(data)
    label_set =  list(np.unique(labels))
    labels = np.array([label_set.index(l) for l in labels])
    
    classifier_model = Classifier(np.shape(embeddings)[1], 250, len(label_set)).to(device)
    classifier_model.load_state_dict(torch.load(model_path))

    X_test = torch.tensor(embeddings)
    y_test = torch.tensor(labels)
    pred_dataset = TensorDataset(X_test, y_test)
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=32, shuffle=False)
    
    
    with torch.no_grad():
            
        for data, _ in pred_dataloader:

            data = data.to(device).float()
            val_outputs = classifier_model(data)
            _, predicted = torch.max(val_outputs.data, 1)
            pred_labels.extend(predicted.cpu().numpy().tolist())
                    
    np.save(output_path, pred_labels)
        
    return




    
def evaluate_kmeans(data):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    embeddings, labels = load_embedding_data(data)
    label_set =  list(np.unique(labels))
    labels = np.array([label_set.index(l) for l in labels])
    num_classes = len(label_set)


    for num_clusters in [
    num_classes,
    num_classes * 2,
    num_classes * 50,
    num_classes * 60,
    ]:

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        start_time = time.time()

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


            knn = KNeighborsClassifier(n_neighbors=30)

            start_time = time.time()

            knn.fit(embeddings, kmeans_results_new)
            print(f"KNN took {time.time() - start_time} seconds")

            knn_results = knn.predict(embeddings)

            accuracy = accuracy_score(labels, knn_results)


            f1 = f1_score(labels, knn_results, average="weighted")


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

    return




if __name__ == "__main__":

    cmdline_parser = argparse.ArgumentParser('clean up data')


    cmdline_parser.add_argument('-d', '--data',
                                help='data',
                                default = 'embedding.npz',
                                type=str)
    cmdline_parser.add_argument('-r', '--run_type',
                                help='nn/kmeans/predict',
                                default = 'nn',
                                type=str)
    cmdline_parser.add_argument('-m', '--model_path',
                                help='model path',
                                default = 'model_classifier_300.pth',
                                type=str)
    cmdline_parser.add_argument('-o', '--evaluation',
                                help='output evaluation',
                                default = 'output.csv',
                                type=str)

                                
    args, unknowns = cmdline_parser.parse_known_args()
    
    if args.run_type == "nn": train_classifier(args.data, args.evaluation, args.model_path)
    elif args.run_type == "predict": predict_classifier(args.data, args.model_path, args.evaluation)
    elif args.run_type == "kmeans": evaluate_kmeans(args.data)
