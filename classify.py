import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vae import HVAE_Lightning7
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import time
from models.classifier import Classifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

        

def load_embedding_data(data_path):

    data = np.load(data_path, pickle = True)
    print(data)
    embeddings = data["embedding"]
    labels = data["labels"]
    
    return



def train_classifier(data_path):

    embeddings, labels = load_embedding_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    label_set = list(np.unique(labels))
    num_classes = len(label_set)

    classifier_model = Classifier(np.shape(embeddings)[0], 250, len(label_set)).to(device)
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
            label_data = [l for l in label_data]
            label_data = [label_set.index(l) for l in label_data]
            label_data = torch.tensor(label_data).to(device)

            classification = classifier_model(data)
            loss = criterion(classification, label_data)

        
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
            
        with torch.no_grad():
            
            for data, label_data,_, batched_data, gene in val_dataloader:

                data = data.to(device).float()
                gene = gene.to(device).float()
                batched_data = batched_data.to(device)
                label_data = [l for l in label_data]
                label_data = [label_set.index(l) for l in label_data]
                label_data = torch.tensor(label_data).to(device)

                val_outputs = classifier_model(data)
                val_loss += criterion(val_outputs, label_data)
                _, predicted = torch.max(val_outputs.data, 1)
                total += label_data.size(0)
                correct += (predicted == label_data).sum().item()
                pred_labels.extend(predicted.cpu().numpy().tolist())
                true_labels.extend(label_data.cpu().numpy().tolist())
                    
                    
                val_labels.extend(label_data.cpu().numpy().tolist())
                val_reps.extend(mu.cpu().numpy().tolist())
                _ = None


            val_loss /= len(val_dataset)
            print(
                    f"Val. Loss: {val_loss.item():.4f}, Val. Accuracy: {100*correct/total:.2f}%"
                )

        classifier_model.train()

    torch.save(classifier_model.state_dict(), f"model_classifier_300.pth")
    
    return

def predict_classifier():

    pred_labels = []
    with torch.no_grad():
            
        for data, label_data,_, batched_data, gene in pred_dataloader:

            data = data.to(device).float()
            gene = gene.to(device).float()
            batched_data = batched_data.to(device)
            output = model(data, gene, batched_data)
            mu = output["z_mu"]
                    
            val_outputs = classifier_model(mu)
            _, predicted = torch.max(val_outputs.data, 1)
            pred_labels.extend(predicted.cpu().numpy().tolist())
                    
    np.save("pred_labels_coster.npy", pred_labels)
        

    return




    
def evaluate_kmeans():

    val_reps = np.array(val_reps)
    count = len(val_reps)

    # Embeddings
    embeddings = np.copy(val_reps[:count])

    # Labels
    # labels = np.copy(val_dataset.label_indices[:count])
    val_labels = [v for v in val_labels]
    labels = np.array(val_labels)
    uniques = list(np.unique(val_labels))
    labels = np.array([uniques.index(i) for i in val_labels])

    for num_clusters in [
    num_classes,
    num_classes * 2,
    num_classes * 4,
    num_classes * 8,
    num_classes * 16,
    num_classes * 32,
    num_classes * 50,
    num_classes * 60,
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
                                default = 'classifier.pth',
                                type=str)
    cmdline_parser.add_argument('-o', '--evaluation',
                                help='output evaluation',
                                default = 'output.csv',
                                type=str)
                                
                                
    args, unknowns = cmdline_parser.parse_known_args()
    
    if args.run_type == "nn": train_classifier(data_path)
    elif args.run_type == "predict": predict(data_path)
    elif args.run_type == "kmeans": evaluate_knn(data_path)
