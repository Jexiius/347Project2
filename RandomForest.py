# CSE 347
# Project 2: Random Forest Implementation on Cho and MNIST Datasets
# Connor McDowell, Jake Carlin, Will Hoog

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Function to load the Cho dataset
def load_cho(filename):
    data = pd.read_csv(filename, sep=r"\s+", header=None)
    data = data[data[1] != -1].reset_index(drop=True)  # Remove rows with outlier ground truth value (-1)
    print(data.head())
    
    # Extracting metadata and attributes
    gene_ids = data.iloc[:, 0].values
    ground_truth = data.iloc[:, 1].values
    attributes = data.iloc[:, 2:].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(ground_truth)

    print(f"Samples: {attributes.shape[0]}, Features: {attributes.shape[1]}, Classes: {len(np.unique(y))}")

    return gene_ids, y, attributes

# Function to preprocess the Cho dataset (scaling, PCA, train-test split)
def preprocess_cho(X, y, pca_variance = 0.95, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_mnist():
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    # Flatten 28x28 images to 784-dimensional vectors
    X_train = train_examples.reshape(train_examples.shape[0], -1)
    X_test = test_examples.reshape(test_examples.shape[0], -1)

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, train_labels, test_labels

if __name__ == "__main__":
    # Load Cho dataset
    gene_ids, y, X = load_cho("cho.txt")
    X_train, X_test, y_train, y_test = preprocess_cho(X, y)

    #Load MNIST dataset
    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = load_mnist()