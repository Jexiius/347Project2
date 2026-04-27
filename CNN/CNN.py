# CSE 347
# Project 2: CNN Implementation on Cho and MNIST Datasets
# Connor McDowell, Jake Carlin, Will Hoog

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

# Function to load the Cho dataset
# PARAM filename: The path to the Cho dataset file
def load_cho(filename):
    data = pd.read_csv(filename, sep=r"\s+", header=None)
    data = data[data[1] != -1].reset_index(drop=True)  # Remove rows with outlier ground truth value (-1)
    print(data.head())

    gene_ids = data.iloc[:, 0].values
    ground_truth = data.iloc[:, 1].values
    attributes = data.iloc[:, 2:].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(ground_truth)

    print(f"Samples: {attributes.shape[0]}, Features: {attributes.shape[1]}, Classes: {len(np.unique(y))}")

    return gene_ids, y, attributes

# Function to preprocess the Cho dataset (scaling and train-test split)
# PARAM X: The feature matrix of the Cho dataset
# PARAM y: The label vector of the Cho dataset
# PARAM test_size: The proportion of the dataset to include in the test split (default is 0.2)
# PARAM random_state: The seed used by the random number generator for reproducibility (default is 42)
def preprocess_cho(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

# Function to load the MNIST dataset
def load_mnist():
    with np.load("../mnist.npz") as data:
        train_images = data['x_train']
        train_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']

    # Normalize to [0, 1] and add channel dimension
    X_train = train_images / 255.0
    X_test = test_images / 255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, train_labels, test_labels

# Function to preprocess the MNIST dataset (train-val split)
# PARAM X_train: The training feature matrix of the MNIST dataset
# PARAM X_test: The testing feature matrix of the MNIST dataset
# PARAM y_train: The training label vector of the MNIST dataset
# PARAM random_state: The seed used by the random number generator for reproducibility (default is 42)
def preprocess_mnist(X_train, X_test, y_train, random_state=42):
    # Carve out 10% validation set from training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train)

    print(f"Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    return X_tr, X_val, X_test, y_tr, y_val

# Build a 1D CNN model for the Cho time-series dataset
def build_cho_model(n_classes, filters=32, dense_units=64):
    m = models.Sequential([
        layers.Input(shape=(16, 1)),
        layers.Conv1D(filters, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters // 2, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# Build a 2D CNN model for the MNIST image dataset
def build_mnist_model(filters=32, dense_units=64):
    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters // 2, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# Function to train a CNN classifier on the Cho dataset with hyperparameter tuning using K-Fold Cross Validation
# PARAM X_train: The training feature matrix of the Cho dataset
# PARAM X_test: The testing feature matrix of the Cho dataset
# PARAM y_train: The training label vector of the Cho dataset
# PARAM y_test: The testing label vector of the Cho dataset
# PARAM t: The number of trials to perform with different random seeds for reproducibility (default is 3)
def train_cho(X_train, X_test, y_train, y_test, t=3):
    n_classes = len(np.unique(y_train))

    # Params for hyperparameter tuning
    param_grid = [
        {'filters': 32, 'dense_units': 32},
        {'filters': 64, 'dense_units': 64},
        {'filters': 32, 'dense_units': 64},
        {'filters': 64, 'dense_units': 32},
    ]

    accuracies = []
    best_model_overall = None

    # Add channel dimension for 1D CNN
    X_train_3d = X_train[..., np.newaxis]
    X_test_3d = X_test[..., np.newaxis]

    for i in range(t):
        print(f"Trial {i+1}/{t}")
        tf.random.set_seed(i)

        # K-Fold cross validation on training data for hyperparameter tuning
        kf = KFold(n_splits=3, shuffle=True, random_state=i)
        best_params = None
        best_val_acc = -1

        for params in param_grid:
            fold_accs = []
            for train_idx, val_idx in kf.split(X_train_3d):
                Xf_tr, Xf_val = X_train_3d[train_idx], X_train_3d[val_idx]
                yf_tr, yf_val = y_train[train_idx], y_train[val_idx]
                m = build_cho_model(n_classes, **params)
                m.fit(Xf_tr, yf_tr, epochs=50, batch_size=16, verbose=0)
                _, acc = m.evaluate(Xf_val, yf_val, verbose=0)
                fold_accs.append(acc)
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_val_acc:
                best_val_acc = mean_acc
                best_params = params

        print(f"Best params: {best_params}")
        print(f"Best CV accuracy: {best_val_acc:.4f}")

        # Train final model on all training data with best params
        best_model = build_cho_model(n_classes, **best_params)
        best_model.fit(X_train_3d, y_train, epochs=50, batch_size=16, verbose=0)

        y_pred = np.argmax(best_model.predict(X_test_3d), axis=1)
        y_prob = best_model.predict(X_test_3d)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        best_model_overall = best_model
        print(f"Test accuracy: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Average Test Accuracy over {t} trials: {mean_acc:.4f} ± {std_acc:.4f}")
    print(classification_report(y_test, y_pred))

    return mean_acc, std_acc, best_model_overall, y_pred, y_prob

# Function to train a CNN on the MNIST dataset with hyperparameter tuning using a validation set
# PARAM X_tr: The training feature matrix of the MNIST dataset
# PARAM X_val: The validation feature matrix of the MNIST dataset
# PARAM X_test: The testing feature matrix of the MNIST dataset
# PARAM y_tr: The training label vector of the MNIST dataset
# PARAM y_val: The validation label vector of the MNIST dataset
# PARAM y_test: The testing label vector of the MNIST dataset
def train_mnist(X_tr, X_val, X_te, y_tr, y_val, y_test):
    # Params for hyperparameter tuning
    param_grid = [
        {'filters': 32, 'dense_units': 64},
        {'filters': 64, 'dense_units': 64},
        {'filters': 32, 'dense_units': 128},
        {'filters': 64, 'dense_units': 128},
        {'filters': 128, 'dense_units': 128}
    ]

    best_params = None
    best_val_acc = -1

    # Use a subset of training data for faster hyperparameter search
    X_tune, _, y_tune, _ = train_test_split(X_tr, y_tr, train_size=10000, random_state=42, stratify=y_tr)

    for params in param_grid:
        m = build_mnist_model(**params)
        m.fit(X_tune, y_tune, epochs=3, batch_size=64, verbose=0)
        _, acc = m.evaluate(X_val, y_val, verbose=0)
        print(f"Params {params} -> val accuracy: {acc:.4f}")
        if acc > best_val_acc:
            best_val_acc = acc
            best_params = params

    print(f"Best params: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Train final model on full training data with best params
    best_model = build_mnist_model(**best_params)
    best_model.fit(X_tr, y_tr, epochs=5, batch_size=64,
                   validation_data=(X_val, y_val), verbose=1)

    y_pred = np.argmax(best_model.predict(X_te), axis=1)
    y_prob = best_model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return best_model, y_pred, y_prob

# Function to evaluate the model's performance on the test set using accuracy, F1 score, and AUC
# PARAM y_test: The true labels of the test set
# PARAM y_pred: The predicted labels of the test set
# PARAM y_prob: The predicted probabilities of the test set
# PARAM dataset_name: The name of the dataset being evaluated
def evaluate_model(y_test, y_pred, y_prob, dataset_name):
    classes = np.unique(y_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Weighted F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # AUC OVR using binarized labels
    y_test_bin = label_binarize(y_test, classes=classes)
    auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')

    print(f"\n--- {dataset_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return acc, f1, auc


if __name__ == "__main__":
    CHO_MODEL_PATH = "cho_cnn_model.keras"
    MNIST_MODEL_PATH = "mnist_cnn_model.keras"

    # Load Cho dataset and preprocess it (scaling and train-test split)
    gene_ids, y, X = load_cho("../cho.txt")
    X_train_cho, X_test_cho, y_train_cho, y_test_cho = preprocess_cho(X, y)

    # Load MNIST dataset and preprocess it (train-val split)
    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = load_mnist()
    X_tr_mnist, X_val_mnist, X_te_mnist, y_tr_mnist, y_val_mnist = preprocess_mnist(X_train_mnist, X_test_mnist, y_train_mnist)

    # Train or load CNN on Cho dataset
    if os.path.exists(CHO_MODEL_PATH):
        print(f"Loading saved Cho model from {CHO_MODEL_PATH}...")
        cho_best_model = tf.keras.models.load_model(CHO_MODEL_PATH)
        X_test_cho_3d = X_test_cho[..., np.newaxis]
        cho_y_prob = cho_best_model.predict(X_test_cho_3d)
        cho_y_pred = np.argmax(cho_y_prob, axis=1)
        cho_mean_acc = accuracy_score(y_test_cho, cho_y_pred)
        cho_std_acc = 0.0
    else:
        cho_mean_acc, cho_std_acc, cho_best_model, cho_y_pred, cho_y_prob = train_cho(X_train_cho, X_test_cho, y_train_cho, y_test_cho)
        cho_best_model.save(CHO_MODEL_PATH)
        print(f"Cho model saved to {CHO_MODEL_PATH}")

    # Train or load CNN on MNIST dataset
    if os.path.exists(MNIST_MODEL_PATH):
        print(f"Loading saved MNIST model from {MNIST_MODEL_PATH}...")
        mnist_best_model = tf.keras.models.load_model(MNIST_MODEL_PATH)
        mnist_y_prob = mnist_best_model.predict(X_te_mnist)
        mnist_y_pred = np.argmax(mnist_y_prob, axis=1)
    else:
        mnist_best_model, mnist_y_pred, mnist_y_prob = train_mnist(X_tr_mnist, X_val_mnist, X_te_mnist, y_tr_mnist, y_val_mnist, y_test_mnist)
        mnist_best_model.save(MNIST_MODEL_PATH)
        print(f"MNIST model saved to {MNIST_MODEL_PATH}")

    # Evaluate Cho model
    cho_acc, cho_f1, cho_auc = evaluate_model(y_test_cho, cho_y_pred, cho_y_prob, "Cho Dataset")

    # Evaluate MNIST model
    mnist_acc, mnist_f1, mnist_auc = evaluate_model(y_test_mnist, mnist_y_pred, mnist_y_prob, "MNIST Dataset")