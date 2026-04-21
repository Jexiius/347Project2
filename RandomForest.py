# CSE 347
# Project 2: Random Forest Implementation on Cho and MNIST Datasets
# Connor McDowell, Jake Carlin, Will Hoog

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, PredefinedSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

# Function to load the Cho dataset
# PARAM filename: The path to the Cho dataset file
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

# Function to preprocess the Cho dataset (scaling, train-test split)
# PARAM X: The feature matrix of the Cho dataset
# PARAM y: The label vector of the Cho dataset
# PARAM pca_variance: The percentage of variance to retain when performing PCA (default is 0.95)
# PARAM test_size: The proportion of the dataset to include in the test split (default is 0.2)
# PARAM random_state: The seed used by the random number generator for reproducibility (default is 42)
def preprocess_cho(X, y, pca_variance = 0.95, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f" Cho Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

# Function to load the MNIST dataset
def load_mnist():
    with np.load("mnist.npz") as data:
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

# Function to preprocess the MNIST dataset (PCA and train-val split)
# PARAM X_train: The training feature matrix of the MNIST dataset
# PARAM X_test: The testing feature matrix of the MNIST dataset
# PARAM y_train: The training label vector of the MNIST dataset
# PARAM n_components: The number of principal components to keep when performing PCA (default is 100)
# PARAM random_state: The seed used by the random number generator for reproducibility (default is 42)
def preprocess_mnist(X_train, X_test, y_train, n_components=100, random_state=42):
    
    #PCA for dimensionality reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Carve out 10% Validation set from training data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state, stratify=y_train)

    print(f" MNIST Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    return X_tr, X_val, X_test, y_tr, y_val

# Function to train a Random Forest classifier on the Cho dataset with hyperparameter tuning using K-Fold Cross Validation
# PARAM X_train: The training feature matrix of the Cho dataset
# PARAM X_test: The testing feature matrix of the Cho dataset
# PARAM y_train: The training label vector of the Cho dataset
# PARAM y_test: The testing label vector of the Cho dataset
# PARAM t: The number of trials to perform with different random seeds for reproducibility (default is 3)
def train_cho(X_train, X_test, y_train, y_test, t=3):

    # Params for Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    accuracies = []

    # Repeating the training 3 times
    for i in range(t):
        print(f"Trial {i+1}/{t}")
        
        # K fold cross validation for hyperparameter tuning
        kf = KFold(n_splits = 3, shuffle=True, random_state = i)
        rf = RandomForestClassifier(random_state=i)
        grid_search = GridSearchCV(rf, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        # Train final model with best params and evaluate on the test set
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        y_prob = best_rf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Test accuracy: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Average Test Accuracy over {t} trials: {mean_acc:.4f} ± {std_acc:.4f}")
    print(classification_report(y_test, y_pred))

    return mean_acc, std_acc, best_rf, y_pred, y_prob

# Function to train the MNIST dataset using a Random Forest classifier with hyperparameter tuning
# PARAM X_tr: The training feature matrix of the MNIST dataset
# PARAM X_val: The validation feature matrix of the MNIST dataset
# PARAM X_test: The testing feature matrix of the MNIST dataset
# PARAM y_tr: The training label vector of the MNIST dataset
# PARAM y_val: The validation label vector of the MNIST dataset
# PARAM y_test: The testing label vector of the MNIST dataset
def train_mnist(X_tr, X_val, X_test, y_tr, y_val, y_test):

    # Params for Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'max_features': ['sqrt'],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }


    # Splitting training data into train and validation sets for hyperparameter tuning using PredefinedSplit
    split_index = np.concatenate([
        np.full(X_tr.shape[0], -1),
        np.zeros(X_val.shape[0])
    ])
    X_trainval = np.concatenate([X_tr, X_val], axis=0)
    y_trainval = np.concatenate([y_tr, y_val], axis=0)

    ps = PredefinedSplit(test_fold=split_index)

    # Building the Random Forest model
    rf = RandomForestClassifier(random_state = 42, n_jobs=1)
    grid_search = GridSearchCV(rf, param_grid, cv=ps, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_trainval, y_trainval)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best validation accuracy: {grid_search.best_score_:.4f}")

    # Train final model with best params and evaluate on the test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return best_rf, y_pred, y_prob

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

    # AUC OVR using binarize labels
    y_test_bin = label_binarize(y_test, classes=classes)
    auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class = 'ovr')

    print(f"\n--- {dataset_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return acc, f1, auc



if __name__ == "__main__":
    # Load Cho dataset and preprocess it (scaling, PCA, train-test split)
    gene_ids, y, X = load_cho("cho.txt")
    X_train_cho, X_test_cho, y_train_cho, y_test_cho = preprocess_cho(X, y)

    #Load MNIST dataset and preprocess it (PCA and train-val split)
    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = load_mnist()
    X_tr_mnist, X_val_mnist, X_te_mnist, y_tr_mnist, y_val_mnist = preprocess_mnist(X_train_mnist, X_test_mnist, y_train_mnist)

    # Train Random Forest on Cho dataset
    cho_mean_acc, cho_std_acc, cho_best_rf, cho_y_pred, cho_y_prob = train_cho(X_train_cho, X_test_cho, y_train_cho, y_test_cho)

    # Train Random Forest on MNIST dataset
    mnist_best_rf, mnist_y_pred, mnist_y_prob = train_mnist(X_tr_mnist, X_val_mnist, X_te_mnist, y_tr_mnist, y_val_mnist, y_test_mnist)

    # Evaluate Cho model
    cho_acc, cho_f1, cho_auc = evaluate_model(y_test_cho, cho_y_pred, cho_y_prob, "Cho Dataset")

    # Evaluate MNIST model
    mnist_acc, mnist_f1, mnist_auc = evaluate_model(y_test_mnist, mnist_y_pred, mnist_y_prob, "MNIST Dataset")

