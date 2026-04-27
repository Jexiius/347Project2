# CSE 347 - Project 2
# Hyperparameter sweep + heatmap visualization for CNN on Cho and MNIST
# Run AFTER the main CNN.py so you know your data paths work.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models

# ---------- Reuse your loaders (lightweight copies) ----------
def load_cho(filename):
    data = pd.read_csv(filename, sep=r"\s+", header=None)
    data = data[data[1] != -1].reset_index(drop=True)
    y = LabelEncoder().fit_transform(data.iloc[:, 1].values)
    X = data.iloc[:, 2:].values.astype(float)
    return X, y

def load_mnist():
    with np.load("mnist.npz") as data:
        X_train = data['x_train'] / 255.0
        X_test = data['x_test'] / 255.0
        y_train, y_test = data['y_train'], data['y_test']
    return X_train[..., np.newaxis], X_test[..., np.newaxis], y_train, y_test

# ---------- Model builders (mirror your CNN.py) ----------
def build_cho_model(n_classes, filters, dense_units):
    m = models.Sequential([
        layers.Input(shape=(16, 1)),
        layers.Conv1D(filters, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(max(filters // 2, 4), 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def build_mnist_model(filters, dense_units):
    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(max(filters // 2, 4), (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# ---------- Sweep functions ----------
FILTERS_GRID = [16, 32, 64, 128]
DENSE_GRID   = [16, 32, 64, 128]

def sweep_cho(n_seeds=3):
    X, y = load_cho("cho.txt")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)[..., np.newaxis]
    X_te = sc.transform(X_te)[..., np.newaxis]
    n_classes = len(np.unique(y))

    results = np.zeros((len(FILTERS_GRID), len(DENSE_GRID)))
    for i, f in enumerate(FILTERS_GRID):
        for j, d in enumerate(DENSE_GRID):
            accs = []
            for seed in range(n_seeds):
                tf.random.set_seed(seed)
                np.random.seed(seed)
                m = build_cho_model(n_classes, f, d)
                m.fit(X_tr, y_tr, epochs=50, batch_size=16, verbose=0)
                pred = np.argmax(m.predict(X_te, verbose=0), axis=1)
                accs.append(accuracy_score(y_te, pred))
            results[i, j] = np.mean(accs)
            print(f"Cho  filters={f:3d}  dense={d:3d}  acc={results[i,j]:.4f}")
    return results

def sweep_mnist():
    X_tr, X_te, y_tr, y_te = load_mnist()
    # Use a 10k subset for sweep speed; the chart is about RELATIVE comparison.
    X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=10000, stratify=y_tr, random_state=42)
    X_val, X_train_final = X_sub[:1000], X_sub[1000:]
    y_val, y_train_final = y_sub[:1000], y_sub[1000:]

    results = np.zeros((len(FILTERS_GRID), len(DENSE_GRID)))
    for i, f in enumerate(FILTERS_GRID):
        for j, d in enumerate(DENSE_GRID):
            tf.random.set_seed(42)
            m = build_mnist_model(f, d)
            m.fit(X_train_final, y_train_final, epochs=3, batch_size=64, verbose=0)
            # Evaluate on the held-out test set for a real chart-worthy number
            pred = np.argmax(m.predict(X_te, verbose=0), axis=1)
            results[i, j] = accuracy_score(y_te, pred)
            print(f"MNIST filters={f:3d}  dense={d:3d}  acc={results[i,j]:.4f}")
    return results

# ---------- Plotting ----------
def plot_heatmap(results, title, outpath, cmap_colors):
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    cmap = LinearSegmentedColormap.from_list("custom", cmap_colors)
    im = ax.imshow(results, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(DENSE_GRID)))
    ax.set_yticks(range(len(FILTERS_GRID)))
    ax.set_xticklabels(DENSE_GRID, fontsize=11)
    ax.set_yticklabels(FILTERS_GRID, fontsize=11)
    ax.set_xlabel("Dense Units", fontsize=12, fontweight='bold')
    ax.set_ylabel("Conv Filters", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Annotate each cell with the accuracy value
    best_i, best_j = np.unravel_index(results.argmax(), results.shape)
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            val = results[i, j]
            color = "white" if val < (results.min() + results.max()) / 2 else "black"
            weight = 'bold' if (i, j) == (best_i, best_j) else 'normal'
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    color=color, fontsize=11, fontweight=weight)

    # Highlight best cell with a box
    rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                         fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Test Accuracy", fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved {outpath}  (best: filters={FILTERS_GRID[best_i]}, "
          f"dense={DENSE_GRID[best_j]}, acc={results[best_i, best_j]:.4f})")
    plt.close()

if __name__ == "__main__":
    print("=== Cho sweep ===")
    cho_results = sweep_cho(n_seeds=3)
    plot_heatmap(cho_results,
                 "CNN Hyperparameter Sweep — Cho Dataset\n(mean test accuracy over 3 seeds)",
                 "cho_sweep.png",
                 ["#0a1929", "#1976d2", "#42a5f5", "#fff59d"])

    print("\n=== MNIST sweep ===")
    mnist_results = sweep_mnist()
    plot_heatmap(mnist_results,
                 "CNN Hyperparameter Sweep — MNIST Dataset\n(test accuracy, 3 epochs on 9k subset)",
                 "mnist_sweep.png",
                 ["#1a0a29", "#7b1fa2", "#ce93d8", "#fff59d"])

    # Save raw numbers too in case you want a backup table for the report
    pd.DataFrame(cho_results, index=[f"f={f}" for f in FILTERS_GRID],
                 columns=[f"d={d}" for d in DENSE_GRID]).to_csv("cho_sweep.csv")
    pd.DataFrame(mnist_results, index=[f"f={f}" for f in FILTERS_GRID],
                 columns=[f"d={d}" for d in DENSE_GRID]).to_csv("mnist_sweep.csv")
    print("\nCSVs saved for the appendix/report.")
