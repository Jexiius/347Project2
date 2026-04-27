# CSE 347
# Project 2: Single-sample prediction using saved CNN models
# Usage: python predict.py <dataset> <index>
#   dataset: 'mnist' or 'cho'
#   index:   sample index into the test set

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def predict_mnist(index):
    model = tf.keras.models.load_model("mnist_cnn_model.keras")

    with np.load("mnist.npz") as data:
        images = data['x_test']
        labels = data['y_test']

    if index < 0 or index >= len(images):
        print(f"Index {index} out of range (0–{len(images)-1})")
        sys.exit(1)

    img = images[index] / 255.0
    img_input = img[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)

    probs = model.predict(img_input, verbose=0)[0]
    predicted = int(np.argmax(probs))
    true_label = int(labels[index])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_title(
        f"Predicted: {predicted}   True: {true_label}\n"
        f"Confidence: {probs[predicted]:.1%}",
        fontsize=13
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def predict_cho(index):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    model = tf.keras.models.load_model("cho_cnn_model.keras")

    data = pd.read_csv("cho.txt", sep=r"\s+", header=None)
    data = data[data[1] != -1].reset_index(drop=True)

    ground_truth = data.iloc[:, 1].values
    attributes = data.iloc[:, 2:].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(ground_truth)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(attributes)

    if index < 0 or index >= len(X_scaled):
        print(f"Index {index} out of range (0–{len(X_scaled)-1})")
        sys.exit(1)

    sample = X_scaled[index][np.newaxis, ..., np.newaxis]  # (1, 16, 1)
    probs = model.predict(sample, verbose=0)[0]
    predicted = int(np.argmax(probs))
    true_label = int(y[index])

    print(f"Sample index : {index}")
    print(f"Predicted    : class {predicted}")
    print(f"True label   : class {true_label}")
    print(f"Confidence   : {probs[predicted]:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <dataset> <index>")
        print("  dataset: 'mnist' or 'cho'")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    try:
        index = int(sys.argv[2])
    except ValueError:
        print(f"Index must be an integer, got: {sys.argv[2]}")
        sys.exit(1)

    if dataset == "mnist":
        predict_mnist(index)
    elif dataset == "cho":
        predict_cho(index)
    else:
        print(f"Unknown dataset '{dataset}'. Use 'mnist' or 'cho'.")
        sys.exit(1)