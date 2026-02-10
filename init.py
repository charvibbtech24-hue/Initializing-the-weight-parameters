import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets

# -------------------------
# Activation functions
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# -------------------------
# Forward propagation
# -------------------------
def forward_propagation(X, parameters):
    """
    Implements forward propagation:
    LINEAR → RELU → LINEAR → RELU → LINEAR → SIGMOID
    """
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)

    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1,
             z2, a2, W2, b2,
             z3, a3, W3, b3)

    return a3, cache

# -------------------------
# Backward propagation
# -------------------------
def backward_propagation(X, Y, cache):
    """
    Implements backward propagation
    """
    m = X.shape[1]
    (z1, a1, W1, b1,
     z2, a2, W2, b2,
     z3, a3, W3, b3) = cache

    dz3 = (a3 - Y) / m
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = da2 * (a2 > 0).astype(int)   # ✅ FIXED
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = da1 * (a1 > 0).astype(int)   # ✅ FIXED
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dz3": dz3, "dW3": dW3, "db3": db3,
        "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
        "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1
    }

    return gradients

# -------------------------
# Parameter update
# -------------------------
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for k in range(1, L + 1):
        parameters[f"W{k}"] -= learning_rate * grads[f"dW{k}"]
        parameters[f"b{k}"] -= learning_rate * grads[f"db{k}"]
    return parameters

# -------------------------
# Loss
# -------------------------
def compute_loss(a3, Y):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3)) / m
    return loss

# -------------------------
# Dataset loaders
# -------------------------
def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")

    train_x = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:]).reshape(1, -1)

    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:]).reshape(1, -1)

    classes = np.array(test_dataset["list_classes"][:])

    train_x = train_x.reshape(train_x.shape[0], -1).T / 255
    test_x = test_x.reshape(test_x.shape[0], -1).T / 255

    return train_x, train_y, test_x, test_y, classes

# -------------------------
# Prediction
# -------------------------
def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, _ = forward_propagation(X, parameters)
    p[0, :] = (a3[0, :] > 0.5).astype(int)

    print("Accuracy:", np.mean(p == y))
    return p

# -------------------------
# Decision boundary utilities
# -------------------------
def predict_dec(parameters, X):
    a3, _ = forward_propagation(X, parameters)
    return (a3 > 0.5)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)

    train_X, test_X = train_X.T, test_X.T
    train_Y, test_Y = train_Y.reshape(1, -1), test_Y.reshape(1, -1)

    return train_X, train_Y, test_X, test_Y
