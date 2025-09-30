# Super exemple de réseau de neurones 1D → 1D pour le cours de Deep Learning from Scratch
# Approximateur de fonction cosinus
# Auteur : Alexandre Meyer, 2025

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Jeu de données en dur
# -----------------------------
#x_max = 5.0
#x_data = np.array([[0.0],[0.55],[0.8],[1.0],[1.4],[2.0],[2.6],[3.1],[3.7],[4.1],[4.3],[5.0]])
# Plage de données plus large pour cas plus difficile
x_max = 10.0
x_data = np.array([[0.0],[0.55],[0.8],[1.0],[1.4],[2.0],[2.6],[3.1],[3.7],[4.1],[4.3],
                    [5.0],[5.7],[6.0],[6.7],[7.0],[7.4],[8.0],[8.5],[9.0],[9.5]])
d_data = np.cos(x_data)

# -----------------------------
# Paramètres du réseau
# -----------------------------
np.random.seed(0)

# N neurones cachés
N = 10
w1 = np.random.randn(1, N)   # 1 entrée → 2 neurones cachés
b1 = np.zeros((1, N))        # un biais par neurone caché
w2 = np.random.randn(N, 1)   # 2 neurones cachés → 1 sortie
b2 = np.zeros((1, 1))        # sortie scalaire



lr = 0.01
epochs = 50000

# -----------------------------
# Activation et dérivée
# -----------------------------
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)

# -----------------------------
# Loss MSE et dérivée
# -----------------------------
def mse(y, d):
    return 0.5 * np.mean((y - d)**2)

def mse_prime(y, d):
    N = y.shape[0]
    return (y - d) / N


# -----------------------------
# Entraînement
# -----------------------------
for epoch in range(epochs):
    # --- Forward ---
    a1 = np.dot(x_data, w1) + b1
    h1 = tanh(a1)
    a2 = np.dot(h1, w2) + b2
    y = linear(a2)  # sortie linéaire

    # --- Loss ---
    loss = mse(y, d_data)

    # --- Backprop ---
    delta2 = mse_prime(y, d_data) * linear_deriv(a2)  # δ2 = dL/da2
    dw2 = np.dot(h1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.dot(delta2, w2.T) * tanh_deriv(a1)
    dw1 = np.dot(x_data.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # --- Mise à jour ---
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# -----------------------------
# Visualisation
# -----------------------------
x_plot = np.linspace(0, x_max, 100).reshape(-1,1)
h1_plot = tanh(np.dot(x_plot, w1) + b1)
y_plot = np.dot(h1_plot, w2) + b2  # sortie linéaire

plt.plot(x_plot, y_plot, label="Réseau approx")
plt.scatter(x_data, d_data, color='red', label="Données originales")
plt.legend()
plt.show()