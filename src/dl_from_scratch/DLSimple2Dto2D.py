import numpy as np
import math
import matplotlib.pyplot as plt
import random


# Fonction d'activation (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y, d):
    return 0.5 * np.mean((y - d)**2)

def mse_prime(y, d):
    N = y.shape[0]
    return (y - d) / N  




# Generation de un point (un échantillon)
def one_sample():
    x = np.array( [ 2.0*3.141592*np.random.ranf(), 2.0*np.random.ranf()-1 ])
    if (math.cos(x[0]) < x[1]):
        y = np.array([ 0, 1])
    else:
        y = np.array([1, 0])
    return x,y


def next_batch(n):
    x = np.zeros( shape=(n,2), dtype=np.float32)
    y = np.zeros( shape=(n,2), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y


def display(X, Y, pred=None):
    for i in range(X.shape[0]):
        if ( pred is not None and np.argmax(Y[i])!= np.argmax(pred[i]) ):
            plt.plot( X[i,0], X[i,1], 'ro', color='red')
        else:
            if (np.argmax(Y[i])==1):
                plt.plot(X[i, 0], X[i, 1], 'ro', color='green')
            else:
                plt.plot(X[i, 0], X[i, 1], 'ro', color='blue')
    plt.show()

x_data,d_data = next_batch(128)
display(x_data, d_data)


# Initialisation des poids et biais
input_layer_neurons = x_data.shape[1]  # Nombre de caractéristiques d'entrée
hidden_layer_neurons = 4  # Nombre de neurones dans la couche cachée
output_neurons = 2  # Nombre de neurones de sortie

# Poids et biais
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Paramètres d'apprentissage
learning_rate = 0.1
epochs = 20000

# Entraînement du réseau
for epoch in range(epochs):
    x_data, d_data = next_batch(128)

    # Propagation avant
    a1 = np.dot(x_data, W1) + b1
    h1 = sigmoid(a1)
    
    a2 = np.dot(h1, W2) + b2
    h2 = sigmoid(a2)
    predicted_output = h2
    
    loss = mse(predicted_output, d_data)        # ne sert que pour affichage

    # Rétropropagation
    # étage 2
    delta2 = mse_prime(predicted_output, d_data,) * sigmoid_derivative(a2)
    dw2 = h1.T.dot(delta2) 
    db2 = np.sum(delta2, axis=0, keepdims=True)

    # étage 1
    delta1 = delta2.dot(W2.T) * sigmoid_derivative(a1)
    dw1 = x_data.T.dot(delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Mise à jour des poids et biais
    W2 -= dw2 * learning_rate
    b2 -= db2 * learning_rate
    W1 -= dw1 * learning_rate
    b1 -= db1 * learning_rate

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Affichage des résultats
print("Poids après entraînement :")
print("W1 :", W1)
print("W2 :", W2)
print("Biais après entraînement :")
print("b1 :", b1)
print("b2 :", b2)

print("Sortie prédite ==> display")
# nouveau batch pour test
x_data, d_data = next_batch(512)
# Propagation avant
a1 = np.dot(x_data, W1) + b1
h1 = sigmoid(a1)
a2 = np.dot(h1, W2) + b2
h2 = sigmoid(a2)
display(x_data, d_data, pred=h2)
for i in range(len(h2)):
    print("Entrée :", x_data[i], "Sortie prédite :", h2[i], "Sortie attendue :", d_data[i])
    

# for i in range(5):
#     X,Y = next_batch(128)

