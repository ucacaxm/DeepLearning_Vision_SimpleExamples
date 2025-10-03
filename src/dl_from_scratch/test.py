import numpy as np
import math
import matplotlib.pyplot as plt
import random


# Fonction d'activation (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction Sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



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

X,Y = next_batch(128)
display(X, Y)


# Initialisation des poids et biais
input_layer_neurons = X.shape[1]  # Nombre de caractéristiques d'entrée
hidden_layer_neurons = 4  # Nombre de neurones dans la couche cachée
output_neurons = 2  # Nombre de neurones de sortie

# Poids et biais
W1 = np.random.uniform(-0.5, 0.5, size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.zeros((1, hidden_layer_neurons))
W2 = np.random.uniform(-0.5, 0.5, size=(hidden_layer_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

# Paramètres d'apprentissage
learning_rate = 0.1
epochs = 20000

def mse(y, d):
    return 0.5 * np.mean((y - d)**2)

def mse_prime(y, d):
    """
    Dérivée de la MSE par rapport à y
    """
    N = y.shape[0]
    return (y - d) / N

# Entraînement du réseau
for epoch in range(epochs):
    X,Y = next_batch(128)
    # TODO Propagation avant : entrée X et calcul de la sortie prédite predicted_output
    a1 = np.dot(X, W1) + b1
    h1 = sigmoid(a1)
    
    a2 = np.dot(h1, W2) + b2
    predicted_output = sigmoid(a2)
    
    # Après ceci vous pouvez lancer le code, il fonctionnera (prediction et affichage) mais le réseau ne s'entraînera pas.


    # TODO Calcul de l'erreur
    error = mse(predicted_output, Y)
    
    # TODO Rétropropagation de l'erreur     
    delta2 = mse_prime(predicted_output, Y) * sigmoid_derivative(a2) # δ2 = dL/da2
    dw2 = np.dot(h1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a1)
    dw1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    
    # TODO Mise à jour des poids et biais
    W2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {error:.6f}")



# Affichage des résultats
print("Poids après entraînement :")
print("W1 :", W1)
print("W2 :", W2)
print("Biais après entraînement :")
print("b1 :", b1)
print("b2 :", b2)

print("Sortie prédite :")
#print(predicted_output)
display(X, Y, pred=predicted_output)
for i in range(len(predicted_output)):
    print("Entrée :", X[i], "Sortie prédite :", predicted_output[i], "Sortie attendue :", Y[i])
    

# for i in range(5):
#     X,Y = next_batch(128)
