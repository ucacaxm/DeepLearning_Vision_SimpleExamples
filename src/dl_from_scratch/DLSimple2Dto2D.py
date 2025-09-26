import numpy as np
import math
import matplotlib.pyplot as plt
import random


# Fonction d'activation (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)



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
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Paramètres d'apprentissage
learning_rate = 0.1
epochs = 1000

# Entraînement du réseau
for epoch in range(epochs):
    # Propagation avant
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    # Calcul de l'erreur
    error = Y - predicted_output
    
    # Rétropropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Mise à jour des poids et biais
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

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

