import numpy as np

# Fonction d'activation (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Données d'entrée (X) et sorties (y)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialisation des poids et biais
input_layer_neurons = X.shape[1]  # Nombre de caractéristiques d'entrée
hidden_layer_neurons = 4  # Nombre de neurones dans la couche cachée
output_neurons = 1  # Nombre de neurones de sortie

# Poids et biais
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Paramètres d'apprentissage
learning_rate = 0.1
epochs = 10000

# Entraînement du réseau
for epoch in range(epochs):
    # Propagation avant
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    # Calcul de l'erreur
    error = y - predicted_output
    
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
print(predicted_output)
