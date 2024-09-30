# %% [markdown]
# L'objectif de ce TP est d'implémenter par vous même l'apprentissage de réseaux de neurones simples. 
# Cette prise en main sera formatrice pour utiliser des frameworks plus évolués (comme PyTorch) où l'apprentissage est automatisé.
# 
# Nous allons utiliser la base de données image MNIST, constituée d'images de caractères 
# manuscrits (60000 images en apprentissage, 10000 en test). L'objectif est de reconnaitre 
# le chiffre par un réseau de neurones. Les images sont en niveau de gris de taille 28x28. 
# Le vecteur d'entrée est de 784 valeurs et un vecteur de 10 valeurs en sortie.
# 
# Voici le code pour récupérer les images avec sklearn :

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml

# %%
def load_mnist_data():
  """ Télécharger les données MNIST dans un numpy array avec sklearn """
  mnist = fetch_openml('mnist_784', version=1)

  # Extraire les données et les étiquettes
  X, y = mnist['data'], mnist['target']

  # Conversion en arrays numpy
  X = X.to_numpy().astype(np.float32)  # Convertir les features en float32 numpy array
  y = y.to_numpy().astype(int)         # Convertir les labels en entiers

  # Normalisation des données (optionnel)
  X /= 255.0

  # Reshape les images en 28x28 pour correspondre au format des images
  #X = X.reshape(-1, 28, 28)
  X = X.reshape(-1, 28*28)

  # Afficher les formes des arrays
  print(f'X shape: {X.shape}, y shape: {y.shape}')
  return X, y

X,Y = load_mnist_data()
print("x: type=",type(X), " shape=",np.shape(X))
print("y: type=",type(Y), " shape=",np.shape(Y))


# %% [markdown]
# Visualisation de quelques images

# %%
def visu(X_train):
  plt.figure(figsize=(28, 28))
  for i in range(5):
    plt.subplot(10,20,i+1)
    plt.imshow( 4*(X_train[i,:].reshape([28,28])), cmap='gray')
    plt.axis('off')
  plt.show()


#visu(X)

# %% [markdown]
# Quelques fonctions utiles.

# %%
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """Dérivée de la fonction sigmoid."""
    return sigmoid(x) * (1.0 - sigmoid(x))


def to_one_hot(y, k):
    """Convertit un entier en vecteur "one-hot".
    to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
    """
    one_hot = np.zeros(k)
    one_hot[y] = 1
    return one_hot

# %% [markdown]
# La définition d'un réseau complétement connecté (FC fuly connected) est une succession 
# de couches (layer). Un layer contient une matrice W des poids et bias : W.x_input + bias. 
# La fonction d'activation est sigmoid ici. Voir le cours pour les détails.

# %%
class Layer:
    """
      Une seule couche de neurones.
    """
    def __init__(self, size, input_size):
      """
        `size` est le nombre de neurones dans la couche.
        `input_size` est le nombre de neurones dans la couche précédente.
      """
      self.size = size
      self.input_size = input_size

      # Les poids sont représentés par une matrice de n lignes et m colonnes.
      # n = le nombre de neurones, m = le nombre de neurones dans la couche précédente.
      self.weights = np.random.randn(size, input_size)

      # Un biais par neurone: y = W.x + B (B=bias)
      self.bias = np.random.randn(size)


    def forward(self, data):
      """
        Résultat du calcul de chaque neurone.
        `data` est un vecteur de longueur `self.input_size`
        retourne un vecteur de taille `self.size`.
      """
      aggregation = self.aggregation(data)
      activation = self.activation(aggregation)
      return activation


    def aggregation(self, data):
      """
        Calcule W.data + bias
      """
      return np.dot(self.weights, data) + self.bias


    def activation(self, x):
        """
          Passe les valeurs aggrégées dans la fonction d'activation.
        """
        return sigmoid(x)


    def activation_prime(self, x):
        return sigmoid_prime(x)


    def update_weights(self, gradient, learning_rate):
        """ Mise à jour des poids à partir du gradient (algo du gradient) """
        self.weights -= learning_rate * gradient


    def update_biases(self, gradient, learning_rate):
        """ Idem mais avec les biais """
        self.bias -= learning_rate * gradient

# %% [markdown]
# Le réseau complet est une succession de couches. La fonction feedforward propage les données.
# La fonction predict retourne l'index du neurone de sortie qui a la plus grande valeur (la classe).
# La fonction evaluate retourne la performance du réseau sur un set de données.
# La fonction train est l'entraînement du réseau. On fait tourner l'algo de rétropropagation
#
#
# %%
class Network:
    """Un réseau constitué de couches de neurones."""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []


    def add_layer(self, size):
        if len(self.layers) > 0:
            input_dim = self.layers[-1].size
        else:
            input_dim = self.input_dim
        self.layers.append(Layer(size, input_dim))


    def feedforward(self, input_data):
        """ Propage les données d'entrée d'une couche à l'autre """
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation


    def predict(self, input_data):
        """ Passe input_data dans le réseau (feedForward) et retourne l'index du neurone de 
            sortie qui a la plus grande valeur (qui est la classe sélectionnée par le réseau). """
        return np.argmax(self.feedforward(input_data))


    def evaluate(self, X, Y):
        """ Évalue la performance du réseau à partir d'un set d'exemples. Retourne un nombre entre 0 et 1."""
        results = 0.0
        n = 0
        for (x,y) in zip(X,Y):
            results += (self.predict(x.reshape( 28*28 )  ) == y)
            n += 1
        #results = [1 if self.predict(x) == y else 0 for (x, y) in zip(X, Y)]
        accuracy = results / n
        return accuracy


    def train(self, X, Y, steps=30, learning_rate=0.3, batch_size=10):
        """
            Fonction d'entraînement du modèle.
            La rétropropagation tourne sur un certain nombre d'exemples (batch_size) avant 
            de calculer un gradient moyen, et de mettre à jour les poids.
        """
        n = Y.size
        for i in range(steps):
            X, Y = shuffle(X, Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = X[batch_start:batch_start + batch_size], Y[batch_start:batch_start + batch_size]
                self.train_batch(X_batch, Y_batch, learning_rate)


    def train_batch(self, X, Y, learning_rate):
        """     Cette fonction combine les algos du retropropagation du gradient + gradient descendant """
        # Initialise les gradients pour les poids et les biais.
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.bias.shape) for layer in self.layers]

        # On fait tourner l'algo de rétropropagation pour calculer les
        # gradients un certain nombre de fois. On fera la moyenne ensuite.
        for (x, y) in zip(X, Y):            # zip produit un tableau de couple (x,y) à partir de 2 tableaux
            new_weight_gradient, new_bias_gradient = self.backprop(x, y)
            weight_gradient = [wg + nwg for wg, nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg, nbg in zip(bias_gradient, new_bias_gradient)]

        # C'est ici qu'on calcule les moyennes des gradients calculés
        avg_weight_gradient = [wg / Y.size for wg in weight_gradient]
        avg_bias_gradient = [bg / Y.size for bg in bias_gradient]

        # Il ne reste plus qu'à mettre à jour les poids et biais en utilisant l'algo du gradient descendant.
        for layer, weight_gradient, bias_gradient in zip(self.layers, avg_weight_gradient, avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)


    def backprop(self, x, y):
        """
        # x=input, y=output
        # L'algorithme de rétropropagation du gradient. C'est là que tout le boulot se fait. 
        # Une passe vers l'avant puis une passe vers l'arrière
        # On profite de la passe vers l'avant pour stocker les calculs
        # intermédiaires, qui seront réutilisés durant la passe vers l'arrière.
        # regarder le code ici pour vous aider, en plus des slides du cours :
        # http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation
        """
        aggregations = []
        activation = x
        activations = [activation]

        # Propagation pour obtenir la sortie (même code que feedForward 
        # mais on stocke les valeurs intermédiaires dans aggregations et activations)
        for layer in self.layers:
            aggregation = layer.aggregation(activation)
            aggregations.append(aggregation)
            activation = layer.activation(aggregation)
            activations.append(activation)
        #print("backprop: agg=",len(aggregations), " acti=", len(activations))

        # Calcul pour la dernière couche (gradient de l'erreur par rapport à l'activation)
        target = to_one_hot(int(y), 10)
        delta = self.compute_cost_derivative(activations[-1], target) 
        #* self.layers[-1].activation_prime( aggregation )
        bias_gradient = [ delta ]
        delta2d = delta[:, np.newaxis]                        # (dim_output) => (1,dim_output)
        prev_activation = activations[-2][np.newaxis,:]     # (dim_prev_activation) => (dim_prev_activation,1)
        weight_gradient = [ np.dot(delta2d, prev_activation)  ]
        #weight_gradient = [ np.dot(self.layers[-1].weights.transpose(), delta)  ]

        # Phase de rétropropagation pour calculer les deltas de chaque couche
        # Puis avec les deltas, on calcule les gradients de w et de b (qui sont les résultats de la fonction)
        nb_layers = len(self.layers)                # ici 2: couche cachée + sortie
        for k in range( nb_layers-1, 0, -1):        # de nb_layers-2 ... à ... 0 (car n-1 est déjà fait)
            layer = self.layers[k-1]                # comme il n'y a que 2 layers, le layer 0 est l'activation 1
            next_layer = self.layers[k]

            activation_prime = layer.activation_prime(aggregations[k-1])                  # g'(a_k)
            delta = np.dot(next_layer.weights.transpose(), delta) * activation_prime      # 
            #deltas.append(delta)
            bias_gradient.insert(0, delta)

            prev_activation = activations[k-1][np.newaxis,:]
            delta2d = delta[:, np.newaxis]                        # (dim) => (1,dim_output)
            wg = np.dot(delta2d, prev_activation)
            # prev_activation = activations[k]
            # wg = np.dot( prev_activation.transpose(), delta)
            weight_gradient.insert( 0, wg )

        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])        # dim(delta)=dim(y)
        # nabla_b[-1] = delta
        # nabla_w[-1] = np.dot(delta, activations[-2].transpose())                        # ?
        # for l in range(2, self.num_layers):
        #     z = zs[-l]
        #     sp = sigmoid_prime(z)
        #     delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return weight_gradient, bias_gradient


    #def compute_cost_derivative(self, aggregation, activation, target):
    def compute_cost_derivative(self, activation, target):
        """    
        # Calcule Grad_E_a pour la dernière couche, en utilisant
        # la sortie du réseau (aggregation et activation) et la valeur cible.
        # a= aggregation, h=activation, target = cible = y
        """
        return  activation - target

# %% [markdown]
# Le main

# %%
# La base de données est coupée en deux : entraînement et test.
# X, Y = load_mnist_data()      # déjà chargée avant
X_train, Y_train = X[:60000], Y[:60000]
X_test, Y_test = X[60000:], Y[60000:]

net = Network(input_dim=784)
net.add_layer(200)
net.add_layer(10)

accuracy = net.evaluate(X_test, Y_test)
print('Performance initiale : {:.2f}%'.format(accuracy * 100.0))

for i in range(5):
    net.train(X_train, Y_train, steps=1, learning_rate=3.0)
    accuracy = net.evaluate(X_test, Y_test)
    print('Nouvelle performance : {:.2f}%'.format(accuracy * 100.0))


