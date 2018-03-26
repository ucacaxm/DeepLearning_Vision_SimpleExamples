# Various Simple Examples of Deep Learning / Vision

* Deep Learning with [Keras](https://keras.io) used on top of [TensorFlow](https://www.tensorflow.org)
* Vision with [DLib](http://dlib.net)

## Installation

Install [Anaconda](https://www.anaconda.com/download/), take the one with python 3 and then launch an Anaconda prompt and create your working env nammed 'p35'
* conda create -n p35 python=3.5 
* activate p35

NumPy
* conda install -c anaconda numpy 

MatPlotLib
* conda install -c conda-forge matplotlib
* conda install -c conda-forge/label/broken matplotlib
* conda install -c conda-forge/label/testing matplotlib
* conda install -c conda-forge/label/rc matplotlib 

DLib
* conda install -c menpo dlib 

TensorFlow
* pip install --ignore-installed --upgrade tensorflow 
or
* pip install --ignore-installed --upgrade tensorflow-gpu 




## Classifier in src/classifier

#Classify 2D points (x,y). Points upper than sin(x) are from class 1 (y>sin(x)), points lower than sin(x) are from class 2 (y<sin(x)).
* classifier_keras.py: with keras, simple network
* classifier_keras_v2-functionnal.py: more elaborate network, functionnal representation of keras
* classifier_keras_v2-sequential.py: more elaborate network, sequential representation of keras

#Images
* classifier_image_CNN_keras: Convolution neural network (ConvNet) with keras
* classifier_generator_tf_mnist.py (BEGONIN Florian 11400915 / GRANDJEAN Valentin 11402835)
Classification ET génération d'image (une sorte d'auto-encoder sur des images)

L'execution du fichier mnist.py permet d'entrainer un réseau neuronal convolutif sur la base de données MNIST.
Après la phase d'entrainement des réseaux neuronaux il est possible de vérifier les résultats obtenus.
Taper 1 dans la console testera le classifier en lui transmettant une image. Le résultat affiché
correspond à l'image ayant été choisie dans la base de donnée, son label ainsi que les prédictions du
réseau.
Taper 2 dans la console testera le réseau inverse en lui demandant de créer une représentation de chaque
chiffre puis passera cette image dans le classifier pour valider la reconaissance de chaque image.
Taper 3 mettra fin à l'execution du programme.




## GAN in src/gan
* classifier_generator_tf_mnist: learn a classifier + a generator of images from the number as input
* gan.py: GAN learn to generate 2D points (x,y) positionned above sin(x). 
=> does not work yet !



## DLib
* dlib_facial_landmarks.py
* dlib_video_facial_landmarks.py

=> test of DLib to get landmarks on faces from photos and video


## Alpha0 on a starship mini game
* starship.py: the mini game. The score of each particles depend on the distance to the target. Action is 2D forces applyed to the particle.
* mcts.py: the continuous mcst algo (it is a really poor implementation of mcts)
* alpha0.ps: use mcts.py to compute couple (observation, action), the mcts provides several config, each node of the tree. And train a network to act according to the osbservation.

