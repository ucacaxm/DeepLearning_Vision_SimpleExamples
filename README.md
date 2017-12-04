# Various Simple Examples of Deep Learning / Vision

* Deep Learning with (Keras)[https://keras.io/] used on top of [TensorFlow](https://www.tensorflow.org/)
* Vision with [DLib](http://dlib.net/)

## Installation

Install [Anaconda](https://www.anaconda.com/download/), take the one with python 3 and then
* conda create -n p35 python=3.5 
* activate p35
* conda install -c anaconda numpy 
* conda install -c menpo dlib 
* pip install --ignore-installed --upgrade tensorflow 
or
* pip install --ignore-installed --upgrade tensorflow-gpu 




## Classifier
* classifier.py: classify 2D points (x,y). Points upper than sin(x) are from class 1 (y>sin(x)), points lower than sin(x) are from class 2 (y<sin(x)).

=> tensorflow and keras



## GAN
* gan.py: GAN learn to generate 2D points (x,y) positionned above sin(x). 

=> does not work yet !



## DLib
* dlib_facial_landmarks.py
* dlib_video_facial_landmarks.py

=> test of DLib to get landmarks on faces from photos and video
