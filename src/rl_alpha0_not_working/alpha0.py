import math
import os
from time import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

#os.environ["KERAS_BACKEND"] = "theano"
from setuptools.command.saveopts import saveopts
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Convolution1D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD


