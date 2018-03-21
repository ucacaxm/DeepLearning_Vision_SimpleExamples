################################################################"
# alpha0
# status: not working
################################################################"

import os
import sys
from time import time

import numpy as np

sys.path.append(os.getcwd() + '\\..\\build\\src\\Debug')
sys.path.append(os.getcwd() + '.')
print("Add path to _pysimea.pyd, sys.path=")
print(sys.path)

# os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

import starship
import mcts


######################################################################################################################
class Alpha0:
    def __init__(self, game, optimizer):
        # parameters for simu
        # self.viewer = sim.PySimeaViewer()
        # self.viewer = gym_wrapper_to_simea.Gym_wrapper()
        self.game = game
        self.optimizer = optimizer

        self.game.init("Game", 800, 600, 50, 50, 100)
        print("py! create batch/action/observation array:",
              self.game.sizeOfBatch(), "/",
              self.game.sizeOfActionArray(), "/",
              self.game.sizeOfObservationArray())

        # parameters for learning
        self.epsilon = .1  # exploration
        self.epoch = 10
        self.action_range = 5.0  # 0.01

        self.input = Input(shape=(self.game.sizeOfObservationArray(),) )
        self.encoded = Dense(64, activation='tanh')(self.input)
        #self.encoded = Convolution1D(32,2, padding='same')(self.encoded)
        self.encoded = Dense(128, activation='tanh')(self.encoded)

        self.decoded = Dense(128, activation='tanh')(self.encoded)
        self.decoded = Dense(64, activation='tanh')(self.decoded)
        self.decoded = Dense(self.game.sizeOfActionArray())(self.decoded)

        # this model maps an input to its reconstruction
        self.model = Model(self.input, self.decoded)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        print("py! Alpha0::init...OK")


    def load(self):
        self.model = load_model('alpha0.h5')
        print("load")

    def save(self):
        self.model.save('alpha0.h5')
        print("save")

    def modelAction(self, obs):
        action = self.model.predict(obs)
        return action

    def train(self):
        for epoch in range(5):
            self.game.resetRandomlyAllBatch()
            obs = np.copy( self.game.observationForAllBatch() )
            act = np.empty( (self.game.sizeOfBatch(),self.game.sizeOfObservationArray()), dtype=float )
            for i in range(self.game.sizeOfBatch):
                act[i] = np.copy( self.optimizer.search( obs[i] ) )
            r = self.model.train_on_batch(obs, act)
            print("train_on_batch: epoch=", epoch, " loss=", r)
        print("training...done")

    def stepModelAction(self):
        obs = self.game.observation(0)
        act = modelAction(obs)
        self.game.setAction( 0, act)
        print("model for 0")

    def run(self):
        useModel = False
        while not self.game.isQuit():
            if self.game.eventKey( ord('l') ):
                self.load()
            elif self.game.eventKey(ord('s')):
                self.save()
            elif self.game.eventKey( ord('t') ):
                self.train()
            elif self.game.eventKey( ord('m') ):
                useModel = True
            elif self.game.eventKey(ord('r')):
                useModel = False

            self.game.setRandomActionForAllBatch()
            if useModel:
                self.stepModelAction()

            self.game.stepBatch()
            self.game.manageEvent()
            self.game.drawSceneMenuAndSwap()



if __name__ == "__main__":
    print("Start...")
    game = starship.Starship()
    opti = mcts.MCTS(game, 3000, 1)
    a0 = Alpha0(game, opti)
    a0.run()
