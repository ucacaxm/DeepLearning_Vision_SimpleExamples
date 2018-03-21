################################################################"
# alpha0
# status: not working
################################################################"

import sys
import os

sys.path.append(os.getcwd() + '\\..\\build\\src\\Debug')
sys.path.append(os.getcwd() + '.')
print("Add path to _pysimea.pyd, sys.path=")
print(sys.path)

# os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.layers import Input, Dense, Convolution1D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

import numpy as np
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

        self.game.init("Game", 1300, 600, 50, 50, 100)
        print("py! create batch/action/observation array:",
              self.game.sizeOfBatch(), "/",
              self.game.sizeOfActionArray(), "/",
              self.game.sizeOfObservationArray())

        # parameters for learning
        self.epsilon = .1  # exploration
        self.epoch = 10
        self.action_range = 5.0  # 0.01

        self.input = Input(shape=(self.game.sizeOfObservationArray()) )
        self.encoded = Dense(64, activation='tanh')(self.input)
        #self.encoded = Convolution1D(32,2, padding='same')(self.encoded)
        self.encoded = Dense(128, activation='tanh')(self.encoded)

        self.decoded = Dense(128, activation='tanh')(self.encoded)
        self.decoded = Dense(64, activation='tanh')(self.decoded)
        self.decoded = Dense(self.game.sizeOfActionArray())(self.decoded)

        # this model maps an input to its reconstruction
        self.model = Model(self.input, self.decoded)

        print("py! Alpha0::init...OK")


    def modelAction(self):
        self.action = self.model.predict(self.observation)

    def stepModelAction(self):
        self.modelAction()
        self.reward, self.done = self.game.stepBatch(self.action, self.observation)
        for i in range(self.game.sizeOfBatch()):
            if self.done[i]:
                self.observation[i] = self.game.resetOne(i)

    def load(self, filename):

    def save(self, filename):

    def train(self):
        print("train is not implemented")

    def stepModelAction(self):
        print("model")

    def run(self):
        self.observation = self.game.reset()
        while not self.b_quit:
            if not self.game.paused():
                if self.game.eventKey( ord('l') ):
                    self.train()
                elif self.game.eventKey( ord('m') ):
                    self.stepModelAction()
                else:
                    self.stepRandomAction()

            self.b_quit = self.game.manageEvent()
            self.game.drawSceneMenuAndSwap()
            # print("py! Save&close...")
            # self.model.save_weights("debug/model.h5", overwrite=True)
            # with open("debug/model.json", "w") as outfile:
            #     json.dump(self.model.to_json(), outfile)
            # self.viewer.close()
            # print("py! Save&close...OK")



if __name__ == "__main__":
    print("Start...")
    game = starship.Starship()
    a0 = Alpha0(game)
    a0.run()
