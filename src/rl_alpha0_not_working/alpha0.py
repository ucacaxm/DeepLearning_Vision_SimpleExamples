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

import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

#import pysimea as sim
import json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

import starship


######################################################################################################################
class Alpha0:
    def __init__(self, game):
        # parameters for simu
        # self.viewer = sim.PySimeaViewer()
        # self.viewer = gym_wrapper_to_simea.Gym_wrapper()
        self.game = game

        self.game.init("Game", 1300, 600, 50, 50, 100)
        print("py! create batch/action/observation array:",
              self.game.sizeOfBatch(), "/",
              self.game.sizeOfActionArray(), "/",
              self.game.sizeOfObservationArray())

        # parameters for learning
        self.epsilon = .1  # exploration
        self.epoch = 10
        self.nite = 1000
        self.action_range = 5.0  # 0.01
        self.m_quit = False

        self.reward = np.empty(self.game.sizeOfBatch(), dtype=float)
        self.done = np.empty(self.game.sizeOfBatch(), dtype=bool)
        self.observation = np.empty((self.game.sizeOfBatch(), self.game.sizeOfObservationArray()), dtype=float)
        self.action = np.empty((self.game.sizeOfBatch(), self.game.sizeOfActionArray()), dtype=float)

        for i in range(self.game.sizeOfBatch()):
            self.reward[i] = 0
            self.done[i] = False
        self.observation = self.game.reset()
        self.randomAction()

        print("py! batch_size=", self.game.sizeOfBatch(), "observation.shape=", self.observation.shape,
              "  action.shape=", self.action.shape)

        print("py! Alpha0::init...OK")

    def randomActionOne(self):
        act = np.empty((self.game.sizeOfActionArray()), dtype=float)
        for j in range(self.game.sizeOfActionArray()):
            act[j] = -self.action_range + self.action_range * 2.0 * np.random.random_sample()
        return act

    def randomAction(self):
        for i in range(self.game.sizeOfBatch()):
            self.action[i] = self.randomActionOne()

    def stepRandomAction(self):
        self.randomAction()
        self.reward, self.done = self.game.stepBatch(self.action, self.observation)
        for i in range(self.game.sizeOfBatch()):
            if self.done[i]:
                self.observation[i] = self.game.resetOne(i)

    def modelAction(self):
        self.action = self.model.predict(self.observation)

    def stepModelAction(self):
        self.modelAction()
        self.reward, self.done = self.game.stepBatch(self.action, self.observation)
        for i in range(self.game.sizeOfBatch()):
            if self.done[i]:
                self.observation[i] = self.game.resetOne(i)

    def print_shape(self, onlyShape=False):
        print("=====================================")
        print("py! batch size=", self.game.sizeOfBatch())
        print("py! observation shape=", self.observation.shape)
        print("py! action shape=", self.action.shape)
        if (not onlyShape):
            print("py! observation=", self.observation, " type=", type(self.observation))
            print("py! action=", self.action, " type=", type(self.action))
        print("=====================================")

    def train(self):
        print("train is not implemented")

    def run(self):
        self.observation = self.game.reset()
        while not self.m_quit:
            if not self.game.paused():
                if self.game.isLearning():
                    self.train()
                else:
                    self.stepRandomAction()
                    # self.stepModelAction()

            self.m_quit = self.game.manageEvent()
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
