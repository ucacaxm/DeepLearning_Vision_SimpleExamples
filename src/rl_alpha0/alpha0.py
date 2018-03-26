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
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.models import load_model

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
        self.obs = np.empty((0, self.game.sizeOfObservationArray()), dtype=float)
        self.act = np.empty((0, self.game.sizeOfActionArray()), dtype=float)

        self.game.init("Game", 800, 600, 50, 50, 100)
        print("py! create batch/action/observation array:",
              self.game.sizeOfBatch(), "/",
              self.game.sizeOfActionArray(), "/",
              self.game.sizeOfObservationArray())

        # parameters for learning
        self.epsilon = .1  # exploration
        self.epoch = 10
        self.action_range = 5.0  # 0.01

        act = 'tanh'
        self.input = Input(shape=(self.game.sizeOfObservationArray(),) )
        self.hidden = Dense(32, activation=act)(self.input)
        self.hidden = Dense(64, activation=act)(self.hidden)
        self.hidden = Dense(64, activation=act)(self.hidden)
        self.hidden = Dense(32, activation=act)(self.hidden)
        self.hidden = Dropout(0.2 )(self.hidden)
        self.hidden = Dense(self.game.sizeOfActionArray())(self.hidden)

        # this model maps an input to its reconstruction
        self.model = Model(self.input, self.hidden)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        print("py! Alpha0::init...OK")


    def loadModel(self):
        outfile = "alpha0_"+self.game.name()+".h5"
        self.model = load_model(outfile)
        print("load: "+outfile)

    def saveModel(self):
        outfile = "alpha0_"+self.game.name()+".h5"
        self.model.save(outfile)
        print("save: "+outfile)

    def modelAction(self, obs):
        action = self.model.predict( np.array([obs]) )
        return action[0]

    def saveOptiData(self):
        outfile = "alpha0_"+self.game.name()+".npz"
        np.savez(outfile, obs=self.obs, act=self.act)
        print("save("+outfile+"===>obsSize="+str(len(self.obs))+" actSize="+str(len(self.act)))

    def loadOptiData(self):
        outfile = "alpha0_"+self.game.name()+".npz"
        npz = np.load(outfile)
        self.obs = npz['obs']
        self.act = npz['act']
        print("load(" + outfile + "===>files="+str(npz.files)+"obsSize=" + str(len(self.obs)) + " actSize=" + str(len(self.act)))

    def computeOptiData(self, n):
        print("computeOptiData..."+str(n))
        for i in range(n):
            self.game.resetRandomlyOneAgent(0)
            self.optimizer.search( self.game.observation(0) )
            o,a = self.optimizer.batchOfObservationAction()
            self.obs = np.append( self.obs, o, axis=0 )
            self.act = np.append( self.act, a, axis=0 )
        print("computeOptiData...done("+str(n)+")")

    def train(self):
        self.model.fit( self.obs, self.act, 32, 100)
        #r = self.model.train_on_batch(obs, act)
        #print("train_on_batch: epoch=", epoch, " loss=", r, "sizeOfObsAct=", str(len(obs)),"/",str(len(act)))
        print("training...done")


    def stepModelAction(self):
        for i in range(self.game.sizeOfBatch()):
            obs = self.game.observation(i)
            act = self.modelAction(obs)
            self.game.setAction( i, act)
        #print("model")

    def help(self):
        self.game.help()
        print("alpha0")
        print("   l: load model")
        print("   s: save model")
        print("   t: train model")
        print("   m: load opti data")
        print("   f: save opti data")
        print("   c: compute opti data: 1 sample")
        print("   v: compute opti data: 10 samples")
        print("obsSize=" + str(len(self.obs)) + " actSize=" + str(len(self.act)))


    def run(self):
        while not self.game.isQuit():

            if not self.game.paused():
                if self.game.noAction:
                    self.game.setZeroActionForAllBatch()
                else:
                    self.stepModelAction()
                self.game.stepBatch()

            done = False
            while not done:
                carac = self.game.manageEvent()
                if carac==None:
                    done = True
                elif carac == 'l':
                    self.loadModel()
                elif carac == 's':
                    self.saveModel()
                elif carac == 'm':
                    self.loadOptiData()
                elif carac == 'f':
                    self.saveOptiData()
                elif carac == 't':
                    self.train()
                elif carac == 'c':
                    self.computeOptiData(1)
                elif carac == 'v':
                    self.computeOptiData(10)
                elif carac == 'h':
                    self.help()
            self.game.drawSceneMenuAndSwap()

        print("alpha0::run...done")



if __name__ == "__main__":
    print("Start...")
    game = starship.Starship()
    opti = mcts.MCTS(game, 2000, 1)
    a0 = Alpha0(game, opti)
    a0.run()
