#!/usr/bin/env python
import time
import random
import math
import logging
import numpy as np
import starship

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
Code modified from https://github.com/haroldsultan/MCTS
=> adapted to continuous
"""

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MCTS')


class Node:
    def __init__(self, observation, action=None, reward=0, parent=None):
        self.visits = 1
        self.action = action
        self.observation = observation
        self.parent = parent
        self.reward = reward
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        s = "Node; children=%d; visits=%d; rewardFomParent=%f reward=%f" % (
        len(self.children), self.visits, self.rewardFromParent, self.reward)
        return s




class MCTS:
    def __init__(self, game):
        # MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
        self.game = game
        self.root = None
        self.budget = 1000
        self.SCALAR = 1 / math.sqrt(2.0)
        self.FULLY_EXTENDED_BATCH_COUNT = 2
        self.SIZE_OF_BATCH = game.sizeOfBatch()

    def UCTSearch(self, observation, budget=1000):          # the main algo
        self.root = Node( observation )
        self.budget = budget
        for iter in range(int(self.budget)):
            if iter%1000==99:
                logger.info("UCTSSearch: %d"%iter)
                logger.info(self.root)
            front=self.TreePolicy(self.root)
            self.ExpandByDefaultPolicyAndBackup(front)
        return self.BestChild(self.root, 0)

    def is_fully_expanded(self, node):
        return ( len(node.children) > self.FULLY_EXTENDED_BATCH_COUNT * self.SIZE_OF_BATCH )

    def TreePolicy(self, node):               # Selection of the last node (leaf) by following the tree
        #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
        while True:
            if len(node.children)==0:
                return node
            elif self.is_fully_expanded(node):
                node = self.BestChild(node, self.SCALAR)
            elif random.uniform(0,1)<.5:
                return node
            else:
                node = self.BestChild(node, self.SCALAR)
        return node

    #current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    def BestChild(self, node, scalar):        # select a best child of a node
        bestscore=0.0
        bestchildren=[]
        for c in node.children:
            exploit=c.reward/c.visits
            explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))
            score=exploit+scalar*explore
            if score==bestscore:
                bestchildren.append(c)
            if score>bestscore:
                bestchildren=[c]
                bestscore=score
        if len(bestchildren)==0:
            logger.warning("OOPS: no best child found, probably fatal")
        return random.choice(bestchildren)

    def ExpandByDefaultPolicyAndBackup(self, node):  # create a new node at the leaf of the tree and run N simulation from it
        if self.is_fully_expanded(node):
            logger.warning("OOPS: expand a node already expanded")

        # EXPAND
        self.game.setObservationForAllBatch( node.observation )
        self.game.setReward(0.0)
        self.game.setRandomActionForAllBatch()
        self.game.stepBatch()
        for i in range(self.game.sizeOfBatch()):
            o = self.game.observation(i)
            a = self.game.action(i)
            r = self.game.reward(i)
            ch = Node(o,a,r, node)
            node.add_child( ch )

        # SIMULATION
        nStep = 10
        for i in range(nStep):
            self.game.setRandomActionForAllBatch()
            self.game.stepBatch()
        for i in range(self.game.sizeOfBatch()):
            self.Backup( node.children[i], self.game.reward(i)/nStep )

    def Backup(self, node, reward):       # back propagation of the reward and the count
        while node!=None:
            #node.update( reward )
            node.reward += reward
            node.visits += 1
            node=node.parent


    def PlayTreePolicy(self):
        node = self.root
        if node==None:
            return
        while len(node.children)>0:
            self.game.setObservationForAllBatch(node.observation)
            node = self.BestChild(node, 0)
            self.game.setActionForAllBatch( node.action )
            self.game.stepBatch()
            print( "game reward="+ str(self.game.reward(0))+"  node_reward="+str(node.reward)+"  node_visit="+str(node.visits) )
            self.game.drawSceneMenuAndSwap()
            time.sleep(10)


if __name__ == "__main__":
    starship = starship.Starship()
    starship.init("starship", 800, 600, 15, 15)
    mcts = MCTS(starship)
    mcts.UCTSearch( starship.observation(0) )
    mcts.PlayTreePolicy()
    print("--------------------------------")
