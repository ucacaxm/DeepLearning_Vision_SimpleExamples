#!/usr/bin/env python
import logging
import math
import random
import time

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
        self.action = np.copy(action)
        self.observation = np.copy(observation)
        self.parent = parent
        self.reward = reward
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        s = "(Node) children=%d; visits=%d; reward=%f av_reard=%f action=%s obs=%s" \
        % (len(self.children), self.visits, self.reward, self.reward/self.visits, self.action, self.observation)
        return s

    def __str__(self):
        return self.__repr__()



class MCTS:
    def __init__(self, game):
        # MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
        self.game = game
        self.root = None
        self.budget = 1000
        self.SCALAR = 1 / math.sqrt(2.0)
        self.max_node_size_in_batch_count = 1
        self.SIZE_OF_BATCH = game.sizeOfBatch()

    def UCTSearch(self, observation, budget=1000, batch_count_per_node=1):          # the main algo
        self.root = Node( observation )
        self.budget = budget
        self.max_node_size_in_batch_count = batch_count_per_node
        for iter in range(int(self.budget)):
            if iter%200==0:
                print("ite "+str(iter)+": root=>"+ str(self.root) )
            front=self.TreePolicy(self.root)
            self.ExpandByDefaultPolicyAndBackup(front)
        return self.BestChild(self.root, 0)

    def is_fully_expanded(self, node):
        return (len(node.children) >= self.max_node_size_in_batch_count * self.SIZE_OF_BATCH)

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
            elif score>bestscore:
                bestchildren=[c]
                bestscore=score
        if len(bestchildren)==0:
            logger.warning("OOPS: no best child found, probably fatal")
        return random.choice(bestchildren)


    def ExpandByDefaultPolicyAndBackup(self, node):  # create a new node at the leaf of the tree and run N simulation from it
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

        # BACKUP
        for i in range(self.game.sizeOfBatch()):
            self.Backup( node.children[i], self.game.reward(i)/nStep )


    def Backup(self, node, reward):       # back propagation of the reward and the count
        cur = node
        while cur!=None:
            #node.update( reward )
            cur.reward += reward
            cur.visits += 1
            cur=cur.parent


    def PlayTreePolicy(self):
        node = self.root
        if node==None:
            return
        self.game.reset()
        self.game.setReward(0.0)
        self.game.setObservationForAllBatch(node.observation)
        self.game.setPaused(True)
        print("Start game reward=" + str(self.game.reward(0)) + "   " + str(node))
        while not self.game.isQuit():
            self.game.manageEvent()
            self.game.drawSceneMenuAndSwap()
            if not self.game.paused():
                if  len(node.children)>0:
                    node = self.BestChild(node, 0)
                    self.game.setActionForAllBatch(node.action)
                    self.game.stepBatch()
                    print("game reward=" + str(self.game.reward(0)) + "   " + str(node))
                else:
                    node = self.root
                    #self.game.reset()
                    #self.game.setReward(0.0)
                    #self.game.setObservationForAllBatch(node.observation)
                    self.game.setPaused(True)
                    print("game...end=>reset: root="+str(node))


    def PlayRandomPolicy(self, n):
        node = self.root
        # if node==None:
        #     return
        self.game.setReward(0.0)
        self.game.setObservationForAllBatch(self.game.observation(0) )
        while not self.game.isQuit():
            self.game.setRandomAction(0)
            self.game.setActionForAllBatch( self.game.action(0) )
            self.game.stepBatch()
            print( "game reward="+ str(self.game.reward(0))+"   "+str(node) )
            self.game.manageEvent()
            self.game.drawSceneMenuAndSwap()
        print("PlayRandomPolicy...done")


if __name__ == "__main__":
    starship = starship.Starship()
    starship.init("starship", 800, 600, 15, 15)
    mcts = MCTS(starship)

    #mcts.PlayRandomPolicy(10)
    starship.reset()
    print( starship.observation(0) )
    print("--------------------------------")
    mcts.UCTSearch( starship.observation(0), 5000, 1 )
    mcts.PlayTreePolicy()
