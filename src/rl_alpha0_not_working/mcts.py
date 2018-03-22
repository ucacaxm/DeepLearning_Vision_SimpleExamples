#!/usr/bin/env python
import logging
import math
import random

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
    def __init__(self, game, budget=2000, max_node_size_in_batch_count=1):
        # MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
        self.game = game
        self.budget = budget
        self.max_node_size_in_batch_count = max_node_size_in_batch_count
        self.SCALAR = 0.115  #1 / math.sqrt(2.0)
        self.root = None

    def search(self, observation):          # the main algo
        self.root = Node( observation )
        for iter in range(int(self.budget)):
            if iter%200==0:
                print("ite "+str(iter)+": root=>"+ str(self.root) )
            front=self.TreePolicy(self.root)
            self.ExpandByDefaultPolicyAndBackup(front)
        return self.BestChild(self.root, 0)

    def is_fully_expanded(self, node):
        return (len(node.children) >= self.max_node_size_in_batch_count * self.game.sizeOfBatch())

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
        if len(bestchildren) == 0:
            return None
        else:
            return random.choice(bestchildren)


    def ExpandByDefaultPolicyAndBackup(self, node):  # create a new node at the leaf of the tree and run N simulation from it
        # EXPAND
        self.game.resetAllBatch( node.observation )
        self.game.setRewardForAllBatch(0.0)
        self.game.setRandomActionForAllBatch()
        self.game.stepBatch()
        for i in range(self.game.sizeOfBatch()):
            o = self.game.observation(i)
            a = self.game.action(i)
            r = self.game.reward(i)
            ch = Node(o,a,r, node)
            node.add_child( ch )

        # SIMULATION
        nStep = 3
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

    def batchOfObservationAction(self):
        obs = np.empty( (0,self.game.sizeOfObservationArray()), dtype=float )
        act = np.empty( (0,self.game.sizeOfActionArray()), dtype=float )
        node = self.root
        while not node != None:
            next_node = None
            if  len(node.children)>0:
                next_node = self.BestChild(node, 0)
                obs = np.append( obs, node.observation )
                act = np.append( act, next_node.action )
            node = next_node
        return obs,act

    def PlayTreePolicy(self):
        node = self.root
        if node==None:
            return
        self.game.resetAllBatch( node.observation )
        self.game.setPaused(True)
        prof = 0
        print("Start game")
        print("reward=" + str(self.game.reward(0)) + "   " + str(node))
        print("mcts node observation="+str(node.observation))
        print("game node observation="+str(self.game.observation(0)))
        while not self.game.isQuit():
            self.game.manageEvent()
            self.game.drawSceneMenuAndSwap()
            if not self.game.paused():
                if  len(node.children)>0:
                    next_node = self.BestChild(node, 0)
                    if self.game.noAction:
                        self.game.setZeroActionForAllBatch()
                    else:
                        self.game.setActionForAllBatch(next_node.action)
                    self.game.stepBatch()
                    node = next_node
                    prof += 1
                    print("game reward=" + str(self.game.reward(0)) + "   " + str(node))
                else:
                    print("profondeur de l'arbre="+str(prof))
                    print("=============================================")
                    prof=0
                    node = self.root
                    self.game.setPaused(True)
                    print("game...end=>reset: root="+str(node))
        print("PlayTreePolicy...done")


    # def PlayRandomPolicy(self, n):
    #     self.resetAllBatch( node.observation )
    #     while not self.game.isQuit():
    #         self.game.setRandomAction(0)
    #         self.game.setActionForAllBatch( self.game.action(0) )
    #         self.game.stepBatch()
    #         print( "game reward="+ str(self.game.reward(0)))
    #         self.game.manageEvent()
    #         self.game.drawSceneMenuAndSwap()
    #     print("PlayRandomPolicy...done")


if __name__ == "__main__":
    starship = starship.Starship()
    starship.init("starship", 1000, 500, 15, 15)

    mcts = MCTS(starship, 2000, 1)
    #mcts.PlayRandomPolicy(10)
    print( starship.observation(0) )
    print("--------------------------------")
    mcts.search(starship.observation(0))
    mcts.PlayTreePolicy()
