import math
import random

import numpy as np
import pygame
from pygame.locals import *


class Vec2(object):
    def __init__(self, x=0, y=0):
        self._x = float(x)
        self._y = float(y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_x):
        self._x = float(new_x)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_y):
        self._y = float(new_y)

    def __add__(self, other):
        types = (int, float)
        if isinstance(self, types):
            return Vec2(self + other.x, self + other.y)
        elif isinstance(other, types):
            return Vec2(self.x + other, self.y + other)
        else:
            return Vec2(self.x + other.x, self.y + other.y)

    def __div__(self, other):
        types = (int, float)
        if isinstance(self, types):
            self = Vec2(self, self)
        elif isinstance(other, types):
            other = Vec2(other, other)
        x = self.x / other.x
        y = self.y / other.y
        return Vec2(x, y)

    def __mul__(self, other):
        types = (int, float)
        if isinstance(self, types):
            return Vec2(self * other.x, self * other.y)
        elif isinstance(other, types):
            return Vec2(self.x * other, self.y * other)
        else:
            return Vec2(self.x * other.x, self.y * other.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __radd__(self, other):
        return Vec2(self.x + other, self.y + other)

    def __rdiv__(self, other):
        return Vec2(other/self.x, other/self.y)

    def __rmul__(self, other):
        return Vec2(other * self.x, other * self.y)

    def __rsub__(self, other):
        return Vec2(other - self.x, other - self.y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Vec2: ({0}, {1})".format(self.x, self.y)

    def __sub__(self, other):
        types = (int, float)
        if isinstance(self, types):
            return Vec2(self - other.x, self - other.y)
        elif isinstance(other, types):
            return Vec2(self.x - other, self.y - other)
        else:
            return Vec2(self.x - other.x, self.y - other.y)

    def ceil(self):
        return Vec2(math.ceil(self.x), math.ceil(self.y))

    def floor(self):
        return Vec2(math.floor(self.x), math.floor(self.y))

    def get_data(self):
        return (self.x, self.y)    

    def inverse(self):
        return Vec2(1.0/self.x, 1.0/self.y)

    def length(self):
        return math.sqrt(self.square_length())

    def normalize(self):
        length = self.length()
        if length == 0.0:
            return Vec2(0, 0)
        return Vec2(self.x/length, self.y/length)

    def round(self):
        return Vec2(round(self.x), round(self.y))

    def square_length(self):
        return (self.x * self.x) + (self.y * self.y)

    """
    def transform(self, matrix):#mat2, mat2d, mat3, mat4
        pass

    @classmethod
    def cross(cls, a, b):
        z = (a.x * b.y) - (a.y * b.x)
        return Vec3(0, 0, z)
    """

    @classmethod
    def distance(cls, a, b):
        c = b - a
        return c.length()

    @classmethod
    def dot(self, a, b):
        return (a.x * b.x) + (a.y * b.y)

    @classmethod
    def equals(cls, a, b, tolerance=0.0):
        diff = a - b
        dx = math.fabs(diff.x)
        dy = math.fabs(diff.y)
        if dx <= tolerance * max(1, math.fabs(a.x), math.fabs(b.x)) and \
           dy <= tolerance * max(1, math.fabs(a.y), math.fabs(b.y)):
            return True
        return False

    @classmethod
    def max(cls, a, b):
        x = max(a.x, b.x)
        y = max(a.y, b.y)
        return Vec2(x, y)

    @classmethod
    def min(cls, a, b):
        x = min(a.x, b.x)
        y = min(a.y, b.y)
        return Vec2(x, y)

    @classmethod
    def mix(cls, a, b, t):
        return a * t + b * (1-t)

    @classmethod
    def random(cls):
        x = random.random()
        y = random.random()
        return Vec2(x, y)

    @classmethod
    def square_distance(cls, a, b):
        c = b - a
        return c.square_length()





class Particle:
    def __init__(self, x ,y):
        self.m = 1.0
        self.p = Vec2(x,y)
        self.v = Vec2(0,0)
        self.f = Vec2(0,0)
        self.lastf = Vec2(0,0)

    def update(self, dt):
        if (self.m>0):
            self.v = self.v + (dt/self.m)*self.f
            self.p = self.p + dt*self.v
            self.lastf = self.f
            self.f = Vec2(0,0)

    def addForce(self, f):
        self.f += f






class Starship:
    def __init__(self): # Notre méthode constructeur
        print("Starship Game ...")
    

    def init(self,name,w,h,px,py,f=0):
        self.m_isLearning = False
        self.m_paused = False
        self.m_quit = False
        self.m_print = False
        self.target = Vec2(w/2, h/2)
        self.starship = np.empty( self.sizeOfBatch(), dtype=Particle )
        self.reward = np.empty( self.sizeOfBatch(), dtype=float )
        self.done = np.empty( self.sizeOfBatch(), dtype=bool )
        self.obs = np.empty( (self.sizeOfBatch(),self.sizeOfObservationArray()), dtype=float )
        self.w = w
        self.h = h
        for i in range(self.sizeOfBatch()):
            self.initOneAgent(i)
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 10)
        print("Starship::init...ok")
        print("obs=",self.obs)
        print("reward=",self.reward)
        print("done=",self.done)

    def initOneAgent(self,i):
        self.starship[i] = Particle(random.random()*(self.w-1), random.random()*(self.h-1))
        self.computeRewardDoneObs(i)

    def sizeOfActionArray(self):
        return 2

    def sizeOfObservationArray(self):
        return 4

    def sizeOfBatch(self):
        return 12

    def paused(self):
        return self.m_paused;

    def isLearning(self):
        return self.m_isLearning;

    def setLearning(self, l):
        self.m_isLearning = l

    def resetOne(self, i):
        self.initOneAgent(i)
        self.computeRewardDoneObs(i)
        return self.obs[i]

    def reset(self):
        for i in range(self.sizeOfBatch()):
            self.resetOne(i)
        return self.obs
    
    def getStarship(self, i):
        return self.starship[i]

    def drawSceneMenuAndSwap(self):
        if not self.m_quit:
            self.screen.fill( (15,15,75) )
            pygame.draw.circle( self.screen, (255,0,0), (int(self.target.x), int(self.target.y)), 5, 0 )
            for i in range(self.starship.size):
                color = (255,255,255)
                colorv = (255,255,0)
                colorf = (255,0,0)
                if i==0:
                    color = (0,0,255)
                p = self.starship[i].p
                v = self.starship[i].v
                f = self.starship[i].lastf
                pygame.draw.circle( self.screen, color, (int(p.x), int(p.y)), 5, 0 )
                pygame.draw.line( self.screen, colorv, (int(p.x), int(p.y)), (int(p.x+3*v.x), int(p.y+3*v.y)) )
                pygame.draw.line( self.screen, colorf, (int(p.x), int(p.y)), (int(p.x+3*f.x), int(p.y+3*f.y)) )
                textsurface = self.myfont.render( str(int(self.reward[i])), False, (255, 0, 50))
                self.screen.blit(textsurface,(int(p.x)+10, int(p.y)+10))
            pygame.display.flip()

    def observe(self, obs):
        obs = self.obs
        return self.obs

    def computeRewardDoneObs(self, i):
        p = self.starship[i].p
        o = self.target-p
        v = self.starship[i].v
        self.obs[i] = ( o.x, o.y, v.x, v.y)
        self.reward[i] = 1000.0/(1.0+o.length())
        if (p.x<0 or p.y<0 or p.x>self.w or p.y>self.h):
            self.done[i] = True
            self.initOneAgent(i)
        else:
            self.done[i] = False

    def stepBatch(self, act, obs):
        #obs = np.empty( (self.sizeOfBatch,self.sizeOfObservationArray()), dtype=float )
        #print(act)
        for i in range(self.sizeOfBatch()):
            self.starship[i].addForce( Vec2(act[i][0],act[i][1]) )
            self.starship[i].update(0.1)
            self.computeRewardDoneObs(i)
        obs = self.obs
        if self.m_print:
            print("py! observation shape=",self.obs.shape, " action shape=",act.shape,
                  "    observation=", self.obs, " action=", act,
                  "   reward=", reward, "  done=",done)
        #reward, done = self.viewer.step(self.action, self.observation)
        return self.reward, self.done

    def randomMove(self):
        act = np.empty( (self.starship.size,2), dtype=float )
        for i in range(self.starship.size):
            act[i][0] = (random.random()*2.0 - 1.0)
            act[i][1] = (random.random()*2.0 - 1.0)
        _,_ = self.stepBatch( act, self.obs )

    def close(self):
        pass

    def manageEvent(self):
        for event in pygame.event.get():	#Attente des événements
            if event.type == QUIT:
                self.m_quit = True
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.m_quit = True 
                elif event.key == K_l:
                    self.m_isLearning = True
                elif event.key == K_m:
                    self.m_isLearning = False

            # elif key.is_pressed('q'):
            #     self.m_quit = True    
            # elif key.is_pressed('l'):
            #     self.m_isLearning = True
            #     print("is learning=", self.m_isLearning)
            # elif key.is_pressed('m'):
            #     self.m_isLearning = False
            #     print("is learning=", self.m_isLearning)
            # elif key.is_pressed('z'):
            #     self.m_paused = False
            #     print("paused=", self.m_paused)
            # elif key.is_pressed('e'):
            #     self.m_paused = True
            #     print("paused=", self.m_paused)
            # elif key.is_pressed('p'):
            #     self.m_print = True
            #     print("print=", self.m_print)
            # elif key.is_pressed('o'):
            #     self.m_print = False
            #     print("paused=", self.m_print)
            # elif key.is_pressed('h'):
            #     print("l/m=learning, z/e=paused, p/o=print")
        return self.m_quit


    def run(self):
        while not self.m_quit:
            self.randomMove()
            self.manageEvent()
            self.drawSceneMenuAndSwap()
            #print("one step")

if __name__ == "__main__":
    print("Starship game starting...")
    ship = Starship()
    ship.init("Starship", 800, 600, 50, 50)
    ship.run()
