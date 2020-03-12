import math
import random

import numpy as np
import matplotlib.pyplot as plt



class ProbaGame:
    def __init__(self):
        print("ProbaGame ...")
        plt.figure(1)
        self.account = 1000
        self.ite = 0
        self.n_gagne = 0
        self.n_perdu = 0
        plt.plot( self.ite, self.account, 'ro', color='red')


    def one_step_auto(self, mise, seuil):
        val = np.random.randint(0,101)
        if mise<0:
            mise = self.account* (-float(mise))/100.0
        print("mise="+str(mise) + "  seuil="+str(seuil)+"  valeur hasard="+str(val))
        if (val<=seuil):
            gain = float(mise) * (100.0-float(seuil))/100.0
            self.account += gain
            self.n_gagne += 1
            print("gagné")
        else:
            self.account -= mise
            self.n_perdu += 1
            print("perdu")
        print("accound="+str(self.account) + "  n_gagne="+str(self.n_gagne)+ " n_perdu="+str(self.n_perdu))
        self.ite += 1
        plt.plot( self.ite, self.account, 'o', color='red')
        #plt.show()


    def one_step_user(self):
        print("accound="+str(self.account))
        mise = input("Entrez votre mise : ")
        seuil = input("Pariez que le nombre sera inférieur à  : ")
        self.one_step_auto(mise,seuil)
        #plt.show()



if __name__ == "__main__":
    pg = ProbaGame()
    for i in range(1000):
        pg.one_step_auto( 1.0, 95)
    plt.show()