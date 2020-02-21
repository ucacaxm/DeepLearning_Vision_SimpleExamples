import cma
import numpy as np




def fitness_1(a):
    return a[0]*a[0] + a[1]*a[1]


def fitness(a):
    r = []
    for i in range(len(a)):
        r1 = fitness_1(a[i])
        r.append(r1)
    return r


x0 = np.array( [ 1,1 ] )
sigma0 = 2
es = cma.CMAEvolutionStrategy(x0, sigma0)

while not es.stop():
    solutions = es.ask()
    fit = fitness( solutions )
    #print(fit)
    es.tell(solutions, fit)
    es.disp()
r = es.best
print(str(r))
#r = es.result()
