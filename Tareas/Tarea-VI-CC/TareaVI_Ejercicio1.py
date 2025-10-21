import numpy as np

def simular_bernoulli(p, size):
    np.random.seed(1234)
    U = np.random.uniform(0, 1, size)
    return np.where(U <= 1-p, 0, 1)

p = 1/3
n = np.array([5, 50])

for i in n:
    simulacion = simular_bernoulli(p, i)
    print("Para n=",i, "la media de éxitos es ", np.mean(simulacion), "número de éxitos", i*np.mean(simulacion),)
    