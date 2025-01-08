import numpy as np
import math

def newton_raphson(e, M, convergence):
    count = 0
    E = 1
    diff = 10
    while (diff > convergence):
        newE = E - ((E - e * np.sin(E) - M)/(1 - e*np.cos(E)))
        diff = abs(E - newE)
        E = newE
        count = count + 1
    print("E = {}".format(E))
    print("Convergence parameter = {}".format(convergence))
    print("Number of iterations = {}".format(count))

def main_newt_raph(convergence):
    M = 0.42 # rad
    e = 0.8
    newton_raphson(e, M, convergence)

main_newt_raph(0.0001) # from graph, E = 1.1503    