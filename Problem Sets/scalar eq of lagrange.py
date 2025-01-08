import numpy as np
import math

def SEL(taus, Sun2, rhohat2, Ds):
    roots = []
    rhos = []

    sun = math.sqrt(Sun2[0]**2 + Sun2[1]**2 + Sun2[2]**2)

    t1 = taus[0]
    t3 = taus[1]
    t0 = taus[2]

    A1 = t3/t0
    B1 = (A1/6) * (t0**2 - t3**2)
    A3 = -t1/t0
    B3 = (A3/6) * (t0**2 - t1**2)

    A = (A1*Ds[1] - Ds[2] + A3*Ds[3])/(-Ds[0])
    B = (B1*Ds[1] + B3*Ds[3])/(-Ds[0])

    E = -2*(np.dot(rhohat2, Sun2))
    F = sun**2

    a = -(A**2 + A*E + F)
    b = -(2*A*B + B*E)
    c = -(B**2)

    allRoots = np.polynomial.polynomial.polyroots([c, 0, 0, b, 0, 0, a, 0, 1])
    for i in range(len(allRoots)):
        if (allRoots[i] > 0 and np.imag(allRoots[i]) == 0):
            roots.append(allRoots[i].real)

    for i in range(len(roots)):
        rhos.append(A + (B/roots[i]**3))

    return (roots, rhos)

# Test Case 1
taus = [-0.15481889055, 0.15481889055, 0.3096377811]
Sun2 = [-0.2398478458274071, 0.9065739917845802, 0.3929623749770952]
rhohat2 = [-0.8518563498182248, -0.2484702599212149, 0.4610892421311239]
Ds = [-0.0010461861084885213, -0.17297581974209159, -0.17201260125558127, -0.16712421570714076]
print("Test Case 1", SEL(taus, Sun2, rhohat2, Ds))

# Test Case 2
taus_2 = [-0.1720209895, 0.1720209895, 0.344041979]
Sun2_2 = [-0.2683394448727136, 0.8997620807182745, 0.3900022331276332]
rhohat2_2 = [0.052719013914983195, -0.9121555187306237, 0.40643943610469035]
Ds_2 = [0.0011797570297704812, 0.052586743761143424, 0.05848153743706686, 0.06274019190783499]
print("Test Case 2", SEL(taus_2, Sun2_2, rhohat2_2, Ds_2))