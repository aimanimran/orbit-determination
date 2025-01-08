# 1 - Ephermieris Generator
import ephem

obs = ephem.Observer()
obs.lon = "-81:24:52.9" # longit
obs.lat = "36:15:09.7" # lat
obs.elev = 922 # in m
obs.date = "2020/07/05 07:08:00" # UTC

line = "2003GE42,e,27.8543085,165.6469288,101.353156,2.63408,,0.375418379,8.4377868,07/05.29722/2020,2000,,"

asteroid = ephem.readdb(line)
asteroid.compute(obs)
print(asteroid.a_ra, asteroid.a_dec)

# 2 - Approximating Pi

import numpy as np
from math import pi

numIn = 0
total = 0
allpi = [] # array for mean/std calculation
for i in range(5000000):

    # get random coordinates
    x = np.random.random()*2
    y = np.random.random()*2

    # circle equation with center point being (1,1)
    if ((x-1)**2 + (y-1)**2 <= 1):
        numIn = numIn + 1
    total = total + 1
    
    # percentage inside circle * total area = circle area = pi*1**2
    piVal = (numIn/total)*4
    allpi.append(piVal)

print("Final Approximation:", piVal)
print("Percent Error:", ((piVal - pi)/pi)*100)
print("Mean:", np.mean(allpi))
print("Standard Deviation", np.std(allpi))