import math
import numpy as np
import matplotlib.pyplot as plt

# for altitude

lat = (35 + (54/60) + (50.58/3600))*(math.pi/180)
dec = (10 + (13/60) + (49/3600))*(math.pi/180)

y = []
x = range(-90, 91)

for i in np.linspace(0, math.pi, 181):
    y.append(np.arcsin(np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(i)))

for j in range(len(y)):
    y[j] = y[j]*(180/math.pi)

plt.plot(x,y)
plt.xlabel("Hour Angle")
plt.ylabel("Altitude (degrees)")