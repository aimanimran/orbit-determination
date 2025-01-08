import numpy as np
import math
from math import pi

# Level 3 and 4 Function
def fgfunc(t1f, t3f, r2, r2dot, r2mag):

    a = 1.056800055578855 #2.248304831625529
    e = 0.3442331103932664 #0.5772560992303707
    n = math.sqrt(1/a**3)

    # Finding f1 and g1
    sign1 = (np.dot(r2, r2dot)/(n*a**2))*math.cos(n*t1f - (np.dot(r2, r2dot)/(n*a**2))) + (1-(r2mag/a))*math.sin(n*t1f - (np.dot(r2, r2dot))/(n*a**2))
    E1 = n*t1f + math.copysign(0.85, sign1) * e - (np.dot(r2, r2dot))/(n*a**2)
    # Debug
    # print("E1_0", E1)
    diff = 10
    while (diff > 10**-12):
        f = E1 - (1 - (r2mag/a))*math.sin(E1) + (np.dot(r2, r2dot)/(n*a**2)) * (1-math.cos(E1)) - n*t1f
        fprime = 1 - (1-(r2mag/a))*math.cos(E1) + (np.dot(r2, r2dot)/(n*a**2)) * math.sin(E1)
        # For tolerance - difference between old and new E
        newE = E1 - f/fprime
        diff = abs(newE - E1)
        E1 = newE
    # Debug
    # print('E1', E1)

    # Final using determined E1
    f1 = 1 - (a/r2mag)*(1 - math.cos(E1))
    g1 = t1f + (1/n)*(math.sin(E1) - E1)

    # Finding f3 and g3
    sign3 = (np.dot(r2, r2dot)/(n*a**2))*math.cos(n*t3f - (np.dot(r2, r2dot)/(n*a**2))) + (1-(r2mag/a))*math.sin(n*t3f - (np.dot(r2, r2dot))/(n*a**2))
    E3 = n*t3f + math.copysign(0.85, sign3) * e - (np.dot(r2, r2dot))/(n*a**2)
    # Debug
    # print("E3_0", E3)
    diff = 10
    while (diff > 10**-12):
        f = E3 - (1 - (r2mag/a))*math.sin(E3) + (np.dot(r2, r2dot)/(n*a**2)) * (1-math.cos(E3)) - n*t3f
        fprime = 1 - (1-(r2mag/a))*math.cos(E3) + (np.dot(r2, r2dot)/(n*a**2)) * math.sin(E3)
        # For tolerance - difference between old and new E
        newE = E3 - f/fprime
        diff = abs(newE - E3)
        E3 = newE
    # Debug
    # print('E3', E3)

    # Final using determined E3
    f3 = 1 - (a/r2mag)*(1 - math.cos(E3))
    g3 = t3f + (1/n)*(math.sin(E3) - E3)

    return f1, f3, g1, g3

# All Levels Function: includes flag specifying 3rd order, 4th order, or real function
def fg(t1, t3, r2, r2dot, flag):
    r2mag = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)
    u = 1/(r2mag**3)
    z = np.dot(r2, r2dot)/r2mag**2
    q = np.dot(r2dot, r2dot)/r2mag**2 - u
    
    if (flag == "3rd order"):
        f1 = 1 - (1/2)*(u*t1**2) + (1/2)*(u*z*t1**3)
        g1 = t1 - (1/6)*u*(t1**3)

        f3 = 1 - (1/2)*(u*t3**2) + (1/2)*(u*z*t3**3)
        g3 = t3 - (1/6)*u*(t3**3)
    elif (flag == "4th order"):
        f1 = 1 - (1/2)*(u*t1**2) + (1/2)*(u*z*t1**3) + (1/24)*(3*u*q - 15*u*z**2 + u**2)*t1**4
        g1 = t1 - (1/6)*u*(t1**3) + (1/4)*u*z*(t1**4)

        f3 = 1 - (1/2)*(u*t3**2) + (1/2)*(u*z*t3**3) + (1/24)*(3*u*q - 15*u*z**2 + u**2)*t3**4
        g3 = t3 - (1/6)*u*(t3**3) + (1/4)*u*z*(t3**4)
    elif (flag == "function"):
        f1, f3, g1, g3 = fgfunc(t1, t3, r2, r2dot, r2mag)
        
    print("f1", f1)
    print("f3", f3)
    print("g1", g1)
    print("g3", g3)

# Test case: 3rd-order series
print("Testing 3rd order:")
fg(-0.32618617484601165, 0.0508408854033231, [0.26799552002875776, -1.3726277901924608, -0.5026729612047128], [0.8456809141954584, -0.3838382184712308, 0.14215854191172816], "3rd order")

# Test case: 4th-order series
print("Testing 4th order:")
fg(-0.3261857571141891, 0.05084081855693949, [0.26662393644794813, -1.381475976476564, -0.5048589337503169], [0.8442117090940343, -0.39728396707075087, 0.14202728258915864], "4th order")

# Test case: function w tolerance 1.e-12
print("Testing f&g function:")
fg(-0.32618569435308475, 0.050840808143482484, [0.26640998194891174, -1.382856212643199, -0.505199925482389], [0.8439832722802604, -0.39937767878456487, 0.14200790188593015], "function")