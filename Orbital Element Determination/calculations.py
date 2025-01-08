import numpy as np
from oedlib.oedlib import *
k = 0.0172020989484


def SEL(taus,Sun2,rhohat2,Ds):
    roots = [0.,0.,0.] #for up to three real, positive roots
    rhos = []
    A1 = taus[1]/taus[2]
    B1 = (A1*((taus[2])**2 - (taus[1])**2))/6
    A3 = (-taus[0])/(taus[2])
    B3 = (A3*((taus[2])**2 - (taus[0])**2))/6
    A = (A1*Ds[1] - Ds[2] + A3*Ds[3])/(-Ds[0])
    B = (B1*Ds[1] + B3*Ds[3])/(-Ds[0])
    E = -2*(np.dot(rhohat2, Sun2))
    F = (np.linalg.norm(Sun2))**2
    a = -(A**2 + A*E + F)
    b = -(2*A*B + B*E)
    c = -B**2
    polyArray = [c, 0, 0, b, 0, 0, a, 0, 1]
    print('coefficients', a, b, c)
    roots = np.polynomial.polynomial.polyroots(polyArray)
    # print(f"roots: {roots}") - DEBUG
    real_positive_roots = []

    for value in roots:
        if np.imag(value) == 0 and np.real(value) > 0.05:
            real_positive_roots.append(np.real(value))
    
             
    for i in range(len(real_positive_roots)):
        rhos.append(A + (B/(real_positive_roots[i]**3)))
   
    print('real positive', real_positive_roots)
    print('rho', rhos)

    r_2 = []
    rho2mag = []

    for i in range(len(rhos)):
        if rhos[i] > 0.05:
            rho2mag.append(rhos[i])
            r_2.append(real_positive_roots[i])

    if len(r_2) == 1:
        return [r_2[0], rho2mag[0]] 
    else:
        indices = int(input("which rho value 0 indexed"))
        return [r_2[indices], rho2mag[indices]]
    

def Final(fileName):
#getting the observation data and time from reading the input file
    DateNtime1 = getDate_obs1(fileName)
    DateNtime2 = getDate_obs2(fileName)
    DateNtime3 = getDate_obs3(fileName)
#converting to julian date 
    J_1 = JulianDate(DateNtime1)
    J_2 = JulianDate(DateNtime2)
    J_3 = JulianDate(DateNtime3)

    # print(f"Julian date {J_1, J_2, J_3}") - DEBUG
#calculating the tau values to input into the lagrange function, to get r2 roots and rho2 initial value 
    tau1 = k*((J_1 - J_2))
    tau3 = k*((J_3 - J_2))
    tau0 = k*((J_3 - J_1))

    taus = np.array([tau1, tau3, tau0])
    # print(f"taus {taus}") - DEBUG
#read file and got the ra and dec for each observation to calculate the unit rho vector for each observation times
    RADEC1 = (RA_DEC1(fileName))
    RADEC2 = (RA_DEC2(fileName))
    RADEC3 = (RA_DEC3(fileName))
    # print(RADEC1, RADEC2, RADEC3) - DEBUG

    rho_hat_1 = [np.cos(np.radians((RADEC1[0])))*np.cos(np.radians((RADEC1[1]))), np.sin(np.radians((RADEC1[0])))*np.cos(np.radians((RADEC1[1]))), np.sin(np.radians(RADEC1[1]))]
    rho_hat_2 = [np.cos(np.radians((RADEC2[0])))*np.cos(np.radians((RADEC2[1]))), np.sin(np.radians((RADEC2[0])))*np.cos(np.radians((RADEC2[1]))), np.sin(np.radians(RADEC2[1]))]
    rho_hat_3 = [np.cos(np.radians((RADEC3[0])))*np.cos(np.radians((RADEC3[1]))), np.sin(np.radians((RADEC3[0])))*np.cos(np.radians((RADEC3[1]))), np.sin(np.radians(RADEC3[1]))]
    # print(np.linalg.norm(rho_hat_1), np.linalg.norm(rho_hat_2), np.linalg.norm(rho_hat_3)) - DEBUG

#sunvector read from file for further computation and calculating the D's and array to use in the lagrange function defined SEL
    SunVec1 = getSunVec1(fileName, 3)
    SunVec2 = getSunVec1(fileName, 7)
    SunVec3 = getSunVec1(fileName, 11)

    # print(SunVec1, SunVec2, SunVec3) - DEBUG
   
    D_o = np.dot(rho_hat_1, np.cross(rho_hat_2, rho_hat_3))
    D_21 = np.dot(np.cross(rho_hat_1, SunVec1), rho_hat_3)
    D_22 = np.dot(np.cross(rho_hat_1, SunVec2), rho_hat_3)
    D_23 = np.dot(np.cross(rho_hat_1, SunVec3), rho_hat_3)

    # print(f"Ds {D_o, D_21, D_22, D_23}") - DEBUG
    
    D_11 = np.dot(np.cross(SunVec1, rho_hat_2), rho_hat_3)
    D_12 = np.dot(np.cross(SunVec2, rho_hat_2), rho_hat_3)
    D_13 = np.dot(np.cross(SunVec3, rho_hat_2), rho_hat_3)

    # print(D_11, D_12, D_13) - DEBUG

    D_31 = np.dot(rho_hat_1, np.cross(rho_hat_2, SunVec1))
    D_32 = np.dot(rho_hat_1, np.cross(rho_hat_2, SunVec2))
    D_33 = np.dot(rho_hat_1, np.cross(rho_hat_2, SunVec3))

    # print(D_31, D_32, D_33) - DEBUG

    Ds = np.array([D_o, D_21, D_22, D_23])
   
    r2_rhos = (SEL(taus,SunVec2,rho_hat_2,Ds))

    # print(f" lagrange {r2_rhos}") - DEBUG
   
    r2mag = np.array(r2_rhos[0])
    rho2mag = np.array(r2_rhos[1])

    # print(f"rho2mag {rho2mag}") - DEBUG

#defining initial f and g variables to be further used to define small d's and C's, then using the d arrays to find the r2 dot as a linear combination of r1 and r3
    f1_g1 = np.array(getFG(taus, 0, r2mag))
    f3_g3 = np.array(getFG(taus, 1, r2mag))
    f1 = f1_g1[0]
    g1 = f1_g1[1]
    f3 = f3_g3[0]
    g3 = f3_g3[1]
    # print(f"FG {f1, g1, f3, g3}") - DEBUG

    Cs = [(g3/(f1*g3 - g1*f3)), -1, (-g1/(f1*g3 - g1*f3))]

    #print(f"Cs {Cs}") - DEBUG

# rho magnitudes used in calculating r1 and r3 vectors from the sun, rho and position triangle
    rho1mag = (Cs[0]*D_11 + Cs[1]*D_12 + Cs[2]*D_13)/(Cs[0]*D_o)
    rho3mag = (Cs[0]*D_31 + Cs[1]*D_32 + Cs[2]*D_33)/(Cs[2]*D_o)

    rhoMags = np.array([rho1mag, rho2mag, rho3mag])

    # print(f"rhomags {rhoMags}") - DEBUG
  
    rho1 = np.multiply(rho_hat_1, rho1mag)
    rho3 = np.multiply(rho_hat_3, rho3mag)

    #print(f"rho 1 rho 3{rho1, rho3}") - DEBUG

    r1 = rho1 - SunVec1
    r2 = np.multiply(rho_hat_2, rho2mag) - SunVec2
    r3 = rho3 - SunVec3

    #print(r1, r2, r3) - DEBUG    

    ds = [(-f3/(f1*g3 - f3*g1)), (f1/(f1*g3 - f3*g1))]

    #print(ds) - DEBUG

#initial r2 dot value for the fourth order f&g iteration 
    print('ds', ds[0], ds[1])
    print('r1', r1)
    print('r3', r3)
    r2dot = np.add(np.dot(ds[0], r1),np.dot(ds[1], r3))

    print('r2 dot', r2dot)
    print(rho_hat_1, rho_hat_2, rho_hat_3)

    iterations = 0 
    diff = 100
    print("************************************************")
    print('Main Iteration Loop')
    while diff >= 10E-9 :
        prev = rhoMags[1] 
        corrected_times = ligthtimeCorrection(J_1, J_2, J_3, rhoMags)
        print('corrected_times', corrected_times)
        a = get_semi_major(np.linalg.norm(r2), np.linalg.norm(r2dot))
        e = get_eccent(np.cross(r2, r2dot), a)
        FG = fgfunc(corrected_times[0], corrected_times[1], r2, r2dot, np.linalg.norm(r2), a, e)
        #FG = fg(corrected_times[0], corrected_times[1], r2, r2dot)
        # print('Fg', FG)
        ds = [(-FG[1]/(FG[0]*FG[3] - FG[1]*FG[2])), (FG[0]/(FG[0]*FG[3] - FG[1]*FG[2]))]
        print('ds', ds)
        Cs = [(FG[3]/(FG[0]*FG[3] - FG[2]*FG[1])), -1, (-FG[2]/(FG[0]*FG[3] - FG[2]*FG[1]))]
        print('Cs', Cs)
        # print('r2', r2)
        # print('a', a)
        # print('c', Cs)
        rho1mag = (Cs[0]*D_11 + Cs[1]*D_12 + Cs[2]*D_13)/(Cs[0]*D_o)
        print((Cs[0]*D_11 + Cs[1]*D_12 + Cs[2]*D_13))
        print((Cs[0]*D_o))
        rho2mag = (Cs[0]*D_21 + Cs[1]*D_22 + Cs[2]*D_23)/(Cs[1]*D_o)
        rho3mag = (Cs[0]*D_31 + Cs[1]*D_32 + Cs[2]*D_33)/(Cs[2]*D_o)
    
        diff = abs(prev - rho2mag)
        # print(diff)
        rhoMags = np.array([rho1mag, rho2mag, rho3mag])
        print('rhoMags', rhoMags)
        print(f"julian dates {J_1, J_2, J_3}")
           
        rho1 = np.multiply(rho_hat_1, rho1mag)
        rho2 = np.multiply(rho_hat_2, rho2mag)
        rho3 = np.multiply(rho_hat_3, rho3mag)
        # print(5)
        r1 = rho1 - SunVec1
        # print('rho_hat_2', rho_hat_2)
        # print('rho2mag', rho2mag)
        r2 = rho2 - SunVec2
        r3 = rho3 - SunVec3
        print('r values', r1, r2, r3)

        r2dot = np.add(np.multiply(ds[0], r1), np.multiply(ds[1], r3))
        print('r2dot', r2dot)

        iterations += 1

#final position and velocity vectors form the second observation in equitorial
    print("final in equitorial", r2, iterations, r2dot)
#converting to ecliptic coordinates (rotation matrix)
    position = ToEquitorial(r2)
    velocity = ToEquitorial(r2dot)
    print("final in ecliptic, r and r2", position, velocity)
    #print(f"position: {position}, velocity: {velocity}") - DEBUG

#ALL orbital elements from the OD codes 1 to 4 
    # print("************************************************")
    # print("Orbital Elements")
#updated time for the second observation from 00:00 to 7:00 and updated outputs 
    # time = np.array([2021.00, 7.00, 24.00, 7, 0, 0])

    corrected_jTimes = ligthtimeCorrection1(J_1, J_2, J_3, rhoMags)
    # J_o = JulianDate(time)

    
    # corrected_JTimes = ligthtimeCorrection1(J_1, J_o, J_3, rhoMags)
   
    OrbitalElements(position, velocity, corrected_jTimes, corrected_jTimes[1])


Final("viviInput.txt")
