import numpy as np
from numpy import loadtxt
import math
from math import pi

#Angular Momentum
def ang_momentum(rvec, rdotvec):
    return np.cross(rvec, rdotvec)

# Semimajor Axis
def get_semi_major(rmag, rdotmag):
    # if just file is inputted
        # r = math.sqrt(file[0]**2 + file[1]**2 + file[2]**2)
        # v = math.sqrt(file[3]**2 + file[4]**2 + file[5]**2)

    a = 1/((2/rmag) - rdotmag**2)
    return a
    #print("Semimajor axis: {}".format(a))

# Eccentricity
def get_eccent(h_vec, a):
    h = math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)
    e = math.sqrt(1 - (h**2/a))
    #print("Eccentricity: {}".format(e))
    return e

# Inclination
def get_inclination(h_vec):
    i = math.atan(math.sqrt((h_vec[0])**2 + (h_vec[1])**2)/h_vec[2])
    return i * (180/pi)
    #print("Inclination: {}".format(i_degrees))

# Longitude of the Ascending Node
def get_longit_asc(h_vec, i):
    h = math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)

    omega_cos = (-1*(h_vec[1]))/(h * np.sin(i)) 
    omega_sin = (h_vec[0])/(h * np.sin(i))

    if (omega_sin > 0):
        omega = math.acos(omega_cos)
    else:
        omega = 2*pi - math.acos(omega_cos)

    omega_degrees = omega*(180/pi)
    return omega_degrees
    #print(omega_degrees)

# True Anom
def get_anom(rvec, rdotvec, a, h_vec, e):
    h = math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)
    r = math.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    
    anom_sin = (a*(1-e**2)/(e*h)) * ((np.dot(rvec, rdotvec))/r)
    anom_cos = (1/e)*(((a*(1-e**2))/r)-1)

    if (anom_sin > 0):
        anom = math.acos(anom_cos)
    else:
        anom = 2*pi - math.acos(anom_cos)
    return anom*(180/pi)

# Argument of Perihelion
def get_arg_perih(rvec, anom, omega, i):
    r = math.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    z = rvec[2]

    sin_u = (z / r*math.sin(i*(pi/180)))
    cos_u = ((rvec[0]*math.cos(omega*(pi/180))) + (rvec[1]*math.sin(omega*(pi/180))))/r

    if (sin_u > 0):
        U = math.acos(cos_u) * (180/pi)
    else:
        U = 2*pi - math.acos(cos_u) * (180/pi)

    min_omega = U - anom
    min_omega = min_omega % 360
    return min_omega

# Convert dates
def getDate_obs1(fileName):
    data = np.array(open(fileName).readlines())
    date = np.array(data[0].split())
    years = date[0].astype(float)
    months = date[1].astype(float)
    days = date[2].astype(float)
    hours = date[3].astype(float)
    minutes = date[4].astype(float)
    seconds = date[5].astype(float)
    return np.array([years, months, days, hours, minutes, seconds])  

def getDate_obs2(fileName):
    data = np.array(open(fileName).readlines())
    date = np.array(data[4].split())
    years = date[0].astype(float)
    months = date[1].astype(float)
    days = date[2].astype(float)
    hours = date[3].astype(float)
    minutes = date[4].astype(float)
    seconds = date[5].astype(float)
    return np.array([years, months, days, hours, minutes, seconds]) 

def getDate_obs3(fileName):
    data = np.array(open(fileName).readlines())
    date = np.array(data[8].split())
    years = date[0]
    months = date[1]
    days = date[2]
    hours = date[3]
    minutes = date[4]
    seconds = date[5]
    return np.array([years, months, days, hours, minutes, seconds]) 

# Getting data from input file
def RA_DEC1(fileName):
    data = np.array(open(fileName).readlines())
    c = np.array((data[2]).split())
    e = ((c[0]).astype(float))
    f = c[1].astype(float)
    return [e, f]

def RA_DEC2(fileName):
    data = np.array(open(fileName).readlines())
    c = np.array((data[6]).split())
    e = c[0].astype(float)
    f = c[1].astype(float)
    return [e, f]

def RA_DEC3(fileName):
    data = np.array(open(fileName).readlines())
    c = np.array((data[10]).split())
    e = c[0].astype(float)
    f = c[1].astype(float)
    return [e, f]

def getSunVec1(fileName, i):
    data = np.array(open(fileName).readlines())
    vec = np.array(data[i].split())
    vec = vec.astype(float)
    return vec

def getSunVec2(fileName, i):
    data = np.array(open(fileName).readlines())
    vec = np.array(data[i].split())
    vec = vec.astype(float)
    return vec

def getSunVec3(fileName, i):
    data = np.array(open(fileName).readlines())
    np.array(data[i].split())
    position = position.astype(float)
    return position

# Adjust time for speed of light
def ligthtimet(J_1, J_2, J_3, rhomags):
    cAU = 173.144643267
    #k = 0.0172020989484
    t_1 = J_1 - (rhomags[0]/cAU)
    t_2 = J_2 - (rhomags[1]/cAU)
    t_3 = J_3 - (rhomags[2]/cAU)
    return [t_1, t_2, t_3]

def ligthtimetau(J_1, J_2, J_3, rhomags):
    cAU = 173.144643267
    k = 0.0172020989484
    t_1 = J_1 - (rhomags[0]/cAU)
    t_2 = J_2 - (rhomags[1]/cAU)
    t_3 = J_3 - (rhomags[2]/cAU)
    tau1 = (t_1-t_2)*k
    tau3 = (t_3-t_2)*k
    tau0 = (t_3-t_1)*k 
    return [tau1, tau3, tau0]

# Convert system
def ToEquitorial(vec):
    Epsilon = math.radians(23.44)
    rot_matrix_4 = np.array([[1, 0, 0], [0, np.cos(Epsilon), (-1)*(np.sin(Epsilon))], [0, np.sin(Epsilon), np.cos(Epsilon)]])
    rot_matrix = np.linalg.inv(rot_matrix_4)
    eq_vec = np.dot(rot_matrix, vec)
    return eq_vec

# Get all Orbital Elements
def calc_elements(file_name):
    file = loadtxt(file_name, comments="#")
    # structure is x,y,z,vx,vy,vz,a,ec,in,ra,ta,w (with enter b/w)

    # convert AU/standard days --> AU/Gaussian days
    for i in range(3,6):
        file[i] = file[i] * 365.2568983/(2*pi)
    h_vec = ang_momentum(file)
    
    # get elements
    axis = get_semi_major(file) # A
    e = get_eccent(h_vec, axis) # EC
    i = get_inclination(h_vec) #IN
    longit_asc = get_longit_asc(h_vec, (i/(180/pi))) #OM
    anom = get_anom(file, axis, h_vec, e) #TA
    arg_perih = get_arg_perih(file, anom, longit_asc, i) #W

    print("1 - Semi Major Axis")
    print("Expected Value: {}".format(file[6]))
    print("Calculated Value: {}".format(axis))
    print("Percent Error: {}".format((file[6] - axis)/file[6]))
    print("")

    print("2 - Eccentricity")
    print("Expected Value: {}".format(file[7]))
    print("Calculated Value: {}".format(e))
    print("Percent Error: {}".format((file[7] - e)/file[7]))
    print("")

    print("3 - Inclination")
    print("Expected Value: {}".format(file[8]))
    print("Calculated Value: {}".format(i))
    print("Percent Error: {}".format((file[8] - i)/file[8]))
    print("")

    print("4 - Longitude of Ascending Node")
    print("Expected Value: {}".format(file[9]))
    print("Calculated Value: {}".format(longit_asc))
    print("Percent Error: {}".format((file[9] - longit_asc)/file[9]))
    print("")

    print("5 - True Anomaly")
    print("Expected Value: {}".format(file[10]))
    print("Calculated Value: {}".format(anom))
    print("Percent Error: {}".format((file[10] - anom)/file[10]))
    print("")

    print("6 - Argument of Perihelion")
    print("Expected Value: {}".format(file[11]))
    print("Calculated Value: {}".format(arg_perih))
    print("Percent Error: {}".format((file[11] - arg_perih)/file[11]))
    print("")

# Orbital Elements + Date
def julian_date(year, month, day, hour, min, sec):
    UT = (hour + 4) + (min/60) + (sec/3600)
    j0 = 367*year - int(7*(year + int((month+9)/12))/4) + int(275*month/9) + day + 1721013.5
    jd = j0 + (UT/24)
    return jd

# Get Newton-Raphson "E" Value
def newton_raphson(e, M, convergence):
    count = 0
    E = 1
    diff = 10
    while (diff > convergence):
        newE = E - ((E - e * np.sin(E) - M)/(1 - e*np.cos(E)))
        diff = abs(E - newE)
        E = newE
        count = count + 1
    return E
    print("E = {}".format(E))
    print("Convergence parameter = {}".format(convergence))
    print("Number of iterations = {}".format(count))

# Scalar Equation of Lagrange
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

def getFG(taus, i, r_2):
    r2mag = np.linalg.norm(r_2)
    u = 1/((r2mag)**3)
    f = 1 - (1/2)*u*(taus[i]**2) 
    g = taus[i] - (1/6)*u*(taus[i]**3) 
    return f, g 

def allFG(t1f, t3f, r2, r2dot, r2mag, a, e):
    n = math.sqrt(1/a**3)

    # NUM 1: Finding f1 and g1
    sign1 = (np.dot(r2, r2dot)/(n*a**2))*math.cos(n*t1f - (np.dot(r2, r2dot)/(n*a**2))) + (1-(r2mag/a))*math.sin(n*t1f - (np.dot(r2, r2dot))/(n*a**2))
    E1 = n*t1f + math.copysign(0.85, sign1) * e - (np.dot(r2, r2dot))/(n*a**2)
    # print("E1_0", E1) - DEBUG
    
    diff = 10
    while (diff > 10**-12):
        f = E1 - (1 - (r2mag/a))*math.sin(E1) + (np.dot(r2, r2dot)/(n*a**2)) * (1-math.cos(E1)) - n*t1f
        fprime = 1 - (1-(r2mag/a))*math.cos(E1) + (np.dot(r2, r2dot)/(n*a**2)) * math.sin(E1)
        # For tolerance - difference between old and new E
        newE = E1 - f/fprime
        diff = abs(newE - E1)
        E1 = newE
    # print('E1', E1) - DEBUG

    # Final using determined E1
    f1 = 1 - (a/r2mag)*(1 - math.cos(E1))
    g1 = t1f + (1/n)*(math.sin(E1) - E1)

    # NUM 2: Finding f3 and g3
    sign3 = (np.dot(r2, r2dot)/(n*a**2))*math.cos(n*t3f - (np.dot(r2, r2dot)/(n*a**2))) + (1-(r2mag/a))*math.sin(n*t3f - (np.dot(r2, r2dot))/(n*a**2))
    E3 = n*t3f + math.copysign(0.85, sign3) * e - (np.dot(r2, r2dot))/(n*a**2)
        # print("E3_0", E3) - DEBUG
    diff = 10
    while (diff > 10**-12):
        f = E3 - (1 - (r2mag/a))*math.sin(E3) + (np.dot(r2, r2dot)/(n*a**2)) * (1-math.cos(E3)) - n*t3f
        fprime = 1 - (1-(r2mag/a))*math.cos(E3) + (np.dot(r2, r2dot)/(n*a**2)) * math.sin(E3)
            # For tolerance - difference between old and new E
        newE = E3 - f/fprime
        diff = abs(newE - E3)
        E3 = newE
        # print('E3', E3) - DEBUG

    # Final using determined E3
    f3 = 1 - (a/r2mag)*(1 - math.cos(E3))
    g3 = t3f + (1/n)*(math.sin(E3) - E3)

    return f1, f3, g1, g3


# Iterations to get more accurate values
def Final(fileName, k):
#getting the observation data and time from reading the input file
    DateNtime1 = getDate_obs1(fileName)
    DateNtime2 = getDate_obs2(fileName)
    DateNtime3 = getDate_obs3(fileName)

#converting to julian date 
    J_1 = julian_date(DateNtime1)
    J_2 = julian_date(DateNtime2)
    J_3 = julian_date(DateNtime3)
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
        corrected_times = ligthtimetau(J_1, J_2, J_3, rhoMags)
        print('corrected_times', corrected_times)
        a = get_semi_major(np.linalg.norm(r2), np.linalg.norm(r2dot))
        e = get_eccent(np.cross(r2, r2dot), a)
        FG = allFG(corrected_times[0], corrected_times[1], r2, r2dot, np.linalg.norm(r2), a, e)
        # print('Fg', FG) - DEBUG

        ds = [(-FG[1]/(FG[0]*FG[3] - FG[1]*FG[2])), (FG[0]/(FG[0]*FG[3] - FG[1]*FG[2]))]
        print('ds', ds)
        Cs = [(FG[3]/(FG[0]*FG[3] - FG[2]*FG[1])), -1, (-FG[2]/(FG[0]*FG[3] - FG[2]*FG[1]))]
        print('Cs', Cs)
        rho1mag = (Cs[0]*D_11 + Cs[1]*D_12 + Cs[2]*D_13)/(Cs[0]*D_o)
        print((Cs[0]*D_11 + Cs[1]*D_12 + Cs[2]*D_13))
        print((Cs[0]*D_o))

        rho2mag = (Cs[0]*D_21 + Cs[1]*D_22 + Cs[2]*D_23)/(Cs[1]*D_o)
        rho3mag = (Cs[0]*D_31 + Cs[1]*D_32 + Cs[2]*D_33)/(Cs[2]*D_o)
        diff = abs(prev - rho2mag)
        # print(diff) - DEBUG

        rhoMags = np.array([rho1mag, rho2mag, rho3mag])
        print('rhoMags', rhoMags)
        print(f"julian dates {J_1, J_2, J_3}")
           
        rho1 = np.multiply(rho_hat_1, rho1mag)
        rho2 = np.multiply(rho_hat_2, rho2mag)
        rho3 = np.multiply(rho_hat_3, rho3mag)

        r1 = rho1 - SunVec1
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

    corrected_jTimes = ligthtimet(J_1, J_2, J_3, rhoMags)
    calc_elements(position, velocity, corrected_jTimes, corrected_jTimes[1])
