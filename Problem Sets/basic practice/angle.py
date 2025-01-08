# PURPOSE OF THIS CODE
# PROJECT
# DATE
# NAME

import math

def convertAngle(degrees, minutes, seconds, ifRad, ifNorm):
    # handle a negative angle
    if (math.copysign(1, degrees) < 0):
        minutes = -minutes
        seconds = -seconds

    # perform angle conversion
    if ifRad:
        angle = degrees*math.pi/180 + (minutes/60)*math.pi/180 + (seconds/3600)*math.pi/180
    else:
        angle = degrees + (minutes/60) + (seconds/3600)

    if ifNorm and ifRad:
        angle = angle % (2*math.pi)
    elif ifNorm:
        angle = angle % 360

    # return result
    return angle



# test cases for part a
# these are the test cases you will demonstrate when getting this homework checked off
# test cases for part c (uncomment these, comment out previous tests)
print(convertAngle(90, 6, 36, False, False)) # should print 90.11
print(convertAngle(90, 6, 36, True, False)) # should print 1.57271618897
print(convertAngle(90, 6, 36, False, True)) # should print 90.11
print(convertAngle(90, 6, 36, True, True)) # should print 1.57271618897
print(convertAngle(-90, 6, 36, False, False)) # should print -90.11
print(convertAngle(-90, 6, 36, True, False)) # should print -1.57271618897
print(convertAngle(-90, 6, 36, False, True)) # should print 269.89
print(convertAngle(-90, 6, 36, True, True)) # should print 4.71046911821
print(convertAngle(540, 0, 0, False, True)) # should print 180.0
print(convertAngle(-0.0, 30, 45, False, False)) # should print -0.5125
