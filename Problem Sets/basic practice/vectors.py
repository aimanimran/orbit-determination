import math

def mag(vec):
    if (len(vec) == 1):
        return vec
    else:
        sum = 0
        for i in range(len(vec)):
            sum = sum + vec[i]**2
        return math.sqrt(sum)
print("mag:", mag([1, 1, 1, 1]))

def dot(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum = sum + vec1[i]*vec2[i]
    return sum

print("dot prod:", dot([1,0,1,0], [2,2,0,2]))

def crossprod(a, b):
    one = a[1]*b[2] - a[2]*b[1]
    two = a[2]*b[0] - a[0]*b[2]
    three = a[0]*b[1] - a[1]*b[0]
    return [one, two, three]

print("cross prod", crossprod([2,5,6], [3,7,8]))