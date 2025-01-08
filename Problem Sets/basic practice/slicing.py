import numpy as np

fruits = np.array([["Apple","Banana","Blueberry","Cherry"],
["Coconut","Grapefruit","Kumquat","Mango"],
["Nectarine","Orange","Tangerine","Pomegranate"],
["Lemon","Raspberry","Strawberry","Tomato"]])

fruits[3][3]
fruits[-1][-1]

fruits[1:3][:, 1:3]

fruits[[0,2]]

np.flip(fruits[1:3][:, 1:3], [0,1])

fruits = np.array([["Apple","Banana","Blueberry","Cherry"],
["Coconut","Grapefruit","Kumquat","Mango"],
["Nectarine","Orange","Tangerine","Pomegranate"],
["Lemon","Raspberry","Strawberry","Tomato"]])
fruits2 = np.copy(fruits)
fruits[:,0] = fruits[:,3]
fruits[:,3] = fruits2[:,0]
fruits

fruits[0:4] = "SLICED!"
fruits
