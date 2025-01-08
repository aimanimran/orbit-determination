import matplotlib.pyplot as plt

# oz_bluescreen and meadow are a RGB images, so image and background are 3D arrays
# and have 3 values at every pixel location
# the slice is to remove an unnecessary alpha channel, if present
image = plt.imread("oz_bluescreen.png")[:, :, :3]
background = plt.imread("meadow.png")[:, :, :3]


# put wizard and ballooon in the meadow
for i in range(len(image)):
    for j in range(len(image[0])):
        if (image[i][j][2] > image[i][j][0] + image[i][j][1]):
            image[i][j][0] = background[i][j][0]
            image[i][j][1] = background[i][j][1]
            image[i][j][2] = background[i][j][2]

# save the modified image to a new file called oz_meadow.png
plt.imsave("oz_meadow.png", image)
