import matplotlib.pyplot as plt

# beach_portrait_gray.png is an RGB image, so image is a 3D array with 3 values at each pixel location
# the slice is to remove an unnecessary alpha channel, if present
image = plt.imread("beach_portrait.png")[:, :, :3]


def blur(img, radius):
    img2 = img.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                avg = image[i-radius:i+radius, j-radius:j+radius, k].mean()
                img2[i][j][k] = avg
    return img2


plt.imsave("beach_portrait_blur.png", blur(image, 3))