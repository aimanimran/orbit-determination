import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def findCentroid(fits_file, target_x, target_y, radius=3, sky_radius=5):
    data = fits.getdata(fits_file)
    y = target_y
    x = target_x

    subimage = data[(y - radius):(y + radius + 1), (x - radius):(x + radius + 1)].astype(float)
    bkgimage = data[(y - radius - sky_radius):(y + radius + sky_radius + 1), (x - radius - sky_radius):(x + radius + sky_radius + 1)].copy()

    bkgimage[sky_radius:sky_radius + 2*radius + 1, sky_radius:sky_radius + 2*radius + 1] = 0
    bkgavg = np.sum(bkgimage)/240

    # subtact brightness from star values
    for i in range(len(subimage)):
        for j in range(len(subimage[0])):
            subimage[i][j] = subimage[i][j] - bkgavg

    # average of star values
    x_num = 0
    y_num = 0
    denom = 0
    for i in range(len(subimage)):
        for j in range(len(subimage[0])):
            y_num = y_num + i*(subimage[i][j])
            denom = denom + subimage[i][j]

    for i in range(len(subimage)):
        for j in range(len(subimage[0])):
            x_num = x_num + j*(subimage[i][j])

    x_num = x_num / denom
    y_num = y_num / denom

    return(x_num-radius+target_x, y_num-radius+target_y)

centroid_x, centroid_y= findCentroid("sampleimage.fits", 351,
154, 3, 5)

# centroid_x, centroid_y, uncert_x, uncert_y = findCentroid("sampleimage.fits", 459,
# 397, 2)

if abs(centroid_x - 350.7806) < 0.1 and abs(centroid_y - 153.5709) < 0.1:
    print("centroid calculation CORRECT")
else:
    print(
        "centroid calculation INCORRECT, expected (350.7806, 153.5709), got ({}, {})".format(
        centroid_x, centroid_y))
    
# works fine!!