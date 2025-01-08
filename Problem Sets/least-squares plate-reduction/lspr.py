def flex_lspr(ref_file, degrees, astr_x, astr_y):
    import numpy as np
    from numpy import loadtxt
    from decimal import Decimal
    import math
    file = loadtxt(ref_file, dtype=str)
    
    # create separate lists with correct units
    x_list = []
    y_list = []
    ra_list = []
    dec_list = []

    for i in range(len(file)):
            x_list.append(float(file[i][0]))
    x_list = np.array(x_list)

    for i in range(len(file)):
        y_list.append(float(file[i][1]))
    y_list = np.array(y_list)

    if degrees == False:
        ra_og = []
        dec_og = []

        for i in range(len(file)):
            ra_og.append(file[i][2])
        ra_og = np.array(ra_og)

        for i in range(len(file)):
            dec_og.append(file[i][3])
        dec_og = np.array(dec_og)
        
        #convert ra and dec -> number
        for i in range(len(ra_og)):
            ra_list.append((float(ra_og[i][0:2]) + (float((ra_og[i][3:5]))/60) + (float((ra_og[i][6:11]))/3600))*15)

        for i in range(len(dec_og)):
            dec_list.append(float(dec_og[i][1:3]) + (float((dec_og[i][4:6]))/60) + (float((dec_og[i][7:11]))/3600))
    else:
        for i in range(len(file)):
            ra_list.append(float(file[i][2]))
        ra_list = np.array(ra_list)

        for i in range(len(file)):
            dec_list.append(float(file[i][3]))
        dec_list = np.array(dec_list)

    # create matrixes and get 6 plate constants
    x_sum = np.sum(x_list)
    y_sum = np.sum(y_list)
    ra_sum = np.sum(ra_list)
    dec_sum = np.sum(dec_list)

    x_sq = 0
    xy_mult = 0
    xra_mult = 0
    xdec_mult = 0
    for i in range(len(x_list)):
        x_sq = x_sq + (x_list[i]**2)
        xy_mult = xy_mult + (x_list[i] * y_list[i])
        xra_mult = xra_mult + (x_list[i]*ra_list[i])
        xdec_mult = xdec_mult + (x_list[i]*dec_list[i])

    y_sq = 0
    yra_mult = 0
    ydec_mult = 0
    for i in range(len(y_list)):
        y_sq = y_sq + (y_list[i]**2)
        yra_mult = yra_mult + (y_list[i]*ra_list[i])
        ydec_mult = ydec_mult + (y_list[i]*dec_list[i])

    if degrees == False:
        xy_eq = [
            [12, x_sum, y_sum],
            [x_sum, x_sq, xy_mult],
            [y_sum, xy_mult, y_sq]
        ]
    else:
        xy_eq = [
            [10, x_sum, y_sum],
            [x_sum, x_sq, xy_mult],
            [y_sum, xy_mult, y_sq]
        ]

    ra_eq = [ra_sum, xra_mult, yra_mult]
    dec_eq = [dec_sum, xdec_mult, ydec_mult]

    xy_invert = np.linalg.inv(xy_eq)
    ra_mult = np.dot(xy_invert, ra_eq)
    dec_mult = np.dot(xy_invert, dec_eq)

    print("6 Plate Constants: ")
    print(ra_mult)
    print(dec_mult)

    # get RA and DEC coords

    ra_final = (ra_mult[0] + ra_mult[1]*astr_x + ra_mult[2]*astr_y)/15
    dec_final = dec_mult[0] + dec_mult[1]*astr_x + dec_mult[2]*astr_y

    print("Right Ascension: {}".format(ra_final))
    print("Declination: {}".format(dec_final))

    # get uncertainty

    ra_uncert = 0
    for i in range(len(ra_list)):
        ra_uncert = ra_uncert + (ra_list[i] - ra_mult[0] - ra_mult[1]*x_list[i] - ra_mult[2]*y_list[i])**2

    dec_uncert = 0
    for i in range(len(dec_list)):
        dec_uncert = dec_uncert + (dec_list[i] - dec_mult[0] - dec_mult[1]*x_list[i] - dec_mult[2]*y_list[i])**2

    # adjust for degrees vs. radians
    if degrees == False:
        ra_uncert = math.sqrt(ra_uncert / (12 - 3)) * 3600
        dec_uncert = math.sqrt(dec_uncert / (12 - 3)) * 3600
    else:
        ra_uncert = math.sqrt(ra_uncert / (10 - 3))
        dec_uncert = math.sqrt(dec_uncert / (10 - 3))

    print("RA Uncertainty: {}".format(ra_uncert))
    print("Dec Uncertainty: {}".format(dec_uncert))

flex_lspr("input.txt", False, 484.35, 382.62)
