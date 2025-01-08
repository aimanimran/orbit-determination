def ap_phot(image, x, y, ap_rad, sky_inrad, sky_outrad, pad=10, find_centroid=True,
 dark_current=10., read_noise=11.):
    """
    Measure instrumental magnitude of a star in a fits image.
    image = fits image filename
    x = Approx. x position of object [pix]
    y = Approx. y position of object [pix]
    ap_rad = Aperture radius in pixels
    sky_inrad = Sky annulus inner radius in pixels
    sky_outrad = Sky annulus outer radius in pixels 
    pad = number of pixels to pad around this sky annulus for centroiding purposes. This should
    be larger than the possible difference between x,y and the actual centroid.
    """

    # Write here your code
    data = fits.getdata(image)
    x, y, bkgavg, bkgimage = findCentroid(image, x, y, ap_rad, (sky_outrad-sky_inrad))
    
    for row in range(len(bkgimage)):
        for col in range(len(bkgimage[0])):
            if ((row-y)**2 + (col)**2 <= ap_rad**2):
                sum = sum + image[row][col]

    signal = np.sum(data[y-ap_rad:y+ap_rad, x-ap_rad:x+ap_rad])
    SNR = np.sum(data[y-ap_rad:y+ap_rad, x-ap_rad:x+ap_rad])/bkgavg
        
    # Do not modify these print statements
    print("Centroid at: ({}, {})".format(round(x,2),round(y,2)))
    print("Signal:", int(signal), "+/-", int(signal/SNR), "ADU")
    print("SNR:", round(SNR,1))
    print("m_inst:", round(m_inst,2), "+/-", round(sig_m_inst,2), "mag")

    return

ap_phot("aptest.fit",490,293,5,8,13)