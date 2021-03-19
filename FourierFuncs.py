# Define function for clever fourier combination of images,
# Inputs: HDU containing low-res data; HDU containing high-res data; array of low-res beam gridded to the pixel scale of high-res image; array of high-res beam gridded to pixel scale of high-res image(); (optional boolean/float for giving angular scale in degrees at which to apply a tapering transition; boolean of whether to employ subpixel low-pass filter to low-res image to remove pixel edge artefacts; boolean/string for saving combined image to file instead of returning; boolean for also returning additional data; boolean describing whether to use beam-mediated feather and only use window for cross-calibrating
# Outputs: The combined image
def FourierCombine(lores_hdu, hires_hdu, lores_beam_img, hires_beam_img,
                   taper_cutoffs_deg=False, apodise=False, to_file=False, return_all=False, beam_cross_corr=False):

    # If input images are being provided as paths to files, discern this and read them in
    lores_hdu_path = False
    if isinstance(lores_hdu, str):
        lores_hdu_path = lores_hdu
        lores_hdu = astropy.io.fits.PrimaryHDU(data=astropy.io.fits.getdata(lores_hdu),
                                               header=astropy.io.fits.getheader(lores_hdu))
    hires_hdu_path = False
    if isinstance(hires_hdu, str):
        hires_hdu_path = hires_hdu
        hires_hdu = astropy.io.fits.PrimaryHDU(data=astropy.io.fits.getdata(hires_hdu),
                                               header=astropy.io.fits.getheader(hires_hdu))
    if isinstance(lores_beam_img, str):
        lores_beam_img = astropy.io.fits.getdata(lores_beam_img)
    if isinstance(hires_beam_img, str):
        hires_beam_img = astropy.io.fits.getdata(hires_beam_img)

    # Make clean copies of input arrays, then put into float32, and delete HDUs, to save memory
    lores_img = lores_hdu.data.copy().astype(np.float32)
    lores_hdr = lores_hdu.header.copy()
    hires_img = hires_hdu.data.copy().astype(np.float32)
    hires_img_orig = hires_hdu.data.copy().astype(np.float32)
    hires_hdr = hires_hdu.header.copy()
    lores_beam_img = lores_beam_img.copy().astype(np.float32)
    hires_beam_img = hires_beam_img.copy().astype(np.float32)
    del(lores_hdu, hires_hdu)

    # Calculate low -resolution pixel size
    lores_wcs = astropy.wcs.WCS(lores_hdr)
    lores_pix_width_arcsec = 3600.0 * np.abs(np.max(lores_wcs.pixel_scale_matrix))

    # Calculate high-reoslution pixel size
    hires_wcs = astropy.wcs.WCS(hires_hdr)
    hires_pix_width_arcsec = 3600.0 * np.abs(np.max(hires_wcs.pixel_scale_matrix))
    hires_pix_width_deg = hires_pix_width_arcsec / 3600.0

    # Impute (temporarily) any NaNs surrounding the coverage region with the clipped average of the data (so that the fourier transformers play nice)
    hires_img[np.where(np.isnan(hires_img))] = SigmaClip(hires_img, median=True, sigma_thresh=1.0)[1]

    # Temporarily interpolate over any NaNs in low-resolution data, reproject to high-resolution pixel scale
    lores_img = ImputeImage(lores_img)
    lores_img = astropy.convolution.interpolate_replace_nans(lores_img, astropy.convolution.Gaussian2DKernel(3),
                                                             astropy.convolution.convolve_fft, allow_huge=True, boundary='wrap').astype(np.float32)
    lores_img = reproject.reproject_interp((lores_img, lores_hdr), hires_hdr, order='bicubic')[0].astype(np.float32) # Ie, following how SWarp supersamples images
    lores_edge = np.zeros(lores_img.shape)
    lores_edge[np.where(np.isnan(lores_img))] = 1.0
    lores_img[np.where(lores_edge == 1.0)] = np.nanmedian(lores_img)

    # If requested, low-pass filter low-resolution data to remove pixel-edge effects
    if apodise:
        lores_apodisation_kernel_sigma = 0.5 * 2.0**-0.5 * (lores_pix_width_arcsec / hires_pix_width_arcsec) # 0.5 * 2.0**-0.5
        lores_apodisation_kernel = astropy.convolution.Gaussian2DKernel(lores_apodisation_kernel_sigma).array
        lores_img = astropy.convolution.convolve_fft(lores_img, lores_apodisation_kernel,
                                                     boundary='reflect', allow_huge=True, preserve_nan=False) # As NaNs already removed

        # Incorporate apodisation filter into the low-resolution beam (as we have to account for the fact that its resolution is now ever-so-slightly lower)
        lores_beam_img = astropy.convolution.convolve_fft(lores_beam_img, lores_apodisation_kernel,
                                                     boundary='reflect', allow_huge=True, preserve_nan=False)
        lores_beam_img -= np.min(lores_beam_img)
        lores_beam_img /= np.sum(lores_beam_img)

    # Fourier transform all the things
    hires_beam_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_beam_img))).astype(np.complex64)
    lores_beam_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lores_beam_img))).astype(np.complex64)
    hires_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_img))).astype(np.complex64)
    lores_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(lores_img))).astype(np.complex64)
    del(hires_img, lores_img, hires_beam_img)

    # Add miniscule offset to any zero-value elements to stop inf and nan values from appearing later.
    lores_beam_fourier.real[np.where(lores_beam_fourier.real == 0)] = 1E-50

    # Divide the low-resolution data by the low-resolution beam (ie, deconvolve it), then multiply by the high-resoluiton beam, to normalise amplitudes
    fourier_norm = hires_beam_fourier / lores_beam_fourier
    fourier_norm[np.where(np.isinf(fourier_norm))] = 1E-50
    lores_fourier *= fourier_norm

    # If requested, start by cross-calibrating the hires and lores data within the tapering angular window
    if taper_cutoffs_deg != False:
        hires_fourier_corr_factor = FourierCalibrate(lores_fourier, hires_fourier, taper_cutoffs_deg, hires_pix_width_deg)
        hires_fourier_corr = hires_fourier * hires_fourier_corr_factor[0]

        # Perform tapering between specificed angular scales to weight data in Fourier space, following a Hann filter profile
        taper_filter = FourierTaper(taper_cutoffs_deg, hires_wcs)
        hires_weight = 1.0 - taper_filter
        hires_fourier_weighted = hires_fourier_corr.copy()
        hires_fourier_weighted *= hires_weight
        lores_weight = taper_filter
        lores_fourier_weighted = lores_fourier.copy()
        lores_fourier_weighted *= lores_weight

    # Otherwise, in standard operation, use low-resolution beam to weight the tapering from low-resolution to high-resolution data
    elif taper_cutoffs_deg == False:
        hires_fourier_corr_factor = [1.0, 0.0]
        hires_weight = 1.0 - lores_beam_fourier
        hires_fourier_weighted = hires_fourier * hires_weight
        lores_weight = 1.0 * lores_beam_fourier
        lores_fourier_weighted = lores_fourier * lores_weight

    # Remove edge effects from high-resolution map, where possible
    hires_weighted_img = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(hires_fourier_weighted)))).astype(np.float32)
    hires_weighted_img[np.where(np.isnan(hires_img_orig))] = SigmaClip(hires_weighted_img, median=True, sigma_thresh=1.0)[1]
    hires_fourier_weighted = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(hires_weighted_img))).astype(np.complex64)

    # Combine the images, then convert back out of Fourier space
    comb_fourier = lores_fourier_weighted + hires_fourier_weighted
    comb_fourier_shift = np.fft.ifftshift(comb_fourier)
    comb_img = np.fft.fftshift(np.real(np.fft.ifft2(comb_fourier_shift))).astype(np.float32)
    del(hires_weight,lores_weight,hires_fourier_weighted,
        lores_fourier_weighted,hires_weighted_img,comb_fourier,comb_fourier_shift)

    # Estimate the size of the low-resolution beam
    lores_beam_demislice = lores_beam_img[int(round(0.5*lores_beam_img.shape[1])):,int(round(0.5*lores_beam_img.shape[1]))]
    lores_beam_width_pix = float(np.argmin(np.abs(lores_beam_demislice - (0.5 * lores_beam_demislice.max()))))

    # Purely for prettiness sake, identify any messy transition region between high- and low-resolution
    hires_mask = hires_img_orig.copy()
    hires_mask = (hires_mask * 0.0) + 1.0
    hires_mask[np.where(np.isnan(hires_mask))] = 0.0
    hires_mask_dilated_out = scipy.ndimage.binary_dilation(hires_mask, iterations=int(1.5*round(lores_beam_width_pix))).astype(int)
    hires_mask = -1.0 * (hires_mask - 1.0)
    hires_mask_dilated_in = scipy.ndimage.binary_dilation(hires_mask, iterations=int(1.5*round(lores_beam_width_pix))).astype(int)
    hires_mask_border = (hires_mask_dilated_out + hires_mask_dilated_in) - 1
    comb_img[np.where(hires_mask_border)] = np.nan
    comb_img[np.where(comb_img == 0)] = np.nan

    # Create an even larger border region for creating the interpolation to make the transition smooth
    hires_mask_border_expanded = scipy.ndimage.binary_dilation(hires_mask_border, iterations=int(2.0*round(lores_beam_width_pix))).astype(int)
    comb_fill = comb_img.copy()
    comb_fill[np.where(hires_mask_border_expanded)] = np.nan
    comb_fill = astropy.convolution.interpolate_replace_nans(comb_fill,
                                                             astropy.convolution.Gaussian2DKernel(round(2.0*lores_beam_width_pix)),
                                                             astropy.convolution.convolve_fft,
                                                             allow_huge=True,
                                                             boundary='wrap')
    comb_img[np.where(hires_mask_border==1)] = comb_fill[np.where(hires_mask_border==1)]
    #comb_img = astropy.convolution.interpolate_replace_nans(comb_img, astropy.convolution.Gaussian2DKernel(round(2.0*lores_beam_width_pix)), astropy.convolution.convolve_fft, allow_huge=True, boundary='wrap')
    comb_img[np.where(lores_edge == 1.0)] = np.nan

    # Create mask describing tegions of the map where the only data is the good high-resolution combined data
    comb_mask = hires_mask_border.copy()
    comb_mask[np.where(np.isnan(hires_img_orig))] = 0

    # Remove any temp files
    if lores_hdu_path != False:
        os.remove(lores_hdu_path)
    if hires_hdu_path != False:
        os.remove(hires_hdu_path)

    # Return combined image (or save to file if that was requested, with added possibility of returing calibration correction)
    if not to_file:
        if return_all:
            return (comb_img, hires_fourier_corr_factor[0], hires_fourier_corr_factor[1], comb_mask)
        else:
            return comb_img
    else:
        astropy.io.fits.writeto(to_file, data=comb_img, header=hires_hdr, overwrite=True)
        if return_all:
            return (to_file, hires_fourier_corr_factor[0], hires_fourier_corr_factor[1], comb_mask)
        else:
            return to_file

          

# Function to create a 2D tapering fourier-space filter, transitioning between two defined angular scales, according to a Hann (ie, cosine bell) filter
# Inputs: Iterable giving angular scale of high- and low resolution cutoffs (in deg); WCS of the data to be combined
# Outputs: Array containing requested filter, with low-frequency passpand and high-frequency stopband
def FourierTaper(taper_cutoffs_deg, in_wcs):

    # Use provided WCS to construct array to hold the output filter
    if in_wcs.array_shape[0] != in_wcs.array_shape[1]:
        raise Exception('Input data to be combined are not square; this will not end well')
    out_filter = np.zeros([in_wcs.array_shape[1], in_wcs.array_shape[0]])
    out_i_centre = (0.5*out_filter.shape[0])-0.5
    out_j_centre = (0.5*out_filter.shape[1])-0.5

    # Convert cutoff scales to units of fourier-pixels
    pix_width_deg = np.abs( np.max( in_wcs.pixel_scale_matrix ) )
    cutoff_min_deg = min(taper_cutoffs_deg)
    cutoff_max_deg = max(taper_cutoffs_deg)
    cutoff_max_frac = (cutoff_max_deg / pix_width_deg) / out_filter.shape[0]
    cutoff_min_frac = (cutoff_min_deg / pix_width_deg) / out_filter.shape[0]
    cutoff_max_pix = 1.0 * (1.0 / cutoff_max_frac)
    cutoff_min_pix = 1.0 * (1.0 / cutoff_min_frac)

    # Use meshgrids to find distance of each pixel from centre of filter array
    i_linespace = np.linspace(0, out_filter.shape[0]-1, out_filter.shape[0])
    j_linespace = np.linspace(0, out_filter.shape[1]-1, out_filter.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')
    i_grid -= out_i_centre
    j_grid -= out_j_centre
    rad_grid = np.sqrt(i_grid**2.0 + j_grid**2.0)

    # Rejigger the distance grid to give distance past inner edge of taper region
    rad_grid -= cutoff_max_pix

    # Set central region (ie, low-resolution regime) to be entirely passband
    out_filter[np.where(rad_grid <= 0)] = 1.0

    # Construct well-sampled Hann filter to compute taper for transition region
    hann_filter = np.hanning(2000)[1000:]
    hann_pix = np.linspace(0, cutoff_min_pix-cutoff_max_pix, num=1000)
    hann_interp = scipy.interpolate.interp1d(hann_pix, hann_filter, bounds_error=False, fill_value=np.nan)

    # Identify pixels where Hann filter is to be applied, and compute taper
    hann_where = np.where((rad_grid > 0) & (rad_grid <= cutoff_min_pix))
    out_filter[hann_where] = hann_interp(rad_grid[hann_where])
    out_filter[np.where(np.isnan(out_filter))] = 0.0

    # Return final filter
    return out_filter



# Function to cross-calibrate two data sets' power over a range of angular scales, as a precursor to fourier combination
# Input: Fourier transform of low-resolution data; fourier transform of high-resolution data; iterable giving angular scale of high- and low resolution cutoffs (in deg)
# Output: Correction factor to be applied to high-resolution data; uncertainty on the correction factor
def FourierCalibrate(lores_fourier, hires_fourier, taper_cutoffs_deg, hires_pix_width_deg):

    # Find centre of fourier arrays (ie, the zeroth order fourier frequency)
    freq_i_centre = (0.5*hires_fourier.shape[0])-0.5
    freq_j_centre = (0.5*hires_fourier.shape[1])-0.5

    # Calculate grid of radial fourier frequency distance from zeroth order
    i_linespace = np.linspace(0, hires_fourier.shape[0]-1, hires_fourier.shape[0])
    j_linespace = np.linspace(0, hires_fourier.shape[1]-1, hires_fourier.shape[1])
    i_grid, j_grid = np.meshgrid(i_linespace, j_linespace, indexing='ij')
    i_grid -= freq_i_centre
    j_grid -= freq_j_centre
    rad_grid = np.sqrt(i_grid**2.0 + j_grid**2.0)

    # Convert cutoff scales from degrees to fourier frequencies
    cutoff_min_deg = min(taper_cutoffs_deg)
    cutoff_max_deg = max(taper_cutoffs_deg)
    cutoff_min_frac = (cutoff_min_deg / hires_pix_width_deg) / hires_fourier.shape[0]
    cutoff_max_frac = (cutoff_max_deg / hires_pix_width_deg) / hires_fourier.shape[0]
    cutoff_min_pix = 1.0 / cutoff_min_frac
    cutoff_max_pix = 1.0 / cutoff_max_frac

    """# Slice out subsets of radial distance grid corresponding to cutoffs, and their overlap
    rad_grid_cutoff_min = rad_grid[np.where(rad_grid<cutoff_min_pix)]
    rad_grid_cutoff_max = rad_grid[np.where(rad_grid>cutoff_max_pix)]
    rad_grid_overlap = rad_grid[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]
    power_lores_cutoff_min = power_lores[np.where(rad_grid<cutoff_min_pix)]"""

    # Calculate power of each scale (square-rooted, so that it's power, not the power spectrum)
    power_lores = np.sqrt((lores_fourier.real)**2.0)
    power_hires = np.sqrt((hires_fourier.real)**2.0)

    # Slice out power at regions corresponding to cutoffs, and their overlap
    power_lores_overlap = power_lores[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]
    power_hires_overlap = power_hires[np.where((rad_grid>cutoff_max_pix) & (rad_grid<cutoff_min_pix))]

    # Caltulate correction factor (in log space, as distribution in power space is logarithmic)
    power_dex_overlap = np.log10(power_hires_overlap) - np.log10(power_lores_overlap)
    power_hires_corr_factor = 1.0 / 10.0**np.median(power_dex_overlap)#SigmaClip(power_dex_overlap, median=True, sigma_thresh=1.0)[1]

    # Calculate uncertainty on correction factor by bootstrapping
    power_hires_corr_bs = []
    for b in range(100):
        #power_hires_corr_bs.append(SigmaClip(np.random.choice(power_dex_overlap, size=len(power_dex_overlap)), median=True, sigma_thresh=1.0)[1])
        power_hires_corr_bs.append(np.median(np.random.choice(power_dex_overlap, size=len(power_dex_overlap))))
    power_hires_corr_factor_unc = np.std(1.0 / (10.0**np.array(power_hires_corr_bs)))

    # Return results
    return (power_hires_corr_factor, power_hires_corr_factor_unc)
