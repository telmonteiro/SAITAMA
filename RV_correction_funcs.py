'''
This file contains functions to be used to correct and flag the spectra by the RV in the ACTINometer pipeline
'''
import numpy as np
from PyAstronomy import pyasl # type: ignore
from general_funcs import (calc_fits_wv_1d, read_fits)
from get_spec_funcs import (_get_simbad_data)

def get_rv_ccf(star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument, quick_RV):
    '''
    Uses crosscorrRV function from PyAstronomy to get the CCF of the star comparing with a spectrum of the Sun.
    Returns the BJD in days, the RV, the max value of the CCF, the list of possible RV and CC, as well as the flux and raw wavelength.
    To maximize the search for RV and avoid errors, the script searches the RV in SIMBAD and makes a reasonable interval. 
    '''
    
    if instrument == "HARPS" or instrument == "UVES" or instrument == "ESPRESSO":
        bjd = stellar_header["HIERARCH ESO DRS BJD"] #may change with instrument
    else: bjd = None

    #get wavelength and flux of both sun (template) and star
    w = stellar_wv; f = stellar_flux
    tw = calc_fits_wv_1d(template_hdr); tf = template_spec

    #to make the two spectra compatible, cut the stellar one
    w_ind_common = np.where((w < np.max(tw)) & (w > np.min(tw)))
    w_cut = w[w_ind_common]
    f_cut = f[w_ind_common]

    #only uses two 250 Angstrom intervals, that change according to the instrument
    if quick_RV == True: 
        if instrument == "UVES": #doesn't include CaII H&K lines
            w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5500 > w) & (w > 5250)) #Halpha region and leftmost side of spectrum
        else: #HARPS or ESPRESSO
            w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5250 > w) & (w > 4900)) #Halpha region and CaII H&K region
        w_cut = w_cut[w_ind_tiny]
        f_cut = f_cut[w_ind_tiny]   

    try:
        rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
        if instrument == "HARPS":
            rvmin = rv_simbad - 5; rvmax = rv_simbad + 5
        elif instrument == "UVES":
            rvmin = rv_simbad - 80; rvmax = rv_simbad + 80
        elif instrument == "ESPRESSO":
            rvmin = rv_simbad - 80; rvmax = rv_simbad + 80
    except:
        rvmin = -150; rvmax = 150

    #get the cross-correlation
    skipedge_values = [0, 100, 500, 1000, 5000, 20000]

    for skipedge in skipedge_values:
        try:
            rv, cc = pyasl.crosscorrRV(w=w_cut, f=f_cut, tw=tw, tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
            break  # Break out of the loop if successful
        except Exception as e:
            #print(f"Error with skipedge={skipedge}: {e}")
            continue
            # Continue to the next iteration

    #index of the maximum cross-correlation function
    maxind = np.argmax(cc)
    radial_velocity = rv[maxind]

    if units == "m/s":
        radial_velocity *= 1000
        rv *= 1000
    
    return bjd, radial_velocity, cc[maxind], np.around(rv,0), cc, w, f

def correct_spec_rv(wv, rv, units):
    '''
    Correct wavelength of spectrum by the RV of the star with a Doppler shift.
    '''
    c = 299792.458 #km/s
    if units == "m/s":
        c *= 1000
    #delta_wv = wv * rv / c #shift
    #wv_corr = wv + delta_wv #o problema era o sinal... é wv - delta_wv
    wv_corr = wv / (1+rv/c)
    delta_wv = wv - wv_corr
    return wv_corr, np.mean(delta_wv)

################################

def line_ratio_indice(data, line="CaI"):
    '''
    Computes a ratio between the continuum and line flux of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,"NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 2
    else: window = 0.6

    ratio_arr = np.zeros(len(data)); center_flux_line_arr = np.zeros(len(data)); flux_continuum_arr = np.zeros(len(data)) 

    for i in range(len(data)):
        wv = data[i][0]; flux = data[i][1]
        #print(wv[np.where((6500 < wv) & (wv < 6580))])
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
        wv = wv[wv_array]
        #print(wv_array)
        flux = flux[wv_array]
        #flux_normalized = flux/np.linalg.norm(flux)
        flux_normalized = (flux-np.min(flux))/(np.max(flux)-np.min(flux))
        if len(flux_normalized) < 50:
            lim = 5
        else: lim = 20
        flux_left = np.median(flux_normalized[:lim])
        flux_right = np.median(flux_normalized[:-lim])
        flux_continuum = np.median([flux_left,flux_right]) #median
        #print("Flux continuum: ",flux_continuum)
        
        wv_center_line = np.where((line_wv-window/30 < wv) & (wv < line_wv+window/30))
        flux_line = flux_normalized[wv_center_line]
        center_flux_line = np.median(flux_line)
        #print("Flux center line: ",center_flux_line)

        ratio = center_flux_line/flux_continuum 

        ratio_arr[i] = ratio
        center_flux_line_arr[i] = center_flux_line
        flux_continuum_arr[i] = flux_continuum
         
    return ratio_arr, center_flux_line_arr, flux_continuum_arr

def flag_ratio_RV_corr(files,instr):
    '''
    For each spectrum, run an interval of offsets and if the minimum ratio is not in [-0.02,0.02], raises a flag = 1.
    Flag = 0 is for good spectra. Then the flag ratio #0/#total is computed
    '''
    offset_list = np.linspace(-1,1,1001)
    flag_list = np.zeros((len(files)))
    #print(files)
    for i,file in enumerate(files):
        wv, flux, flux_err, hdr = read_fits(file,instrument=instr,mode="rv_corrected")
        #if i == 0: wv += 0.2 #just to fake a bad spectrum
        ratio_list = np.zeros_like(offset_list)
        for j,offset in enumerate(offset_list):
            ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv+offset,flux)], line="CaI")
            ratio_list[j]=ratio_arr

        min_ratio_ind = np.argmin(ratio_list)
        offset_min = offset_list[min_ratio_ind]
        #print(offset_min)
        if offset_min < -0.05 or offset_min > 0.05:
            flag_list[i] = 1
        else: flag_list[i] = 0

    good_spec_ind = np.where((flag_list == 0))
    N_good_spec = len(flag_list[good_spec_ind])
    flag_ratio = N_good_spec / len(flag_list)

    return flag_ratio, flag_list