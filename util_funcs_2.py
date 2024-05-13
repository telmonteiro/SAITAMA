'''
This file contains functions to be used by the pipeline function in the ACTINometer pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math
from astropy.io import fits
from astropy.table import Table
from PyAstronomy import pyasl # type: ignore
from PyAstronomy.pyTiming import pyPeriod # type: ignore
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from util_funcs_1 import _get_simbad_data

def read_fits(file_name,instrument,mode):
    '''
    Read fits file and get header and data. Varies if instrument is HARPS, ESPRESSO, UVES or FEROS. missing espresso
    '''
    hdul = fits.open(file_name)
    if instrument == "HARPS":
        if mode == "raw":
            if "s1d_A" in file_name:
                flux = hdul[0].data
                header = hdul[0].header
                wv = calc_fits_wv_1d(header)
                flux_err = np.zeros_like(flux)
            elif "ADP" in file_name:
                wv = hdul[1].data[0][0]
                flux = hdul[1].data[0][1]
                bjd = hdul[0].header["HIERARCH ESO DRS BJD"]
                header = hdul[0].header
                header["HIERARCH ESO DRS BJD"] = bjd
                if math.isnan(hdul[1].data["ERR"][0][0]):
                    flux_err = np.zeros_like(flux)
                else: 
                    flux_err = hdul[1].data["ERR"][0]
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header

    elif instrument == "UVES":
        if mode == "raw":
            header = hdul[0].header
            wv = hdul[1].data["WAVE"][0]
            try:
                flux = hdul[1].data["FLUX"][0]
                flux_err = hdul[1].data["ERR"][0]
            except:
                flux = hdul[1].data["FLUX_REDUCED"][0]
                flux_err = hdul[1].data["ERR_REDUCED"][0]
            bjd = hdul[0].header["MJD-OBS"]+2400000.5
            header["HIERARCH ESO DRS BJD"] = bjd
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header
    
    elif instrument == "ESPRESSO":
        if mode == "raw":
            if "s1d_A" in file_name:
                flux = hdul[0].data
                header = hdul[0].header
                wv = calc_fits_wv_1d(header)
                flux_err = np.zeros_like(flux)
            elif "ADP" in file_name:
                #print(hdul[0].data)
                header = hdul[0].header
                try:
                    wv = hdul[1].data["WAVE_AIR"][0]
                except:
                    wv = hdul[1].data["WAVE"][0]
                try:
                    flux = hdul[1].data["FLUX_EL"][0]
                    flux_err = hdul[1].data["ERR_EL"][0]
                    
                    valid_indices = ~np.isnan(flux)  #perigo de haver valores NaN neste tipo de flux
                    wv = wv[valid_indices]
                    flux = flux[valid_indices]
                    flux_err = flux_err[valid_indices]
                except:
                    flux = hdul[1].data["FLUX"][0]
                    flux_err = hdul[1].data["ERR"][0]
                bjd = hdul[0].header["MJD-OBS"]+2400000.5
                header["HIERARCH ESO DRS BJD"] = bjd
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header

    else:
        flux = hdul[0].data
        header = hdul[0].header
        wv = calc_fits_wv_1d(header)
        flux_err = np.zeros_like(flux)

    hdul.close()

    return wv, flux, flux_err, header

########################

def calc_fits_wv_1d(hdr, key_a='CRVAL1', key_b='CDELT1', key_c='NAXIS1'):
    '''
    Compute wavelength axis from keywords on spectrum header.
    '''
    try:
        a = hdr[key_a]; b = hdr[key_b]
    except KeyError:
        a = hdr["WAVELMIN"]*10; b = hdr["WAVELMAX"]*10
    try: 
        c = hdr[key_c]
    except KeyError:
        c = hdr["NELEM"]

    return a + b * np.arange(c)

########################

def sigma_clip(df, cols, sigma):
    '''
    Rough sigma clipping of a data frame.
    '''
    for col in cols:
        if math.isnan(list(df[col])[0]) == False:
            mean= df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - sigma * std) & (df[col] <= mean + sigma * std)]
    return df

########################

def plot_RV_indices(star,df,indices,save, path_save):
    """
    Plot RV and indices given as a function of time
    """
    plt.figure(figsize=(6, (len(indices)+1)*2))
    plt.suptitle(star, fontsize=14)
    plt.subplot(len(indices)+1, 1, 1)
    if "rv_err" not in df.columns: yerr = 0
    else: yerr = df.rv_err
    plt.errorbar(df.bjd - 2450000, df.rv, yerr, fmt='k.')
    plt.ylabel("RV [m/s]")
    #print(indices)
    for i, index in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        plt.ylabel(index)
        plt.errorbar(df.bjd - 2450000, df[index], df[index + "_err"], fmt='k.')
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.subplots_adjust(top=0.95)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

#########################
    
def stats_indice(star,cols,df):
    """
    Return pandas data frame with statistical data on the indice(s) given: max, min, mean, median, std and N (number of spectra)
    """
    df_stats = pd.DataFrame(columns=["star","indice","max","min","mean","median","std","time_span","N_spectra"])
    if len(cols) == 1:
            row = {"star":star,"column":cols,
                "max":max(df[cols]),"min":min(df[cols]),
                "mean":np.mean(df[cols]),"median":np.median(df[cols]),
                "std":np.std(df[cols]),"time_span":max(df["bjd"])-min(df["bjd"]),
                "N_spectra":len(df[cols])}
            df_stats.loc[len(df_stats)] = row
    elif len(cols) > 1:
        for i in cols:
            if i != "rv":
                indices = df[df[i+"_Rneg"] < 0.01].index
                data = df.loc[indices, i]
            else: data = df["rv"]
            if len(data) != 0:
                row = {"star": star, "indice": i,
                    "max": max(data), "min": min(data),
                    "mean": np.mean(data), "median": np.median(data),
                    "std": np.std(data), "time_span": max(df["bjd"]) - min(df["bjd"]),
                    "N_spectra": len(data)}
                df_stats.loc[len(df_stats)] = row
            else: 
                row = {"star": star, "indice": i,
                    "max": 0, "min": 0,
                    "mean": 0, "median": 0,
                    "std": 0, "time_span": max(df["bjd"]) - min(df["bjd"]),
                    "N_spectra": len(data)}
                df_stats.loc[len(df_stats)] = row


    else:
        print("ERROR: No columns given")
        df_stats = None
    
    return df_stats

########################

def get_rv_ccf(star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument):
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

    try:
        rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
        if instrument == "HARPS":
            rvmin = rv_simbad - 10; rvmax = rv_simbad + 10
        elif instrument == "UVES":
            rvmin = rv_simbad - 100; rvmax = rv_simbad + 100
        elif instrument == "ESPRESSO":
            rvmin = rv_simbad - 100; rvmax = rv_simbad + 100
    except:
        rvmin = -150; rvmax = 150

    #get the cross-correlation
    skipedge_values = [0, 100, 500, 1000, 5000, 20000, 50000]

    for skipedge in skipedge_values:
        try:
            rv, cc = pyasl.crosscorrRV(w=w_cut, f=f_cut, tw=tw,
                                    tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
            break  # Break out of the loop if successful
        except Exception as e:
            #print(f"Error with skipedge={skipedge}: {e}")
            #print(f"Error with skipedge={skipedge}")
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

########################

def plot_line(data, line, line_color,offset, line_legend="", lstyle = "-", normalize=True, legend_plot = False, plot_continuum_vlines = True, plot_lines_vlines = True):
    '''
    Plots the spectra used in the position of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,
             "NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 22
    else: window = 0.7

    for array in data:
        wv = array[0]; flux = array[1]
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))

        if wv_array[0].shape == (0,):
            continue

        wv = wv[wv_array]
        flux = flux[wv_array]
        #flux_normalized = flux/np.linalg.norm(flux)
        if normalize == True:
            flux_normalized = (flux-np.min(flux))/(np.max(flux)-np.min(flux))
        else: flux_normalized = flux
        plt.plot(wv, flux_normalized+offset, lstyle, label=line_legend, color=line_color)
        if len(flux_normalized) < 40:
            lim = 4
        else: lim = 19
        if plot_continuum_vlines == True:
            plt.axvline(x=wv[lim],ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=wv[-lim],ymin=0,ymax=1,ls="--",ms=0.1)
        if plot_lines_vlines == True:
            plt.axvline(x=line_wv-window/30,ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=line_wv+window/30,ymin=0,ymax=1,ls="--",ms=0.1)

    plt.axvline(x=line_wv,ymin=0,ymax=1,ls="-",ms=0.2)
    if legend_plot == True: plt.legend()
    
    if normalize == True: ylab = "Normalized Flux"
    else: ylab = "Flux"
    plt.xlabel(r"Wavelength ($\AA$)"); plt.ylabel(ylab)
    plt.title(f"{line} line")

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

        if len(flux_normalized) < 40:
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

#################################

def FWHM_power_peak(power,period):
    '''Computes the FWHM of the power peak. Finds the maximum power and then the points to the left and right with power lower
    than half the maximum. Used to compute the standard deviation of the period.'''
    max_power_index = np.argmax(power)
    max_power = power[max_power_index]
    # Determine the half-maximum power level
    half_max_power = max_power / 2
    # Find the left and right boundaries at half-maximum power
    try:
        right_boundary_index = np.where(power[:max_power_index] < half_max_power)[0][-1]
        right_boundary_period = period[right_boundary_index]
        left_boundary_index = np.where(power[max_power_index:] < half_max_power)[0][0] + max_power_index
        left_boundary_period = period[left_boundary_index]
        fwhm = right_boundary_period - left_boundary_period
    except: 
        right_boundary_period = 0
        left_boundary_period = 0
        fwhm=0
    return fwhm, left_boundary_period, right_boundary_period, half_max_power

def are_harmonics(period1, period2, tolerance=0.01):
    '''Check if the two periods given are harmonic.'''
    ratio = period1 / period2
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False
    
def get_harmonic_list(period):
    '''Compute the existence or not of harmonics in the given period list.'''
    harmonics_list = []
    for i in range(len(period)):
        for j in range(i+1, len(period)):
            if are_harmonics(period[j], period[i], tolerance=0.01):
                print(f"Period {period[i]} and {period[j]} are harmonics of each other")
                harmonics_list.append([period[i], period[j]])
    return harmonics_list

def periodogram_flagging(harmonics_list, period, period_err, power_list, plevels, t_span):
    '''
    Green/4: error < 20% and no harmonics in significant peaks
    Yellow/3: 30% > error > 20% and no harmonics in significant peaks
    Orange/2: harmonics in significant peaks, no error or error > 30%
    Red/1: many periods with power close to each under: if number of periods over 85% of the max > 0
           or if the period obtained is bigger than the time span
    Black/0: discarded - period under 1 yr or over 100 yrs, below FAP 1% level
    '''
    error = period_err/period * 100
    powers_close_max = [n for n in power_list if n > 0.85*np.max(power_list)]

    if np.max(power_list) < plevels[-1] or period < 365 or period > 100*365:
        flag = "black"
    else:
        if 0 < error <= 20 and len(harmonics_list) == 0 and period < t_span and len(powers_close_max) == 0:
            flag = "green"
        elif 20 < error <= 30 and len(harmonics_list) == 0 and period < t_span:
            flag = "yellow"
        elif (len(harmonics_list) > 0 or period_err == 0.0 or error > 30) and period < t_span:
            flag = "orange"
        elif len(powers_close_max) > 0 or period > t_span:
            flag = "red"

    return flag

def gaps_time(BJD):
    '''Takes the BJD array and returns the 10 biggest gaps in time.'''
    time_sorted = BJD[np.argsort(BJD)] #sorting time
    gaps = np.diff(time_sorted)
    gaps = gaps[np.argsort(gaps)][-10:]
    return gaps

def get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False):
    """Get GLS significant peaks and excludes peaks close to window function peaks and to gaps in BJD."""
    sign_peaks = [per for per, power in zip(df_peaks['peaks_period'], df_peaks['peaks_power']) if power > fap1]
    sign_peaks_win = [per for per, power in zip(df_peaks_WF['peaks_period_win'], df_peaks_WF['peaks_power_win']) if power > fap1]

    # exclude peaks close to win peaks
    exc_peaks = []
    for peak in sign_peaks:
        atol = peak * 0.1
        for peak_win in sign_peaks_win:
            if np.isclose(peak, peak_win, atol=atol):
                exc_peaks.append(peak)
                if verb:
                    print(f"{peak:.2f} is close to win peak {peak_win:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")
        if evaluate_gaps == True:
            for gap in gaps:
                if np.isclose(peak,gap, atol=atol):
                    exc_peaks.append(peak)
                    if verb:
                        print(f"{peak:.2f} is close to gap {gap:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")               

    sel_peaks = [peak for peak in sign_peaks if peak not in exc_peaks]

    return np.sort(sel_peaks)

#* GLS periodogram (Astropy):
def gls(star, x, y, y_err=None, pmin=1.5, pmax=1e4, steps=1e5):
    """Generalised Lomb-Scargle Periodogram using the `astropy` implementation.

    Args:
        x (array-like): Time coordinate.
        y (array-like): Y-coordinate.
        y_err (array-like, None): Y error if not None.
        pmin (float): Minimum period to compute periodogram.
        pmax (float): Maximum period to compute periodogram.
        steps (int): Step of frequency grid.

    Returns:
        (dictionary): Dictionary with following keys:
            period (ndarray): Period grid of periodogram.
            power (ndarray): GLS power at each period grid point.
            peaks_power (ndarray): GLS power at the peaks.
            peaks_period (ndarray): Period at the peaks.

            fap_maxp (float): FAP for period at maximum GLS power (highest peak).
            fap_5 (float): GLS power for 5% FAP level.
            fap_1 (float): GLS power for 1% FAP level.
            FAPS (ndarray): List of FAPs in the same order as periods and power

            power_win (ndarray): GLS power of window function at each period grid point.

    Ref: Zechmeister & Kürster (2009)
    """
    x = np.asarray(x); y = np.asarray(y)
    t = max(x) - min(x); n = len(x)

    #Nyquist frequencies computation
    k1 = pmax/t; fmin = 1./(k1 * t)
    k2 = pmin * n/t; fmax = n/(k2 * t)

    freq = np.linspace(fmin, fmax, int(steps))
    period = 1./freq

    gls = LombScargle(x, y, y_err)
    power = gls.power(freq)

    peaks, _ = find_peaks(power)
    #print("Peaks",peaks)
    sorted_peak_indices = np.argsort(power[peaks])[::-1]  # Sort in descending order of power
    sorted_peaks = peaks[sorted_peak_indices]
    peaks_power = power[sorted_peaks]
    peaks_period = period[sorted_peaks]
    
    fap_max_power = gls.false_alarm_probability(np.nanmax(power))
    faps = gls.false_alarm_probability(power)
    fap_levels = np.array([0.05, 0.01])
    fap5, fap1 = gls.false_alarm_level(fap_levels)

    #* Window function:
    y_win = np.ones_like(y)
    power_win = LombScargle(x, y_win, fit_mean=False, center_data=False).power(freq)
    
    results = dict()
    results['freq'] = freq; results['period'] = period
    results['power'] = power
    results['fap_maxp'] = fap_max_power
    results['fap_1'] = fap1; results['fap_5'] = fap5
    results['FAPS'] = faps
    # window function:
    results['power_win'] = power_win

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))
    fig.suptitle(f"GLS Astropy periodogram for {star} I_CaII", fontsize=12)
    axes[0].set_xlabel("Period [days]"); axes[0].set_ylabel("Power")
    axes[0].plot(period, power, 'b-')
    axes[0].set_title(f"Power vs Period for GLS Periodogram")
    plevels = [fap5,fap1]
    for i in range(len(fap_levels)): # Add the FAP levels to the plot
        axes[0].plot([min(period), max(period)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fap_levels[i]*100))
    axes[0].legend()
    
    fwhm, left_boundary_period, right_boundary_period, half_max_power = FWHM_power_peak(power,period)
    results["FWHM_period_best"] = fwhm
    results["std_period_best"] = fwhm / (2*np.sqrt(2*np.log(2)))

    period_max = peaks_period[0]
    results["period_best"] = period_max

    def sin_model(x, a, phi, omega):
        return a * np.sin(2 * np.pi * x/period_max + phi) + omega
    x_grid = np.arange(min(x), max(x), 0.01)
    p1, _ = curve_fit(sin_model, x, y)
    y_grid = sin_model(x_grid, *p1)
    axes[1].plot(x_grid, y_grid, 'b-')
    axes[1].errorbar(x,y,yerr=y_err,fmt='.', color='k',label='data')
    axes[1].set_xlabel('BJD $-$ 2450000 [days]'); axes[1].set_ylabel(r'$S_\mathrm{CaII}$')
    axes[1].set_title("Fitting the data with GLS")
    axes[1].legend()

    plt.figure(4, figsize=(7, 4))
    plt.xlabel("Period [days]"); plt.ylabel("Power")
    period_list_WF = period
    period_max_WF = period[np.argmax(power_win)]
    results["period_best_WF"] = period_max_WF
    plt.plot(period_list_WF, power_win, 'b-')
    plt.title(f"Power vs Period for Astropy GLS Periodogram {star} Window Function")
    for i in range(len(fap_levels)): # Add the FAP levels to the plot
        plt.plot([min(period_list_WF), max(period_list_WF)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fap_levels[i]*100))
    plt.legend()

    peaks_WF, _ = find_peaks(power_win)
    sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]  # Sort in descending order of power
    sorted_peaks = peaks_WF[sorted_peak_indices]
    peaks_power_win = power_win[sorted_peaks]
    peaks_period_win = period_list_WF[sorted_peaks]

    df_peaks = pd.DataFrame({"peaks_period":peaks_period,"peaks_power":peaks_power})
    df_peaks_WF = pd.DataFrame({"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win})
    gaps = gaps_time(x)
    print("Gaps in BJD:", gaps)
    sel_peaks = get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False)
    print("Significant Peaks:",sel_peaks)
    for peak in sel_peaks:
        axes[0].axvline(peak, ls='--', lw=0.7, color='orange')

    harmonics_list = get_harmonic_list(sel_peaks)
    period_err = results["std_period_best"]
    flag = periodogram_flagging(harmonics_list, period_max, period_err, power, plevels, t_span=t)
    print("Flag: ",flag)

    return results, gaps, flag, period_max, period_err, harmonics_list

def get_report_periodogram(hdr,gaps,period,period_err,flag_period,harmonics_list,folder_path):
    '''Writes into a txt file a tiny report on the periodogram.'''
    instr = hdr["INSTR"]; star = hdr["STAR_ID"]
    t_span = hdr["TIME_SPAN"]
    snr_min = hdr["SNR_MIN"]; snr_max = hdr["SNR_MAX"]
    n_spec = hdr["I_CAII_N_SPECTRA"]
    flag_rv = hdr["FLAG_RV"]

    name_file = folder_path+f"report_periodogram_{star}.txt"
    with open(name_file, "w") as f:
        f.write("###"*30+"\n")
        f.write("\n")
        f.write("Periodogram Report\n")
        f.write("---"*30+"\n")
        f.write(f"Star: {star}\n")
        f.write(f"Instrument: {instr}\n")
        f.write(f"SNR: {snr_min} - {snr_max}\n")
        f.write(f"RV flag: {flag_rv}\n")
        f.write(f"Time Span: {t_span}\n")
        f.write(f"Number of spectra: {n_spec}\n")
        f.write("---"*30+"\n")
        f.write(f"Period I_CaII: {period} +/- {period_err} days\n")
        f.write(f"Period flag: {flag_period}\n")
        f.write(f"Harmonics: {harmonics_list}\n")
        f.write(f"Time gaps between data points: {gaps}\n")
        f.write("\n")
        f.write("###"*30+"\n")
    
    f = open(name_file, 'r')
    file_contents = f.read()
    f.close()

    return file_contents

#################################
    
def general_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err, flag_period, flag_rv_ratio):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    if math.isnan(float(period_err)): period_err = 0

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]; #N_spectra = stats_df["N_spectra"][0]
    dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                "INSTR":[instr,"Instrument used"],
                "TIME_SPAN":[time_span, 'Time span in days between first and last observations used'],
                #"N_SPECTRA":[N_spectra,"Number of spectra used"],
                "SNR_MIN":[min_snr,"Minimum SNR"],
                "SNR_MAX":[max_snr,"Maximum SNR"],
                "PERIOD_I_CaII":[period,"Period of CaII activity index"],
                "PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
                "FLAG_PERIOD":[flag_period,"Goodness of periodogram fit flag. Color based."],
                "FLAG_RV":[flag_rv_ratio,"Goodness of RV correction indicador. 1 = all good"],
                "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                "COMMENT1":["RV obtained from CCF measure (m/s)","Comment"],
                "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}

    indices = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
    stats = ["max","min","mean","median","std","N_spectra"]

    for i,ind in enumerate(indices):
        for col in stats:
            stat = stats_df[col][i]
            if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
            elif col == "N_spectra": comment = f"Nr of spectra used in {ind}"
            else: comment = f"{col} of {ind}"
            dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]
    
    for keyword in dict_hdr.keys():
        hdr.append(("HIERARCH "+keyword, dict_hdr[keyword][0], dict_hdr[keyword][1]), end=True)

    df.columns = df.columns.astype(str)
    table = Table.from_pandas(df)
    hdul = fits.BinTableHDU(data=table, header=hdr)

    hdul.writeto(file_path, overwrite=True)

#################################

def read_bintable(file,print_info=False):
    '''
    Simple function to read the header and BinTable from the fits file that contains the statistics in the header and the data frame as BinTable
    '''
    hdul = fits.open(file)
    if print_info == True: hdul.info()
    hdr = hdul[1].header
    table = hdul[1].data
    astropy_table = Table(table)
    data_dict = astropy_table.to_pandas().to_dict()
    df = pd.DataFrame(data_dict)
    return df, hdr