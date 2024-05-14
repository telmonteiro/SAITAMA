'''
This file contains functions to be used to compute the activity periodogram in the ACTINometer pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from astropy.table import Table
from PyAstronomy import pyasl # type: ignore
from PyAstronomy.pyTiming import pyPeriod # type: ignore
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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