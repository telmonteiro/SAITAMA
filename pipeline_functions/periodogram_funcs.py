'''
This file contains functions to be used to compute the activity periodogram in the SAITAMA pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from astropy.table import Table
from PyAstronomy import pyasl # type: ignore
from PyAstronomy.pyTiming import pyPeriod # type: ignore
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def PyAstronomy_error(t,y,power,power_best,freq=None):
    '''Period error estimation of the periodogram based on the GLS implementation by PyAstronomy.'''
    N = len(y)
    if isinstance(freq, np.ndarray) == False:
        th = t - np.min(t)
        tbase = np.max(th)
        ofac = 10; hifac = 1 #default values in PyAstronomy
        
        fstep = 1 / ofac / tbase; fnyq = 1/2 / tbase * N
        fbeg = fstep; fend = fnyq * hifac
        
        freq = np.arange(fbeg, fend, fstep)

    nf = len(freq)

    k = np.where(power == power_best)[0][0]
    
    # Maximum power
    pmax = power[k]
    
    # Get the curvature in the power peak by fitting a parabola y=aa*x^2
    if 1 < k < nf-2:
        # Shift the parabola origin to power peak
        xh = (freq[k-1:k+2] - freq[k])**2
        yh = power[k-1:k+2] - pmax
        # Calculate the curvature (final equation from least square)
        aa = np.dot(yh, xh) / np.dot(xh, xh)
        e_f = np.sqrt(-2./N / aa * (1.-pmax))
        Psin_err = e_f / freq[k]**2
    else:
        e_f = np.nan
        Psin_err= np.nan
    
    return e_f, Psin_err

##################################################################################################################

def are_harmonics(period1, period2, tolerance=0.01):
    '''
    Get boolean based on whether two periods are harmonics of each other or not.
    '''
    ratio = period1 / period2
    if period1 < 2*365 or period2 < 2*365:
        return False
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False
    
##################################################################################################################

def get_harmonic_list(period, print_info=False):
    '''
    Get list of the harmonics in the list of periods.
    '''
    harmonics_list = []
    for i in range(len(period)):
        for j in range(i+1, len(period)):
            if are_harmonics(period[j], period[i], tolerance=0.01):
                if print_info == True:
                    print(f"Period {period[i]} and {period[j]} are harmonics of each other")
                harmonics_list.append([period[i], period[j]])
    return harmonics_list

##################################################################################################################

def periodogram_flagging(harmonics_list, period, period_err, peaks_power, plevels, t_span):
    '''
    Color-based quality indicator of the period obtained from the GLS periodogram.
    Green/4: error < 20% and no harmonics in significant peaks involving the max peak
    Yellow/3: 30% > error > 20% and no harmonics in significant peaks involving the max peak
    Orange/2: harmonics involving the max peak, no error or error > 30%
    Red/1: many periods with power close to each under: if number of periods over 85% of the max > 0
           or if the period obtained is bigger than the time span
    Black/0: discarded - period under 1 yr or over 100 yrs, below FAP 1% level, no significant peak obtained
    '''
    harmonics_list = list(np.unique(np.array(harmonics_list)))
    error = period_err/period * 100
    powers_close_max = [n for n in peaks_power if n > 0.9*np.max(peaks_power) and n != np.max(peaks_power)]

    if np.max(peaks_power) < plevels[-1] or period < 365 or period > 100*365:
        flag = "black"
    else:
        if 0 < error <= 20 and (period not in harmonics_list) and period < t_span and len(powers_close_max) == 0:
            flag = "green"
        elif 20 < error <= 30 and period not in harmonics_list and period < t_span and len(powers_close_max) == 0:
            flag = "yellow"
        elif (period in harmonics_list or period_err == 0.0 or error > 30) and period < t_span and len(powers_close_max) == 0:
            flag = "orange"
        elif len(powers_close_max) > 0 or period > t_span:
            flag = "red"

    return flag

##################################################################################################################

def gaps_time(BJD):
    '''Takes the BJD array and returns the 10 biggest gaps in time.'''
    time_sorted = BJD[np.argsort(BJD)] #sorting time
    gaps = np.diff(time_sorted)
    gaps = gaps[np.argsort(gaps)][-10:]
    return gaps

##################################################################################################################

def get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False):
    """Get GLS significant peaks and excludes peaks close to window function peaks and to gaps in BJD."""
    sign_peaks = [per for per, power in zip(df_peaks['peaks_period'], df_peaks['peaks_power']) if power > fap1]
    sign_peaks_power = [power for per, power in zip(df_peaks['peaks_period'], df_peaks['peaks_power']) if power > fap1]
    sign_peaks_win = [per for per, power in zip(df_peaks_WF['peaks_period_win'], df_peaks_WF['peaks_power_win']) if power > fap1]

    # exclude peaks close to win peaks
    exc_peaks = []; exc_peaks_power = []
    for ind,peak in enumerate(sign_peaks):
        atol = peak * 0.1
        for peak_win in sign_peaks_win:
            if np.isclose(peak, peak_win, atol=atol):
                exc_peaks.append(peak)
                exc_peaks_power.append(ind)
                if verb:
                    print(f"{peak:.2f} is close to win peak {peak_win:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")
        if evaluate_gaps == True:
            for gap in gaps:
                if np.isclose(peak,gap, atol=atol):
                    exc_peaks.append(peak)
                    exc_peaks_power.append(ind)
                    if verb:
                        print(f"{peak:.2f} is close to gap {gap:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")               

    sel_peaks = [peak for peak in sign_peaks if peak not in exc_peaks]  
    sel_peaks_power = [pow for ind, pow in enumerate(sign_peaks_power) if ind not in exc_peaks_power]

    return sel_peaks, sel_peaks_power

##################################################################################################################

#* GLS periodogram (Astropy):
def gls(star, instr, df, x, y, y_err=None, pmin=1.5, pmax=1e4, steps=1e5, print_info = True, save=False, folder_path=None):
    """Generalised Lomb-Scargle Periodogram using the `astropy` implementation.
    Args:
        x (array-like): Time coordinate.
        y (array-like): Y-coordinate.
        y_err (array-like, None): Y error if not None.
        pmin (float): Minimum period to compute periodogram.
        pmax (float): Maximum period to compute periodogram.
        steps (int): Step of frequency grid.
        print_info (bool): Whether to print information or not.
        save (bool): Whether to save the plots or not.
        folder_path (str): Path to save the plots.

    Returns:
        (dictionary): Dictionary with following keys:
            period (ndarray): Period grid of periodogram.
            power (ndarray): GLS power at each period grid point.
            peaks_power (ndarray): GLS power at the peaks.
            peaks_period (ndarray): Period at the peaks.

            fap_maxp (float): FAP for period at maximum GLS power (highest peak).
            fap_5 (float): GLS power for 5% FAP level.
            fap_1 (float): GLS power for 1% FAP level.
            FAPS (ndarray): List of FAPs in the same order as periods and power.

            power_win (ndarray): GLS power of window function at each period grid point.
            period_best_WF (float): Period of the significant peak of the WF with most power.
            
            period_best (float): Period of the significant peak with most power.
            std_period_best (float): Error of the period.
            
        gaps (list): 10 biggest gaps in BJD time.
        flag (str): Color-based period quality indicator.
        period_best (float): Period of the significant peak with most power.
        period_err (float): Error of the period.
        harmonics_list (list): Harmonics in the significant peaks.
        amplitude (float): 
        amplitude_err (float): 
       
        (.pdf): Plots of the periodogram.
    """
    x = x.dropna(); y = y.dropna(); y_err = y_err.dropna()

    x = np.asarray(x); y = np.asarray(y)
    t = max(x) - min(x); n = len(x)

    #Nyquist frequencies computation
    k1 = pmax/t; fmin = 1./(k1 * t)
    k2 = pmin * n/t; fmax = n/(k2 * t)

    freq = np.linspace(fmin, fmax, int(steps))
    period = 1./freq

    #get power from GLS
    gls = LombScargle(x, y, y_err)
    power = gls.power(freq)

    peaks, _ = find_peaks(power)
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

    plevels = [fap5,fap1]

    period_list_WF = period
    period_best_WF = period[np.argmax(power_win)]
    results["period_best_WF"] = period_best_WF

    peaks_WF, _ = find_peaks(power_win)
    sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]  # Sort in descending order of power
    sorted_peaks = peaks_WF[sorted_peak_indices]
    peaks_power_win = power_win[sorted_peaks]
    peaks_period_win = period_list_WF[sorted_peaks]

    df_peaks = pd.DataFrame({"peaks_period":peaks_period,"peaks_power":peaks_power})
    df_peaks_WF = pd.DataFrame({"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win})
    gaps = gaps_time(x)
    sel_peaks, sel_peaks_power = get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False)

    try:
        period_best = sel_peaks[0]
        power_best = sel_peaks_power[0]
    except: #there is no significant peaks in the periodogram
        period_best = 0
        power_best = 0

        period_err = 0
        harmonics_list = []
        amplitude = 0; amplitude_err = 0

        flag = "black"

        return results, gaps, flag, period_best, period_err, harmonics_list, amplitude, amplitude_err

    results["period_best"] = period_best
    results["power_best"] = power_best

    #PyAstronomy method
    e_f, pyastronomy_error = PyAstronomy_error(x,y,power,power_best,freq)
    results["std_period_best"] = pyastronomy_error #taken as the final error
    
    #plot of the periodogram and of the fitted period
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.2, 4))
    axes[0].text(0.13, 0.89, star, fontsize=13, transform=plt.gcf().transFigure)
    axes[0].set_xlabel("Period [days]"); axes[0].set_ylabel("Power")
    axes[0].semilogx(period, power, 'k-')
    plevels = [fap5,fap1]
    for i in range(len(fap_levels)):
        axes[0].plot([min(period), max(period)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fap_levels[i]*100))

    vline_legend = Line2D([0], [0], color='red', linestyle='--', label='Significant peaks')
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(vline_legend)
    axes[0].legend(handles=handles)

    x_grid = np.arange(min(x), max(x), 0.01)
    def sin_model(x, a, phi, omega):
        return a * np.sin(2 * np.pi * x/period_best + phi) + omega
    p1, cov1 = curve_fit(sin_model, x, y)
    amplitude = p1[0]; amplitude_err = cov1[0]
    y_grid = sin_model(x_grid, *p1)
    for peak in sel_peaks:
        axes[0].axvline(peak, ls='--', lw=0.8, color='red')

    axes[1].plot(x_grid, y_grid, 'k-')

    instruments = df["instr"].unique()
    for i, instrument in enumerate(instruments):
        instr_df = df[df["instr"] == instrument]
        if np.isnan(list(instr_df["I_CaII"])[0]) == True:
            continue
        axes[1].errorbar(instr_df["bjd"] - 2450000, instr_df["I_CaII"], yerr=instr_df["I_CaII_err"], fmt='.', alpha=0.7, label=instrument)

    axes[1].set_xlabel('BJD $-$ 2450000 [days]'); axes[1].set_ylabel(r'$S_\mathrm{Ca II}$')
    axes[1].legend()

    #plt.tight_layout()
    if save == True:
        if instr == None:
            plt.savefig(folder_path+ f"{star}_GLS.pdf",bbox_inches="tight",)
        else:
            plt.savefig(folder_path+ f"{star}_{instr}_GLS.pdf",bbox_inches="tight",)

    #plot of the WF periodogram
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    axes.set_xlabel("Period [days]"); axes.set_ylabel("Power")
    axes.semilogx(period_list_WF, power_win, 'b-',label="WF")
    axes.semilogx(period, power, 'r-',lw=0.7,label="data")
    axes.set_title(f"Window Function GLS Periodogram for {star} ")
    for i in range(len(fap_levels)): # Add the FAP levels to the plot
        axes.plot([min(period_list_WF), max(period_list_WF)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fap_levels[i]*100))
    for gap in gaps:
        axes.axvline(gap, ls='--', lw=0.7, color='green')
    axes.legend()
    #plt.tight_layout()
    if save == True:
        if instr == None:
            plt.savefig(folder_path+ f"{star}_GLS_WF.pdf",bbox_inches="tight",)
        else:
            plt.savefig(folder_path+ f"{star}_{instr}_GLS_WF.pdf",bbox_inches="tight",)

    #harmonics and period quality flag
    harmonics_list = get_harmonic_list(sel_peaks,print_info)
    period_err = results["std_period_best"]
    flag = periodogram_flagging(harmonics_list, period_best, period_err, peaks_power, plevels, pmax)
    
    if print_info == True:
        print("Gaps in BJD:", gaps)
        print("Significant Peaks:", sel_peaks)
        print("Period flag: ", flag)

    return results, gaps, flag, period_best, period_err, harmonics_list, amplitude, amplitude_err

##################################################################################################################

def get_report_periodogram(dic, gaps, period, period_err, amplitude, amplitude_err, flag_period, harmonics_list, folder_path):
    '''Writes into a txt file a report on the periodogram.'''
    instr = dic["INSTR"]; star = dic["STAR_ID"]
    t_span = dic["TIME_SPAN"]
    snr_min = dic["SNR_MIN"]; snr_max = dic["SNR_MAX"]
    n_spec = dic["I_CAII_N_SPECTRA"]

    name_file = folder_path+f"report_periodogram_{star}.txt"
    with open(name_file, "w") as f:
        f.write("###"*30+"\n")
        f.write("\n")
        f.write("Periodogram Report\n")
        f.write("---"*30+"\n")
        f.write(f"Star: {star}\n")
        f.write(f"Instrument: {instr}\n")
        f.write(f"SNR: {snr_min} - {snr_max}\n")
        f.write(f"Time Span: {t_span}\n")
        f.write(f"Number of spectra: {n_spec}\n")
        f.write("---"*30+"\n")
        f.write(f"Period I_CaII: {period} +/- {period_err} days\n")
        f.write(f"Period flag: {flag_period}\n")
        f.write(f"Amplitude I_CaII: {amplitude} +/- {amplitude_err}\n")
        f.write(f"Harmonics: {harmonics_list}\n")
        f.write(f"Time gaps between data points: {gaps}\n")
        f.write("\n")
        f.write("###"*30+"\n")
    
    f = open(name_file, 'r')
    file_contents = f.read()
    f.close()

    return file_contents