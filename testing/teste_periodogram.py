import numpy as np, pandas as pd, matplotlib.pylab as plt
from PyAstronomy.pyTiming import pyPeriod # type: ignore
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from general_funcs import read_bintable
from periodogram_funcs import get_report_periodogram

def FWHM_power_peak(power,period):
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


def VanderPlas_52(power,period,T_span,approximate=True):
    '''Computing the standard deviation of the peak using equation 52 from "Understanding the Lomb-Scargle Periodogram", 
    VanderPlas 2018.
    Warning: I have serious doubts if this is well applied. Can't find good info about how the SNR is properly computed.
    The number of samples N is also the sampling of the periodogram, not the real number of observations.'''
    if approximate == True:
        f_1_2 = T_span
    else: 
        max_power_index = np.argmax(power)
        max_power = power[max_power_index]
        half_max_power = max_power / 2
        left_boundary_index = np.where(power[max_power_index:] < half_max_power)[0][0] + max_power_index
        left_boundary_period = period[left_boundary_index]
        f_1_2 = period[max_power_index] - left_boundary_period

    # Number of data points
    N = len(period)

    # Calculate signal-to-noise ratio (SNR)
    mu = np.mean(power)
    sigma_n = np.std(power)
    x = (power - mu) / sigma_n
    Sigma = np.sqrt(np.sum(x**2) / N) #computing the rms to obtain the SNR

    sigma_f = f_1_2 * np.sqrt(2 / (N * Sigma ** 2))

    return sigma_f

def are_harmonics(period1, period2, tolerance=0.01):
    ratio = period1 / period2
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False
    
def get_harmonic_list(period):
    harmonics_list = []
    for i in range(len(period)):
        for j in range(i+1, len(period)):
            if are_harmonics(period[j], period[i], tolerance=0.01):
                print(f"Period {period[i]} and {period[j]} are harmonics of each other")
                harmonics_list.append([period[i], period[j]])
    return harmonics_list

def periodogram_flagging(harmonics_list, period, period_err, peaks_power, plevels, t_span):
    '''
    Green/4: error < 20% and no harmonics in significant peaks involving the max peak
    Yellow/3: 30% > error > 20% and no harmonics in significant peaks involving the max peak
    Orange/2: harmonics involving the max peak, no error or error > 30%
    Red/1: many periods with power close to each under: if number of periods over 85% of the max > 0
           or if the period obtained is bigger than the time span
    Black/0: discarded - period under 1 yr or over 100 yrs, below FAP 1% level
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


def WF_periodogram(star, bjd, print_info, save, path_save):
    #time_array = np.linspace(np.min(bjd), np.max(bjd), int(np.max(bjd) - np.min(bjd)))
    #zeros_array = np.zeros_like(time_array); ones_array = np.ones_like(bjd)
    #time = np.concatenate((time_array, bjd))
    #ones_zeros = np.concatenate((zeros_array, ones_array))

    #print(f"BJD array has {bjd.shape[0]} and ones_zeros has {len([x for x in ones_zeros if x == 1])} 1s")
    #time = bjd; ones_zeros = np.ones_like(bjd)

    #random_array = np.random.normal(size=len(bjd))*0+1
    clp_WF = pyPeriod.Gls((bjd, np.ones_like(bjd)), verbose=print_info)
    dic_clp_WF = clp_WF.info(noprint=True)
    period_WF = dic_clp_WF["best_sine_period"]; period_err_WF = dic_clp_WF["best_sine_period_err"]
    fapLevels = np.array([0.05, 0.01]) # Define FAP levels of 5% and 1%
    plevels = clp_WF.powerLevel(fapLevels)

    plt.figure(2, figsize=(7, 4))
    plt.xlabel("Period [days]"); plt.ylabel("Power")
    period_list_WF = 1/clp_WF.freq
    #print(clp_WF.power)
    #harmonics_list = get_harmonic_list(period_list_WF, clp_WF.power, plevels)
    plt.plot(period_list_WF, clp_WF.power, 'b-')
    plt.xlim([0, period_WF+500])
    plt.title(f"Power vs Period for GLS Periodogram {star} Window Function")
    
    for i in range(len(fapLevels)): # Add the FAP levels to the plot
        plt.plot([min(period_list_WF), max(period_list_WF)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fapLevels[i]*100))
    plt.legend()

    if save == True: plt.savefig(path_save, bbox_inches="tight")

    return round(period_WF,3), round(period_err_WF,3)

def gaps_time(BJD):
    '''Takes the BJD array and returns the 10 biggest gaps in time.'''
    time_sorted = BJD[np.argsort(BJD)] #sorting time
    gaps = np.diff(time_sorted)
    gaps = gaps[np.argsort(gaps)][-10:]
    return gaps

def gls_periodogram(star, I_CaII, I_CaII_err, bjd, print_info, save, path_save, mode="Period"):
    t_span = max(bjd)-min(bjd)
    # Compute the GLS periodogram with default options. Choose Zechmeister-Kuerster normalization explicitly
    clp = pyPeriod.Gls((bjd - 2450000, I_CaII, I_CaII_err), norm="ZK", verbose=print_info,ofac=30)
    dic_clp = clp.info(noprint=True)
    period = dic_clp["best_sine_period"]; period_err = dic_clp["best_sine_period_err"]
    fapLevels = np.array([0.05, 0.01]) # Define FAP levels of 5% and 1%
    plevels = clp.powerLevel(fapLevels)

    plt.figure(1, figsize=(13, 4))
    plt.suptitle(f"GLS periodogram for {star} I_CaII", fontsize=12)
    # and plot power vs. period
    plt.subplot(1, 2, 1); plt.xlabel("Period [days]"); plt.ylabel("Power")
    period_list = 1/clp.freq

    harmonics_list = get_harmonic_list(period_list, clp.power, plevels)
    flag = periodogram_flagging(harmonics_list, period, period_err, clp.power, plevels, t_span)
    print("Flag: ",flag)
    
    plt.plot(period_list, clp.power, 'b-')
    plt.xlim([0, 10000])
    plt.title(f"Power vs {mode} for GLS Periodogram")
    
    for i in range(len(fapLevels)): # Add the FAP levels to the plot
        plt.plot([min(period_list), max(period_list)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fapLevels[i]*100))
    plt.legend()

    plt.subplot(1, 2, 2)
    timefit = np.linspace(min(bjd - 2450000),max(bjd - 2450000),500)
    plt.plot(timefit,clp.sinmod(timefit),label="fit")
    plt.errorbar(bjd - 2450000,I_CaII,yerr=I_CaII_err,fmt='.', color='k',label='data')
    plt.xlabel('BJD $-$ 2450000 [days]'); plt.ylabel(r'$S_\mathrm{CaII}$'); plt.title("Fitting the data with GLS")
    plt.legend()

    plt.subplots_adjust(top=0.85)
    if save == True: plt.savefig(path_save, bbox_inches="tight")

    return round(period,3), round(period_err,3), flag, harmonics_list

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
    axes[0].semilogx(period, power, 'b-')
    axes[0].set_title(f"Power vs Period for GLS Periodogram")
    plevels = [fap5,fap1]
    for i in range(len(fap_levels)): # Add the FAP levels to the plot
        axes[0].plot([min(period), max(period)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fap_levels[i]*100))
    axes[0].legend()
    
    #fwhm, left_boundary_period, right_boundary_period, half_max_power = FWHM_power_peak(power,period)
    #axes[0].axvline(left_boundary_period,color="r"); axes[0].axvline(right_boundary_period,color="r"); axes[0].axhline(half_max_power,color="r")
    #results["FWHM_period_best"] = fwhm
    #results["std_period_best"] = fwhm / (2*np.sqrt(2*np.log(2)))
    T_span = t
    uncertainty = VanderPlas_52(power,period,T_span)
    results["std_period_best"] = uncertainty

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
    plt.semilogx(period_list_WF, power_win, 'b-',label="WF")
    plt.semilogx(period, power, 'r-',lw=0.7,label="data")
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
    for gap in gaps:
        plt.axvline(gap, ls='--', lw=0.7, color='green')
    sel_peaks = get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False)
    print("Significant Peaks:",sel_peaks)
    for peak in sel_peaks:
        axes[0].axvline(peak, ls='--', lw=0.8, color='orange')

    harmonics_list = get_harmonic_list(sel_peaks)
    period_err = results["std_period_best"]
    flag = periodogram_flagging(harmonics_list, period_max, period_err, peaks_power, plevels, t_span)
    print("Flag: ",flag)

    return results, gaps, flag, period_max, period_err, harmonics_list


stars = ["HD209100"]
instr = "HARPS"
#stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
#        'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
#        'HD20794',"HD85512","HD192310"] 

for star in stars:
    print(f"{star} with {instr} data")
    folder_path = f"teste_download_rv_corr/{star}/{star}_{instr}/"
    file_path = folder_path+f"df_stats_{star}.fits"
    df, hdr = read_bintable(file_path,print_info=False)

    t_span = max(df["bjd"])-min(df["bjd"])
    n_spec = len(df)
    if n_spec >= 30 and t_span >= 2*365:  #only compute periodogram if star has at least 30 spectra in a time span of at least 2 years

        #period, period_err, flag_period, harmonics_list = gls_periodogram(star, df["I_CaII"],df["I_CaII_err"],df["bjd"], 
        #                                    print_info = False, save=False, path_save=folder_path+f"{star}_GLS.png")
        #print(f"Period of I_CaII: {period} +/- {period_err} days")

        #period_WF, period_err_WF = WF_periodogram(star, df["bjd"]-2450000, print_info=False, 
        #                                        save=False, path_save=folder_path+f"{star}_WF.png")
        #print(f"Period of WF: {period_WF} +/- {period_err_WF} days")

        results, gaps, flag_period, period_max, period_err, harmonics_list = gls(star, df["bjd"]-2450000, df["I_CaII"], y_err=df["I_CaII_err"], pmin=1.5, pmax=1e4, steps=1e5)
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='power', ascending=False)
        #print(df_sorted)
        report_periodogram = get_report_periodogram(hdr,gaps,period_max,period_err,flag_period,harmonics_list,folder_path=folder_path)
        print(report_periodogram)
        
    else: 
        period = None; period_err = None 

plt.show()
