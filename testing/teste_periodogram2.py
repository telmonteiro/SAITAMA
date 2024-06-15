import numpy as np, pandas as pd, matplotlib.pylab as plt, math
from PyAstronomy.pyTiming import pyPeriod # type: ignore
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error # type: ignore
from general_funcs import read_bintable
from periodogram_funcs import get_report_periodogram

def PyAstronomy_error(t,y,power, ofac, hifac, freq=None):
    '''Period error estimation of the periodogram based on the GLS implementation by PyAstronomy.'''
    N = len(y)
    if freq == None:
        th = t - np.min(t)
        tbase = np.max(th)
        #ofac = 10; hifac = 1 #default values in PyAstronomy
        
        fstep = 1 / ofac / tbase; fnyq = 1/2 / tbase * N #0.01121490182039829
        fbeg = fstep; fend = fnyq * hifac
        
        freq = np.arange(fbeg, fend, fstep)

    nf = len(freq)

    k = np.argmax(power)
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


def VanderPlas_52(t,y,y_err,power,period,T_span,approximate=False):
    '''Computing the standard deviation of the peak using equation 52 from "Understanding the Lomb-Scargle Periodogram", 
    VanderPlas 2018.'''
    max_power_index = np.argmax(power)
    if approximate == True:
        f_1_2 = T_span
    else: 
        max_power = power[max_power_index]
        half_max_power = max_power / 2
        left_boundary_index = np.where(power[max_power_index:] < half_max_power)[0][0] + max_power_index
        left_boundary_period = period[left_boundary_index]
        p_1_2 = period[max_power_index] - left_boundary_period
        f_1_2 = 1/p_1_2
    # Number of data points
    N = len(list(t))

    # Calculate signal-to-noise ratio (SNR)
    def sin_model(t, a, phi, omega):
        return a * np.sin(2 * np.pi * t * (1/period[max_power_index]) + phi) + omega
    p1, _ = curve_fit(sin_model, t, y)
    y_expected = sin_model(t, *p1)

    mu = y_expected
    sigma_n = y_err
    x = (y - mu) / sigma_n
    Sigma = np.sqrt(np.mean(x**2)) #computing the rms to obtain the SNR

    sigma_f = f_1_2 * np.sqrt(2 / (N * Sigma ** 2)) #in frequency

    sigma_p = sigma_f / (1/period[max_power_index])**2

    return sigma_p

def curve_fit_error(x,y, period_max):
    '''Computes the error of the period obtained from periodogram by fitting a curve_fit restrained enough so that the period obtained
    is the same as from the periodogram.'''
    def sin_model(x, a, phi, omega, m,period):
        return a * np.sin(2 * np.pi * x/period + phi) + omega + m*x
    
    #bounds=((0, -np.inf, -np.inf, 0.99999*period_max), (np.inf, np.inf, np.inf, 1.00001*period_max))
    bounds = ((0, -np.inf, -np.inf, -np.inf, period_max-300), (np.inf, np.inf, np.inf, np.inf, period_max+300))

    p1, pcov1 = curve_fit(sin_model, x, y, bounds=bounds)

    return p1, np.sqrt(np.diag(pcov1))

def are_harmonics(period1, period2, tolerance=0.01):
    ratio = period1 / period2
    if period1 < 2*365 or period2 < 2*365:
        return False
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False
    
def get_harmonic_list(period, print=False):
    harmonics_list = []
    for i in range(len(period)):
        for j in range(i+1, len(period)):
            if are_harmonics(period[j], period[i], tolerance=0.01):
                if print == True:
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
    print("close powers",powers_close_max)
    print("harmonics",harmonics_list)
    print("tspan",t_span)
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

    period_max = peaks_period[0]
    results["period_best"] = period_max

    plt.figure(5,figsize=(8,6))
    ofac_array = np.linspace(1,50,50)
    hifac_array = np.linspace(0.001,1,5)
    for hifac in hifac_array:
        error_list = []
        for ofac in ofac_array:
            e_f, Psin_err = PyAstronomy_error(x,y,power,ofac, hifac, freq=None)
            error_list.append(Psin_err)
        plt.plot(ofac_array,error_list,label=f"hifac = {hifac}")
    plt.xlabel("ofac = oversampling factor of frequency grid")
    plt.ylabel("Period error with PyAstronomy [days]")
    plt.legend(title="hifac = maximum frequency")
    
    print(f"Error period with PyAstronomy: {Psin_err}")
    results["std_period_best"] = Psin_err

    period_list_WF = period
    period_max_WF = period[np.argmax(power_win)]
    results["period_best_WF"] = period_max_WF

    peaks_WF, _ = find_peaks(power_win)
    sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]  # Sort in descending order of power
    sorted_peaks = peaks_WF[sorted_peak_indices]
    peaks_power_win = power_win[sorted_peaks]
    peaks_period_win = period_list_WF[sorted_peaks]

    df_peaks = pd.DataFrame({"peaks_period":peaks_period,"peaks_power":peaks_power})
    df_peaks_WF = pd.DataFrame({"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win})
    gaps = gaps_time(x)
    sel_peaks = get_sign_gls_peaks(df_peaks, df_peaks_WF, gaps, fap1, atol_frac=0.1, verb=False, evaluate_gaps=False)

    harmonics_list = get_harmonic_list(sel_peaks)
    period_err = results["std_period_best"]
    flag = periodogram_flagging(harmonics_list, period_max, period_err, peaks_power, plevels, t_span)
    print("Flag: ",flag)

    print(f"Period with astropy: {period_max} days")

    pyastronomy_error = Psin_err

    return results, gaps, flag, period_max, period_err, harmonics_list, pyastronomy_error

instr = "HARPS"
stars = ["HD209100"]

print(f"{stars[0]} with {instr} data")
folder_path = f"teste_download_rv_corr/{stars[0]}/{stars[0]}_{instr}/"
file_path = folder_path+f"df_stats_{stars[0]}.fits"
df, hdr = read_bintable(file_path,print_info=False)

t_span = max(df["bjd"])-min(df["bjd"])
n_spec = len(df)

if n_spec >= 50 and t_span >= 2*365:  #only compute periodogram if star has at least 30 spectra in a time span of at least 2 years

    results, gaps, flag_period, period_max, period_err, harmonics_list, pyastronomy_error = gls(stars[0], df["bjd"]-2450000, df["I_CaII"], y_err=df["I_CaII_err"], pmin=1.5, pmax=1e4, steps=1e5)
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='power', ascending=False)

else: 
    period = None; period_err = None 

plt.show()