import numpy as np, pandas as pd, matplotlib.pylab as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pipeline_functions.general_funcs import read_bintable
from matplotlib.lines import Line2D

def gaussian_fit_error(power,period,period_best,power_best):
    '''Computes the error of the period obtained from periodogram by fitting a Gaussian in a region of [0.99*P, 1.01*P].
    The sigma of the Gaussian is taken as the error.'''
    def gauss(x, H, A, x0, sigma): 
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    interval = np.where((period > 0.9*period_best) & (period < 1.1*period_best))    
    bounds = ((0, power_best*0.5, period_best*0.5, 0), (np.inf, power_best*1.5, period_best*1.5, period_best))
    p0 = (0,power_best,period_best,period_best/5)

    popt, pcov = curve_fit(gauss, period[interval], power[interval], bounds=bounds, p0=p0) 

    x_grid = np.linspace(period[interval][0], period[interval][-1], 100)
    y_grid = gauss(x_grid, *popt)

    plt.figure(1)
    plt.scatter(period[interval], power[interval])
    plt.plot(x_grid,y_grid,c="black")
    plt.ylabel("Power"); plt.xlabel("Period [days]")
    #plt.show()
    
    return popt, np.sqrt(np.diag(pcov))


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


def VanderPlas_52(t,y,y_err,power,period,period_best,power_best,T_span,approximate=False):
    '''Computing the standard deviation of the peak using equation 52 from "Understanding the Lomb-Scargle Periodogram", 
    VanderPlas 2018.'''
    max_power_index = np.where(power == power_best)[0][0]

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


def curve_fit_error(x,y, period_best):
    '''Computes the error of the period obtained from periodogram by fitting a curve_fit restrained enough so that the period obtained
    is the same as from the periodogram.'''
    def sin_model(x, a, phi, omega, m, period):
        return a * np.sin(2 * np.pi * x/period + phi) + omega + m*x
    
    bounds=((0, -np.inf, -np.inf, -np.inf, 0.99999*period_best), (np.inf, np.inf, np.inf, np.inf, 1.00001*period_best))

    p1, pcov1 = curve_fit(sin_model, x, y, bounds=bounds)

    return p1, np.sqrt(np.diag(pcov1))


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
    
def get_harmonic_list(period, print=False):
    '''
    Get list of the harmonics in the list of periods.
    '''
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


#* GLS periodogram (Astropy):
def gls(star, instr, x, y, y_err=None, pmin=1.5, pmax=1e4, steps=1e5):
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
    period_best = sel_peaks[0]
    power_best = sel_peaks_power[0]
    results["period_best"] = period_best
    results["power_best"] = power_best

    print(f"Period with astropy: {period_best} days")
    print(f"Maximum power period: {period[np.argmax(power)]}")

    #Period error methods
    #PyAstronomy method
    e_f, pyastronomy_error = PyAstronomy_error(x,y,power,power_best,freq)
    print(f"Error period with PyAstronomy: {pyastronomy_error}")
    results["std_period_best"] = pyastronomy_error #taken as the final error

    
    #Error from curve_fit parameter
    params, stds = curve_fit_error(x,y, period_best)
    curve_fit_err = stds[-1]
    print(f"Period with curve_fit: {params[-1]} +/- {curve_fit_err} days")
    print(f"Amplitude: {params[0]} +/- {stds[0]} ")


    #Gaussian fit error
    params_gauss, stds_gauss = gaussian_fit_error(power,period,period_best,power_best)
    print("Gaussian fit:")
    #print(f"H: {params_gauss[0]} +/- {stds_gauss[0]}")
    #print(f"A: {params_gauss[1]} +/- {stds_gauss[1]}")
    print(f"x0: {params_gauss[2]} +/- {stds_gauss[2]}")
    print(f"sigma: {params_gauss[3]} +/- {stds_gauss[3]}")
    gaussian_fit_err = params_gauss[3]
    #print(f"Period with Gaussian fit: {params_gauss[3]} +/- {stds_gauss[3]} days")


    #VanderPlas 2018 equation 52 
    VanderPlas_52_error = VanderPlas_52(x,y,y_err,power,period,period_best,power_best,t, approximate=False)
    print(f"Error in period with VanderPlas_52: {VanderPlas_52_error}")


    #plot of the periodogram and of the fitted period
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))
    #fig.suptitle(f"GLS periodogram for {star} I_CaII", fontsize=12)
    axes[0].text(0.13, 0.89, star, fontsize=13, transform=plt.gcf().transFigure)
    axes[0].set_xlabel("Period [days]"); axes[0].set_ylabel("Power")
    axes[0].semilogx(period, power, 'k-')
    #axes[0].set_title(f"Power vs Period")
    plevels = [fap5,fap1]
    for i in range(len(fap_levels)): # Add the FAP levels to the plot
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
    y_grid = sin_model(x_grid, *p1)
    for peak in sel_peaks:
        axes[0].axvline(peak, ls='--', lw=0.8, color='red')

    axes[1].plot(x_grid, y_grid, 'k-')
    axes[1].errorbar(x,y,yerr=y_err,fmt='.', color='r',alpha=0.5)
    axes[1].set_xlabel('BJD $-$ 2450000 [days]'); axes[1].set_ylabel(r'$S_\mathrm{Ca II}$')
    #axes[1].set_title("Fitting the data with GLS")
    #axes[1].legend()
    #plt.show()

    #harmonics and period quality flag
    harmonics_list = get_harmonic_list(sel_peaks)
    period_err = results["std_period_best"]
    flag = periodogram_flagging(harmonics_list, period_best, period_err, peaks_power, plevels, pmax)
    print("Period flag: ",flag)

    return results, gaps, flag, period_best, period_err, harmonics_list, VanderPlas_52_error, curve_fit_err, pyastronomy_error, gaussian_fit_err


instr = "HARPS"
stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
        'HD20794',"HD85512","HD192310"] 

#stars = ['HD209100', 'HD115617', 'HD1461', 'HD10647', 'HD85512', 'HD192310']
#stars = ["HD209100"]

VanderPlas_52_error_list = []; curve_fit_error_list = []; pyastronomy_error_list = []; gaussian_fit_err_list = []; flag_period_list = []
stars_list = []

for star in stars:
    print(f"{star} with {instr} data")
    folder_path = f"/home/telmo/PEEC-24-main/pipeline_products/{star}/{star}_{instr}/"
    file_path = folder_path+f"df_stats_{star}.fits"
    df, hdr = read_bintable(file_path,print_info=False)

    t_span = max(df["bjd"])-min(df["bjd"])
    n_spec = len(df)

    if n_spec >= 50 and t_span >= 2*365:  #only compute periodogram if star has at least 30 spectra in a time span of at least 2 years

        results, gaps, flag_period, period_best, period_err, harmonics_list, VanderPlas_52_error, curve_fit_err, pyastronomy_error, gaussian_fit_err = gls(star, instr, df["bjd"]-2450000, df["I_CaII"], y_err=df["I_CaII_err"], 
                                                                                                                                                           pmin=1.5, pmax=t_span, steps=1e6)
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='power', ascending=False)

        VanderPlas_52_error_list.append(VanderPlas_52_error)
        curve_fit_error_list.append(curve_fit_err)
        pyastronomy_error_list.append(pyastronomy_error)
        gaussian_fit_err_list.append(gaussian_fit_err)
        flag_period_list.append(flag_period)
        stars_list.append(star)

        print(f"{star}, {flag_period}: {period_best} +/- {gaussian_fit_err}")

    else: 
        period = None; period_err = None 

flag_ind_good = [i for i,flag in enumerate(flag_period_list) if flag in ["green","yellow"]]
#print(flag_period_list)

stars_good_list = [star for i,star in enumerate(stars_list) if flag_period_list[i] in ["green","yellow"]]
VanderPlas_52_error_list = np.array(VanderPlas_52_error_list)[flag_ind_good]
curve_fit_error_list = np.array(curve_fit_error_list)[flag_ind_good]
pyastronomy_error_list = np.array(pyastronomy_error_list)[flag_ind_good]
gaussian_fit_err_list = np.array(gaussian_fit_err_list)[flag_ind_good]

'''
correlation_coefficient_1 = np.corrcoef(VanderPlas_52_error_list, curve_fit_error_list)[0, 1]
print("Correlation coefficient between VanderPlas_52 and curve_fit:", correlation_coefficient_1)
correlation_coefficient_2 = np.corrcoef(VanderPlas_52_error_list, pyastronomy_error_list)[0, 1]
print("Correlation coefficient between VanderPlas_52 and PyAstronomy manual:", correlation_coefficient_2)
correlation_coefficient_3 = np.corrcoef(curve_fit_error_list, pyastronomy_error_list)[0, 1]
print("Correlation coefficient between Curve_fit and PyAstronomy manual:", correlation_coefficient_3)
correlation_coefficient_4 = np.corrcoef(gaussian_fit_err_list, curve_fit_error_list)[0, 1]
print("Correlation coefficient between Gaussian_fit and curve_fit:", correlation_coefficient_4)
correlation_coefficient_5 = np.corrcoef(gaussian_fit_err_list, pyastronomy_error_list)[0, 1]
print("Correlation coefficient between Gaussian_fit and PyAstronomy manual:", correlation_coefficient_5)
correlation_coefficient_6 = np.corrcoef(gaussian_fit_err_list, VanderPlas_52_error_list)[0, 1]
print("Correlation coefficient between Gaussian_fit and VanderPlas_52:", correlation_coefficient_6)
'''

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(13.5,7.5))
colors = ["blue","black","red","green","orange","purple"]
print(stars_good_list)
for i,col in enumerate(colors):
    ax[0,0].scatter(VanderPlas_52_error_list[i], curve_fit_error_list[i], color=col)
    ax[0,1].scatter(VanderPlas_52_error_list[i], pyastronomy_error_list[i],color=col)
    ax[0,2].scatter(curve_fit_error_list[i], pyastronomy_error_list[i],color=col)

    ax[1,0].scatter(gaussian_fit_err_list[i], curve_fit_error_list[i],color=col)
    ax[1,1].scatter(gaussian_fit_err_list[i], pyastronomy_error_list[i],color=col)
    ax[1,2].scatter(gaussian_fit_err_list[i], VanderPlas_52_error_list[i],color=col)

diag_line = ax[0,0].plot([0, 1], [0, 1], transform=ax[0,0].transAxes, ls="--", c=".3")
diag_line = ax[0,1].plot([0, 1], [0, 1], transform=ax[0,1].transAxes, ls="--", c=".3")
diag_line = ax[0,2].plot([0, 1], [0, 1], transform=ax[0,2].transAxes, ls="--", c=".3")

diag_line = ax[1,0].plot([0, 1], [0, 1], transform=ax[1,0].transAxes, ls="--", c=".3")
diag_line = ax[1,1].plot([0, 1], [0, 1], transform=ax[1,1].transAxes, ls="--", c=".3")
diag_line = ax[1,2].plot([0, 1], [0, 1], transform=ax[1,2].transAxes, ls="--", c=".3")

ax[0,0].set_xlim([0, 300]); ax[0,0].set_ylim([0, 300])
ax[0,1].set_xlim([0, 300]); ax[0,1].set_ylim([0, 300])
ax[0,2].set_xlim([0, 300]); ax[0,2].set_ylim([0, 300])
ax[1,0].set_xlim([0, 1500]); ax[1,0].set_ylim([0, 1500])
ax[1,1].set_xlim([0, 1500]); ax[1,1].set_ylim([0, 1500])
ax[1,2].set_xlim([0, 1500]); ax[1,2].set_ylim([0, 1500])


ax[0,0].set_xlabel("VanderPlas 52"); ax[0,0].set_ylabel("Curve Fit")
ax[0,1].set_xlabel("VanderPlas 52"); ax[0,1].set_ylabel("PyAstronomy GLS")
ax[0,2].set_xlabel("Curve Fit"); ax[0,2].set_ylabel("PyAstronomy GLS")

ax[1,0].set_xlabel("Gaussian Fit"); ax[1,0].set_ylabel("Curve Fit")
ax[1,1].set_xlabel("Gaussian Fit"); ax[1,1].set_ylabel("PyAstronomy GLS")
ax[1,2].set_xlabel("Gaussian Fit"); ax[1,2].set_ylabel("VanderPlas 52")

plt.suptitle("Comparison of period uncertainties in days by different methods")
plt.tight_layout()

plt.savefig("period_errors_comparison.pdf",bbox_inches="tight")

plt.show()