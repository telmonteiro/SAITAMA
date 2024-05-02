import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from astropy.io import fits
from util_funcs import read_bintable
from PyAstronomy.pyTiming import pyPeriod
from report_generator import get_report_periodogram

def are_harmonics(period1, period2, tolerance=0.01):
    ratio = period1 / period2
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False
    
def get_harmonic_list(period_list, power, plevels):
    array_descending = np.argsort(power[(power >= plevels[1])])[-3:]
    top_5_period = period_list[array_descending]
    harmonics_list = []
    for i in range(len(top_5_period)):
        for j in range(i+1, len(top_5_period)):
            if are_harmonics(top_5_period[j], top_5_period[i], tolerance=0.01):
                print(f"Period {top_5_period[i]} and {top_5_period[j]} are harmonics of each other")
                harmonics_list.append([top_5_period[i], top_5_period[j]])
    return harmonics_list

def periodogram_flagging(harmonics_list, period, period_err, power_list, plevels, t_span):
    '''
    Green/4: error < 10% and no harmonics in 3 periods with most power
    Yellow/3: 10% < error < 20% and no harmonics in 3 periods with most power
    Orange/2: harmonics in 3 periods with most power, no error or error > 20%
    Red/1: many periods with power close to each under: if number of periods over 80% of the max > 3
           or if the period obtained is bigger than the time span
    Black/0: discarded - period under 1 yr or over 100 yrs, below FAP 1% level
    '''
    error = period_err/period * 100
    powers_close_max = [n for n in power_list if n > 0.8*np.max(power_list)]

    if np.max(power_list) < plevels[-1] or period < 365 or period > 100*365:
        flag = "black"
    else:
        if error <= 10 and len(harmonics_list) == 0 and period < t_span:
            flag = "green"
        elif 10 < error <= 20 and len(harmonics_list) == 0 and period < t_span:
            flag = "yellow"
        elif (len(harmonics_list) > 0 or period_err == 0.0 or error > 20) and period < t_span:
            flag = "orange"
        elif len(powers_close_max) > 3 or period > t_span:
            flag = "red"

    return flag

def WF_periodogram(star, bjd, print_info, save, path_save):
    time_start = np.min(bjd); time_end = np.max(bjd)
    time_array = np.linspace(time_start, time_end, int(time_end - time_start))
    zeros_array = np.zeros_like(time_array); ones_array = np.ones_like(bjd)
    time = np.concatenate((time_array, bjd))
    ones_zeros = np.concatenate((zeros_array, ones_array))

    print(f"BJD array has {bjd.shape[0]} and ones_zeros has {len([x for x in ones_zeros if x == 1])} 1s")

    clp_WF = pyPeriod.Gls((time, ones_zeros), norm="ZK", verbose=print_info,ofac=30)
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
    plt.xlim([0, period_WF+15000])
    plt.title(f"Power vs Period for GLS Periodogram {star} Window Function")
    
    for i in range(len(fapLevels)): # Add the FAP levels to the plot
        plt.plot([min(period_list_WF), max(period_list_WF)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fapLevels[i]*100))
    plt.legend()

    if save == True: plt.savefig(path_save, bbox_inches="tight")
    plt.clf()

    return round(period_WF,3), round(period_err_WF,3)

def gaps_time(BJD):
    '''Takes the BJD array and returns the 5 biggest gaps in time.'''
    time_sorted = BJD[np.argsort(BJD)] #sorting time
    gaps = np.diff(time_sorted)
    gaps = gaps[np.argsort(gaps)][-5:]
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
    plt.xlim([0, period+5000])
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
    plt.clf()

    return round(period,3), round(period_err,3), flag, harmonics_list

#star = "HD20794"
instr = "HARPS"
stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
        'HD20794',"HD85512","HD192310"] 

for star in stars:
    print(f"{star} with {instr} data")
    folder_path = f"teste_download_rv_corr/{star}/{star}_{instr}/"
    file_path = folder_path+f"df_stats_{star}.fits"
    df, hdr = read_bintable(file_path,print_info=False)

    t_span = max(df["bjd"])-min(df["bjd"])
    n_spec = len(df)
    if n_spec >= 30 and t_span >= 2*365:  #only compute periodogram if star has at least 30 spectra in a time span of at least 2 years
        gaps = gaps_time(df["bjd"])
        print("Gaps in BJD:", gaps)
        period, period_err, flag_period, harmonics_list = gls_periodogram(star, df["I_CaII"],df["I_CaII_err"],df["bjd"], 
                                            print_info = False, save=True, path_save=folder_path+f"{star}_GLS.png")
        print(f"Period of I_CaII: {period} +/- {period_err} days")

        period_WF, period_err_WF = WF_periodogram(star, df["bjd"]-2450000, print_info=False, 
                                                save=True, path_save=folder_path+f"{star}_WF.png")
        print(f"Period of WF: {period_WF} +/- {period_err_WF} days")

        report_periodogram = get_report_periodogram(hdr,gaps,period,period_err,flag_period,harmonics_list,period_WF,period_err_WF,folder_path=folder_path)
        #print(report_periodogram)
    else: 
        period = None; period_err = None 

#plt.show()
