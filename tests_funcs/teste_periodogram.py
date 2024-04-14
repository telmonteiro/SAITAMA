import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from astropy.io import fits
from util_funcs import read_bintable
from PyAstronomy.pyTiming import pyPeriod

def are_harmonics(period1, period2, tolerance=0.1):
    ratio = period1 / period2
    # Check if the ratio is close to an integer or simple fraction
    if abs(ratio - round(ratio)) < tolerance:
        return True
    else:
        return False

def periodogram_flagging(harmonics_list, period, period_err, power_list, plevels):
    '''
    Green/4: error < 10% and no harmonics in 3 periods with most power
    Yellow/3: 10% < error < 20% and no harmonics in 3 periods with most power
    Orange/2: harmonics in 3 periods with most power or no error
    Red/1: many periods with power close to each under: if number of periods over 80% of the max > 3
    Black/0: discarded, error > 20%, period under 1 yr or over 100 yrs, below FAP 1% level
    '''
    error = period_err/period * 100
    powers_close_max = [n for n in power_list if n > 0.8*np.max(power_list)]

    if error > 20 or np.max(power_list) < plevels[-1] or period < 365 or period > 100*365:
        flag = "black"
    else:
        if error <= 10 and len(harmonics_list) == 0:
            flag = "green"
        elif 10 < error <= 20 and len(harmonics_list) == 0:
            flag = "yellow"
        elif len(harmonics_list) > 0 or period_err == 0.0:
            flag = "orange"
        elif len(powers_close_max) > 3:
            flag = "red"

    return flag
        

def gls_periodogram(star, I_CaII, I_CaII_err, bjd, print_info, mode, save, path_save):
    # Compute the GLS periodogram with default options. Choose Zechmeister-Kuerster normalization explicitly
    clp = pyPeriod.Gls((bjd - 2450000, I_CaII, I_CaII_err), norm="ZK", verbose=print_info,ofac=30)
    dic_clp = clp.info(noprint=True)
    period = dic_clp["best_sine_period"]
    period_err = dic_clp["best_sine_period_err"]
    # Define FAP levels of 10%, 5%, and 1%
    fapLevels = np.array([0.1, 0.05, 0.01])
    # Obtain the associated power thresholds
    plevels = clp.powerLevel(fapLevels)

    plt.figure(1, figsize=(13, 4))
    plt.suptitle(f"GLS periodogram for {star} I_CaII", fontsize=12)

    # and plot power vs. frequency.
    plt.subplot(1, 2, 1)
    if mode == "Period":
        plt.xlabel("Period [days]")
        period_list = 1/clp.freq

        array_descending = np.argsort(clp.power)[-3:]
        top_3_period = period_list[array_descending]
        harmonics_list = []
        for i in range(len(top_3_period)):
            for j in range(i+1, len(top_3_period)):
                if are_harmonics(top_3_period[j], top_3_period[i], tolerance=0.01):
                    print(f"Period {top_3_period[i]} and {top_3_period[j]} are harmonics of each other")
                    harmonics_list.append([top_3_period[i], top_3_period[j]])
       
        flag = periodogram_flagging(harmonics_list, period, period_err, clp.power, plevels)
        print("Flag: ",flag)
        
        plt.plot(period_list, clp.power, 'b-')
        plt.xlim([0, period+2000])

    elif mode == "Frequency":
        plt.xlabel("Frequency [1/days]")
        freq_list = clp.freq
        plt.plot(freq_list, clp.power, 'b-')

    plt.ylabel("Power")
    plt.title(f"Power vs {mode} for GLS Periodogram")
    
    # Add the FAP levels to the plot
    for i in range(len(fapLevels)):
        plt.plot([min(period_list), max(period_list)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fapLevels[i]*100))
    plt.legend()

    plt.subplot(1, 2, 2)
    timefit = np.linspace(min(bjd - 2450000),max(bjd - 2450000),1000)
    plt.plot(timefit,clp.sinmod(timefit),label="fit")
    plt.errorbar(bjd - 2450000,I_CaII,yerr=I_CaII_err,fmt='.', color='k',label='data')
    plt.xlabel('BJD $-$ 2450000 [days]'); plt.ylabel(r'$S_\mathrm{CaII}$')
    plt.title("Fitting the data with GLS")
    plt.legend()

    plt.subplots_adjust(top=0.85)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

    return round(period,3), round(period_err,3), flag

target_save_name = "HD1461"
instr = "HARPS"
folder_path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/"
file_path = folder_path+f"df_stats_{target_save_name}.fits"

df, hdr = read_bintable(file_path,print_info=False)

t_span = max(df["bjd"])-min(df["bjd"])
n_spec = len(df)
if n_spec >= 30 and t_span >= 365:  #only compute periodogram if star has at least 20 spectra in a time span of at least 1 years

    period, period_err, flag_period = gls_periodogram(target_save_name, df["I_CaII"],df["I_CaII_err"],df["bjd"], 
                                         print_info = False, mode = "Period",
                                save=False, path_save=folder_path+f"{target_save_name}_GLS.png")
    print(f"Period of I_CaII: {period} +/- {period_err} days")

else: 
    period = None; period_err = None 

plt.show()
