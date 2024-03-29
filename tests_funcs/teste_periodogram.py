import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from astropy.io import fits
from util_funcs import read_bintable
from PyAstronomy.pyTiming import pyPeriod

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
        x_axis = 1/clp.freq
        plt.xlim([0, period+2000])
    elif mode == "Frequency":
        plt.xlabel("Frequency [1/days]")
        x_axis = clp.freq
    plt.ylabel("Power")
    plt.title(f"Power vs {mode} for GLS Periodogram")
    plt.plot(x_axis, clp.power, 'b-')
    # Add the FAP levels to the plot
    for i in range(len(fapLevels)):
        plt.plot([min(x_axis), max(x_axis)], [plevels[i]]*2, '--',
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

    return round(period,3), round(period_err,3)

target_save_name = "HD46375"
instr = "HARPS"
folder_path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/"
file_path = folder_path+f"df_stats_{target_save_name}.fits"

df, hdr = read_bintable(file_path,print_info=False)

t_span = max(df["bjd"])-min(df["bjd"])
n_spec = len(df)
if n_spec >= 20 and t_span >= 365:  #only compute periodogram if star has at least 20 spectra in a time span of at least 1 years

    period, period_err = gls_periodogram(target_save_name, df["I_CaII"],df["I_CaII_err"],df["bjd"], print_info = False,
                                         mode = "Period",
                                save=False, path_save=folder_path+f"{target_save_name}_GLS.png")
    print(f"Period of I_CaII: {period} +/- {period_err} days")

else: 
    period = None; period_err = None 

plt.show()
