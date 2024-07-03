import numpy as np, pandas as pd, matplotlib.pylab as plt, glob, os
from general_funcs import read_bintable, read_fits
from astropy.io import fits
from actin2 import ACTIN # type: ignore

actin = ACTIN()

stars = ["HD16417"]
instr = "UVES"
indices = ["I_Ha06"]

for star in stars:
    print(f"{star} with {instr} data")
    folder_path = f"teste_download_rv_corr/{star}/{star}_{instr}/ADP/"
    files = glob.glob(os.path.join(folder_path, "ADP*.fits"))
    
    flux_R1_R2 = np.zeros((len(files)))
    flux_R1_R2_err = np.zeros((len(files)))
    I_Ha06 = np.zeros((len(files)))
    I_Ha06_err = np.zeros((len(files)))
    for i,file in enumerate(files):
        #print(file)
        wv, f, f_err, hdr = read_fits(file,instr,mode="rv_corrected")
        #plt.figure(1)
        #plt.plot(wv,f)

        headers = {"file": file,"instr": instr}
        spectrum = dict(wave=wv, flux=f, flux_err=f_err)
        df_ind = actin.CalcIndices(spectrum, headers, indices,full_output=True).indices

        flux_R1_R2[i] = df_ind["HaR1_F"]/df_ind["HaR2_F"]
        flux_R1_R2_err[i] = flux_R1_R2[i] * np.sqrt((df_ind["HaR1_F_err"]/df_ind["HaR1_F"])**2 + (df_ind["HaR2_F_err"]/df_ind["HaR2_F"])**2)
        I_Ha06[i] = df_ind["I_Ha06"]
        I_Ha06_err[i] = df_ind["I_Ha06_err"]

    plt.figure(1)
    plt.errorbar(I_Ha06, flux_R1_R2, yerr=flux_R1_R2_err, xerr=I_Ha06_err, fmt="k.")
    plt.xlabel("I_Ha06")
    plt.ylabel("Flux R1/R2")

plt.tight_layout()
plt.show()