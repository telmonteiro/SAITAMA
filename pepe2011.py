from actin2.actin2 import ACTIN
import glob, os, numpy as np, matplotlib.pylab as plt, astropy, pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from util_funcs import _get_simbad_data, plot_RV_indices, stats_indice
'''
Data from https://archive.eso.org/eso/eso_archive_main.html (see print)
- This script runs ACTIN2 on the fits files of three stars from the paper Pepe et al 2011, to compare the log R_hk values from the paper and ACTIN2.
- It gets the SIMBAD data, plots RV and activity indices Calcium II, Halpha 0.6 Angstrom and Na I vs BJD, and then converts the I_CaII to log R_hk
- Finally it plots the two obtained log R_hk
The data was obtained manually from ESO archive, and then another script (average_spec_night.py) was made because in the paper they averaged the observations
by night. Unfortunately, for some reason the number of data points is not quite the same.
The results show good agreement for HD85512 and HD192310 but for HD20794 there's a displacement of around 0.02 log R_hk.
'''
actin = ACTIN()

stars_list = ["HD20794","HD85512","HD192310"] #pepe et al 2011 stars

fig, axs = plt.subplots(1, len(stars_list), figsize=(5.5 * len(stars_list), 3.8))

for i, star in enumerate(stars_list):

    root_directory = "/home/telmo/PEEC/ACTIN2/pepe_2011/{}/avg_spec/".format(star)
    #files_reduced = find_s1d_A(root_directory)
    #files1 = glob.glob(os.path.join("pepe_2011/{}/".format(star), "*s1d_A.fits")) #spectra used are all from HARPS
    #files = files_reduced + files1
    files = glob.glob(os.path.join("pepe_2011/{}/avg_spec/".format(star), "*s1d_A.fits"))
    if star == "HD192310":
        files.remove("pepe_2011/HD192310/avg_spec/average_2004-12-02_s1d_A.fits")

    #Information from SIMBAD
    result = _get_simbad_data(star=star, alerts=True)
    print(result)

    #run ACTIN2
    indices = ['I_CaII', 'I_Ha06', 'I_NaI']
    df = actin.run(files, indices)
    #crude sigma clipping, couldnt apply directly
    for n,x in enumerate(df.rv):
        mean = np.mean(df.rv)
        std = np.std(df.rv)
        if x < mean - (10 * std) or x > mean + (10 * std):
            df.drop(n,axis=0,inplace=True)

    #visualize RV and three different indices
    plot_RV_indices(star, df, indices)

    #get important stats for all indices
    stats_df = stats_indice(star,indices,df)
    print(stats_df)
    stats_df.drop(["star"], axis=1)
    stats_df.to_csv(f"stats_{star}.csv")

    #convert I_CaII to S_MW, get B-V from SIMBAD and convert to log R_hk
    from pyrhk.pyrhk import calc_smw, get_bv, calc_rhk
    smw, smw_err = calc_smw(caii=df["I_CaII"].values, caii_err=df["I_CaII_err"].values, instr='HARPS_GDS21')
    bv, bv_err, bv_ref = get_bv(star, alerts=True)
    log_rhk, log_rhk_err, rhk, rhk_err = calc_rhk(smw, smw_err, bv, method='middelkoop', evstage='MS')
    bjd = np.array(df.bjd)

    #get log R_hk of article Pepe et al 2011
    fits_table_pepe2011= astropy.io.fits.open('pepe_2011/{}/pepe2011_{}.fit'.format(star,star))
    data = fits_table_pepe2011[1].data
    log_rhk_pepe2011 = np.array([data[x][-2] for x in range(len(data))]) #extract column in numpy format
    log_rhk_err_pepe2011 = np.array([data[x][-1] for x in range(len(data))])
    BJD_pepe2011 = np.array([data[x][0] for x in range(len(data))])

    #visualize log R_hk from Pepe et al 2011 with obtained from ACTIN2
    axs[i].errorbar(bjd - 2450000, log_rhk, log_rhk_err, fmt='k.', label="ACTIN2, N = {}".format(len(log_rhk)))
    axs[i].errorbar(BJD_pepe2011 - 2450000, log_rhk_pepe2011, log_rhk_err_pepe2011, fmt='r.', label="Pepe at al, 2011, N = {}".format(len(log_rhk_pepe2011)))
    axs[i].set_ylabel("log R_hk")
    axs[i].set_xlabel("BJD $-$ 2450000 [days]")
    axs[i].set_title(star)
    axs[i].legend(loc="best")

plt.tight_layout()
plt.show()