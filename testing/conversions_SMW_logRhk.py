import glob, os, numpy as np, matplotlib.pylab as plt, astropy, pandas as pd, math
from astropy.io import fits
from general_funcs import (calc_fits_wv_1d, read_fits, plot_line, read_bintable)
from PyAstronomy import pyasl # type: ignore

from pyrhk.pyrhk import calc_smw, get_bv, calc_rhk, calc_prot_age

instr = "HARPS"

stars = ['HD209100']#, 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        # 'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536']

sweetcat = pd.read_csv("sweet_cat_stars.csv")

for star in stars:
    file_directory = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_{instr}/", f"df_stats_{star}.fits"))
    df_instr, hdr = read_bintable(file_directory[0], print_info=False)

    smw, smw_err = calc_smw(caii=df_instr["I_CaII"].values, caii_err=df_instr["I_CaII_err"].values, instr="HARPS_GDS21")
    bv, bv_err, bv_ref = get_bv(star, alerts=True)
    print(bv_ref)
    log_rhk, log_rhk_err, rhk, rhk_err = calc_rhk(smw, smw_err, bv, method="rutten", evstage='MS')

    bjd = df_instr["bjd"].values

    fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(7, 9))
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f"{star} activity indices for CaII H&K", fontsize=14, y=1)
    axes[0].errorbar(bjd - 2450000, df_instr["I_CaII"].values, df_instr["I_CaII_err"].values, fmt="k.")
    axes[0].set_xlabel("BJD - 2450000"); axes[0].set_ylabel(r"ACTIN $S_{CaII}$")
    axes[1].errorbar(bjd - 2450000, smw, smw_err, fmt="k.")
    axes[1].set_xlabel("BJD - 2450000"); axes[1].set_ylabel(r"$S_{MW}$")
    axes[2].errorbar(bjd - 2450000, log_rhk, log_rhk_err, fmt="k.")
    axes[2].set_xlabel("BJD - 2450000"); axes[2].set_ylabel(r"$\log R'_{HK}$")
    

    prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err = calc_prot_age(log_rhk, bv)
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12, 4.5))
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f"{star} rotational period and age from activity level", fontsize=14, y=1)
    axes[0].errorbar(prot_n84, prot_m08, xerr = prot_n84_err, yerr = prot_m08_err, fmt="k.")
    axes[0].set_xlabel(r"Chromospheric $P_{rot}$ via Noyes et al. (1984)")
    axes[0].set_ylabel(r"Chromospheric $P_{rot}$ Mamajek & Hillenbrand (2008)")
    if math.isnan(age_m08) == False:
        axes[1].errorbar(bjd - 2450000, age_m08, age_m08_err, fmt="k.")
        axes[1].set_xlabel("BJD - 2450000"); axes[1].set_ylabel("Gyrochronology age via Mamajek & Hillenbrand (2008)") 
    
    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(9, 4.5))
    fig.subplots_adjust(top=0.95)
    fig.suptitle(f"{star} rotational period via Mamajek & Hillenbrand (2008)", fontsize=14, y=1)
    axes.errorbar(bjd - 2450000, prot_m08, yerr = prot_m08_err, fmt="k.")
    axes.set_xlabel("BJD - 2450000")
    axes.set_ylabel(r"Chromospheric $P_{rot}$")

    plt.tight_layout()

    ind = sweetcat[sweetcat["hd"] == star[2:]].index[0]
    teff = sweetcat["Teff"][ind]
    feH = sweetcat["[Fe/H]"][ind]
    b = pyasl.BallesterosBV_T()
    bv_ballesteros = b.t2bv(T=teff) 
    print("B-V Ballesteros 2012: ", bv_ballesteros)

    r = pyasl.Ramirez2005()
    bv_ramirez = r.teffToColor(band="B-V", teff=teff, feH=feH)
    print("B-V Ramirez 2005: ", bv_ramirez)

    print("B-V Simbad:", bv)

#plt.show()


stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
         'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536']

bv_ballesteros_list = []; bv_ramirez_list = []; bv_simbad_list = []
for star in stars:
    bv, bv_err, bv_ref = get_bv(star, alerts=True)
    bv_simbad_list.append(bv)

    ind = sweetcat[sweetcat["hd"] == star[2:]].index[0]
    teff = sweetcat["Teff"][ind]
    feH = sweetcat["[Fe/H]"][ind]
    b = pyasl.BallesterosBV_T()
    bv_ballesteros = b.t2bv(T=teff) 
    #print("B-V Ballesteros 2012: ", bv_ballesteros)

    r = pyasl.Ramirez2005()
    bv_ramirez = r.teffToColor(band="B-V", teff=teff, feH=feH)
    #print("B-V Ramirez 2005: ", bv_ramirez)
    bv_ballesteros_list.append(bv_ballesteros)
    bv_ramirez_list.append(bv_ramirez)

fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(14, 4.5))
axes[0].scatter(bv_ballesteros_list,bv_ramirez_list)
diag_line, = axes[0].plot(axes[0].get_xlim(), axes[0].get_ylim(), ls="--", markersize=0.5, c='blue')
axes[0].set_ylabel("B-V Ramirez2005")
axes[0].set_xlabel("B-V Ballesteros")

axes[1].scatter(bv_ballesteros_list,bv_simbad_list)
diag_line, = axes[1].plot(axes[1].get_xlim(), axes[1].get_ylim(), ls="--", markersize=0.5, c='blue')
axes[1].set_ylabel("B-V Ballesteros")
axes[1].set_xlabel("B-V from SIMBAD")

axes[2].scatter(bv_ramirez_list,bv_simbad_list)
diag_line, = axes[2].plot(axes[2].get_xlim(), axes[2].get_ylim(), ls="--", markersize=0.5, c='blue')
axes[2].set_ylabel("B-V Ramirez2005")
axes[2].set_xlabel("B-V from SIMBAD")

fig.suptitle("B-V using 3 different methods")

plt.tight_layout()
plt.show()
