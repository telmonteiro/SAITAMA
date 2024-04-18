import pandas as pd, matplotlib.pyplot as plt, numpy as np, glob, os
from util_funcs import read_bintable

stars_list = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
        "HD20794","HD85512","HD192310"]

for star in stars_list:
    file_harps = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_HARPS/", f"df_stats_{star}.fits"))
    file_uves = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_UVES/", f"df_stats_{star}.fits"))
    file_espresso = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_ESPRESSO/", f"df_stats_{star}.fits"))

    df_harps, hdr_harps = read_bintable(file_harps[0],print_info=False)
    if file_uves != []:
        df_uves, hdr_uves = read_bintable(file_uves[0],print_info=False)
    if file_espresso != []:
        df_espresso, hdr_espresso = read_bintable(file_espresso[0],print_info=False)

    plt.figure(1,figsize=(8, 4))
    plt.suptitle(star, fontsize=14)
    plt.errorbar(df_harps.bjd - 2450000, df_harps.I_Ha06, df_harps.I_Ha06_err, fmt='k.',label="HARPS")
    plt.errorbar(df_uves.bjd - 2450000, df_uves.I_Ha06, df_uves.I_Ha06_err, fmt='r.',label="UVES")
    plt.errorbar(df_espresso.bjd - 2450000, df_espresso.I_Ha06, df_espresso.I_Ha06_err, fmt='b.',label="ESPRESSO")
    plt.legend()
    plt.ylabel("I_Ha06")
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.savefig(f"instruments_comparison/{star}_Ha06.pdf")
    plt.clf()

    plt.figure(2,figsize=(8, 4))
    plt.suptitle(star, fontsize=14)
    plt.errorbar(df_harps.bjd - 2450000, df_harps.I_CaII, df_harps.I_CaII_err, fmt='k.',label="HARPS")
    #plt.errorbar(df_uves.bjd - 2450000, df_uves.I_CaII, df_uves.I_CaII_err, fmt='r.',label="UVES")
    plt.errorbar(df_espresso.bjd - 2450000, df_espresso.I_CaII, df_espresso.I_CaII_err, fmt='b.',label="ESPRESSO")
    plt.legend()
    plt.ylabel("I_CaII")
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.savefig(f"instruments_comparison/{star}_CaII.pdf")

    plt.clf()