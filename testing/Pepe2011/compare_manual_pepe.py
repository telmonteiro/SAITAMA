import matplotlib.pyplot as plt, pandas as pd, numpy as np
from util_funcs import sigma_clip
from astropy.io import fits
from pyrhk.pyrhk import calc_smw, get_bv, calc_rhk

stars_list = ["HD20794","HD85512","HD192310"] #pepe et al 2011 stars

fig1, axs1 = plt.subplots(1, len(stars_list), figsize=(5.5 * len(stars_list), 4.4))
fig1.suptitle(r"$\log R'_{HK}$ for three stars from Pepe et al. 2011", fontsize=11)

for i,star in enumerate(stars_list):

    df_manual = pd.read_csv(f"pepe_2011/results/rv_corr/df_{star}.csv")
    #df_automatic = pd.read_csv(f"pepe_2011/results/df_{star}.csv")

    #df_automatic = sigma_clip(df_automatic, cols=["I_CaII"], sigma=3.5)
    if star == "HD192310":
        sigma = 3.5
    else: sigma = 3.5
    df_manual = sigma_clip(df_manual, cols=["I_CaII"], sigma=sigma)

    '''
    axs1[i].errorbar(df_manual["bjd"] - 2450000, df_manual["I_CaII"], df_manual["I_CaII_err"], fmt='ko', 
                        label="manual RV correction, N = {}".format(len(df_manual["I_CaII"])))
    axs1[i].errorbar(df_automatic["bjd"] - 2450000, df_automatic["I_CaII"], df_automatic["I_CaII_err"], fmt='r.', 
                        label="using CCF from literature, N = {}".format(len(df_automatic["I_CaII"])))
    '''
    smw, smw_err = calc_smw(caii=df_manual["I_CaII"].values, caii_err=df_manual["I_CaII_err"].values, instr="HARPS_GDS21")
    bv, bv_err, bv_ref = get_bv(star, alerts=False)
    log_rhk, log_rhk_err, rhk, rhk_err = calc_rhk(smw, smw_err, bv, method="rutten", evstage='MS')
    bjd = np.around(np.array(df_manual.bjd))

    fits_table_pepe2011= fits.open(f'pepe_2011/{star}/pepe2011_{star}.fit')
    data_pepe = fits_table_pepe2011[1].data
    log_rhk_pepe2011 = np.array([data_pepe[x][-2] for x in range(len(data_pepe))]) #extract column in numpy format
    log_rhk_err_pepe2011 = np.array([data_pepe[x][-1] for x in range(len(data_pepe))])
    bjd_pepe2011 = np.array([data_pepe[x][0] for x in range(len(data_pepe))])

    bjd_pepe2011 = np.around(bjd_pepe2011 - 2450000) 
    bjd = np.around(bjd - 2450000)
    
    #bjd = np.sort(bjd)
    #bjd_pepe2011 = np.sort(bjd_pepe2011)
    #axs1[i].scatter(bjd_pepe2011, bjd, label=f"N = {len(bjd)}")
    
    #axs1[i].errorbar(log_rhk_pepe2011, log_rhk, xerr = log_rhk_err_pepe2011, yerr = log_rhk_err, fmt='k.', 
    #                         label="N = {}".format(len(log_rhk_pepe2011)))
        
    # Set a tolerance for comparing floating-point numbers
    tolerance = 1e-6

    # Initialize empty lists to store common data
    common_bjd_values = []
    log_rhk_common_values = []; log_rhk_err_common_values = []
    log_rhk_pepe2011_common_values = []; log_rhk_pepe2011_err_common_values = []

    # Iterate through each element in bjd and check if it exists in bjd_pepe2011
    for bjd_value, log_rhk_value in zip(bjd, log_rhk):
        # Check if there's a matching BJD value in bjd_pepe2011
        matching_indices = np.where(np.isclose(bjd_pepe2011, bjd_value, atol=tolerance))[0]
        
        # If a match is found, store the data
        if len(matching_indices) > 0:
            for matching_index in matching_indices:
                common_bjd_values.append(bjd_value)
                log_rhk_common_values.append(log_rhk_value)
                log_rhk_pepe2011_common_values.append(log_rhk_pepe2011[matching_index])
                log_rhk_err_common_values.append(log_rhk_err[matching_index])
                log_rhk_pepe2011_err_common_values.append(log_rhk_err_pepe2011[matching_index])

    # Convert lists to arrays for plotting
    common_bjd_values = np.array(common_bjd_values)
    log_rhk_common_values = np.array(log_rhk_common_values)
    log_rhk_pepe2011_common_values = np.array(log_rhk_pepe2011_common_values)

    # Plot error bars
    axs1[i].errorbar(log_rhk_pepe2011_common_values, log_rhk_common_values, xerr=log_rhk_pepe2011_err_common_values, yerr=log_rhk_err_common_values, fmt='k.',
                     label="N = {}".format(log_rhk_pepe2011_common_values.shape[0]))
   
    #axs1[i].set_ylabel("I_CaII")
    #axs1[i].set_xlabel("BJD $-$ 2450000 [days]")
    diag_line, = axs1[i].plot(axs1[i].get_xlim(), axs1[i].get_ylim(), ls="--", markersize=0.5, c='blue')
    axs1[i].set_ylabel("log R_hk - ACTIN2 'manual' RV correction")
    axs1[i].set_xlabel("log R_hk - Pepe et al, 2011")
    axs1[i].set_title(star)
    axs1[i].legend(loc="best")


fig1.savefig("pepe_2011/results/rv_corr/log_Rhk_ACTIN_pepe2011.png", bbox_inches="tight")
plt.show()

