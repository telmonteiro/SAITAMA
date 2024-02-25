from actin2.actin2 import ACTIN
import glob, os, numpy as np, matplotlib.pylab as plt, astropy, pandas as pd
from astropy.io import fits
from util_funcs import _get_simbad_data, plot_RV_indices, sigma_clip, stats_indice, read_fits
from pyrhk.pyrhk import calc_smw, get_bv, calc_rhk

actin = ACTIN()

stars_list = ["HD20794","HD85512","HD192310"] #pepe et al 2011 stars

#plot every combination
combinations = [["HARPS_L11", "middelkoop"], ["HARPS_L11", "rutten"], ["HARPS_GDS21", "middelkoop"], ["HARPS_GDS21", "rutten"]]
row_titles = [comb[0] + " and " + comb[1] for comb in combinations]
fig, axs = plt.subplots(4, len(stars_list), figsize=(16, 14), gridspec_kw={'height_ratios': [0.95, 0.95, 0.95, 0.95], 'hspace': 0.65})
fig.suptitle(r"$\log R'_{HK}$ for three stars from Pepe et al. 2011 with different calibrations", y=0.95, fontsize=15)

#plot only the best combination
fig1, axs1 = plt.subplots(1, len(stars_list), figsize=(5.5 * len(stars_list), 4))
fig1.suptitle(r"$\log R'_{HK}$ for three stars from Pepe et al. 2011 with HARPS_L11 + middelkoop calibrations", fontsize=12)

for k, row_title in enumerate(row_titles):
    fig.text(0.5, 0.905 - k * 0.207, row_title, va='center', ha='center', rotation='horizontal', fontsize=13)


for i, star in enumerate(stars_list):
    root_directory = f"/home/telmo/PEEC/ACTIN2/pepe_2011/{star}/avg_spec_rv_corr/"

    files = glob.glob(os.path.join(f"pepe_2011/{star}/avg_spec_rv_corr/", "*s1d_A.fits"))

    result = _get_simbad_data(star=star, alerts=False)
    #print(result)

    #run ACTIN2
    indices= ['I_CaII', 'I_Ha06', 'I_NaI']

    df = actin.run(files, indices, spec_kw=dict(spec_file_in="test_instrument.py"))
    #spec, hdr = read_fits(files[0])
    #read_spec = actin.ReadSpec(files[0],spec_file_in="test_instrument.py")
    #wave = read_spec.spectrum['wave']
    #flux = read_spec.spectrum['flux']
    #wave = spec[0]; flux = spec[1]
    #spectrum = dict(wave=wave, flux=flux)
    #headers = dict()
    #df = actin.CalcIndices(spectrum, headers, indices).indices
    print(df)

    df.to_csv(f"pepe_2011/results/rv_corr/df_{star}.csv")
    
    cols = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
    df = sigma_clip(df, cols, sigma=3.5)

    plot_RV_indices(star, df, indices, save=False)
    stats_df = stats_indice(star,cols,df)
    print(stats_df)
    stats_df.to_csv(f"pepe_2011/results/rv_corr/stats_{star}.csv")
   
    for k,comb in enumerate(combinations):
        #convert I_CaII to S_MW, get B-V from SIMBAD and convert to log R_hk
        smw, smw_err = calc_smw(caii=df["I_CaII"].values, caii_err=df["I_CaII_err"].values, instr=comb[0])
        bv, bv_err, bv_ref = get_bv(star, alerts=True)
        log_rhk, log_rhk_err, rhk, rhk_err = calc_rhk(smw, smw_err, bv, method=comb[1], evstage='MS')
        bjd = np.around(np.array(df.bjd))

        #get log R_hk of article Pepe et al 2011
        fits_table_pepe2011= fits.open(f'pepe_2011/{star}/pepe2011_{star}.fit')
        data_pepe = fits_table_pepe2011[1].data
        log_rhk_pepe2011 = np.array([data_pepe[x][-2] for x in range(len(data_pepe))]) #extract column in numpy format
        log_rhk_err_pepe2011 = np.array([data_pepe[x][-1] for x in range(len(data_pepe))])
        BJD_pepe2011 = np.array([round(data_pepe[x][0]) for x in range(len(data_pepe))])

        df_pepe = pd.DataFrame({"log_rhk":log_rhk_pepe2011,"log_rhk_err":log_rhk_err_pepe2011,"bjd":BJD_pepe2011})
        df_pepe = sigma_clip(df_pepe, cols=df_pepe.columns, sigma=6.5)

        #visualize log R_hk from Pepe et al 2011 with obtained from ACTIN2
        axs[k, i].errorbar(bjd - 2450000, log_rhk, log_rhk_err, fmt='k.',
                               label=f"ACTIN2, N = {len(log_rhk)}")
        axs[k, i].errorbar(df_pepe["bjd"] - 2450000, df_pepe["log_rhk"], df_pepe["log_rhk_err"], fmt='r.',
                               label=f"Pepe et al, 2011, N = {len(df_pepe['log_rhk'])}")
        axs[k, i].set_ylabel("log R_hk")
        axs[k, i].set_xlabel("BJD $-$ 2450000 [days]")
        axs[k, i].set_title(star)
        axs[k, i].legend(loc="best",fontsize=9)

        if k == 0: #for the best combination
            axs1[i].errorbar(bjd - 2450000, log_rhk, log_rhk_err, fmt='k.', 
                             label="ACTIN2, N = {}".format(len(log_rhk)))
            axs1[i].errorbar(df_pepe["bjd"] - 2450000, df_pepe["log_rhk"], df_pepe["log_rhk_err"], fmt='r.', 
                             label="Pepe at al, 2011, N = {}".format(len(df_pepe['log_rhk'])))
            axs1[i].set_ylabel("log R_hk")
            axs1[i].set_xlabel("BJD $-$ 2450000 [days]")
            axs1[i].set_title(star)
            axs1[i].legend(loc="best")
   
fig.savefig("pepe_2011/results/rv_corr/log_Rhk_ACTIN_pepe2011_all_cal.png", bbox_inches="tight")
fig1.savefig("pepe_2011/results/rv_corr/log_Rhk_ACTIN_pepe2011_best.png", bbox_inches="tight")
plt.tight_layout()
plt.show()