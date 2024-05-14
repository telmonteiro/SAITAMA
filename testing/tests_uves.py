import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, glob
from general_funcs import read_bintable, plot_line, read_fits
from astropy.io import fits

def actin_manual_Ha06(wv, flux):
    '''Manual rough estimate of I_Halpha by taking the reference and activity lines and computing the ratio between
    the mean of the activity line (L1) and the reference lines (R1 and R2) for the Halpha at 0.6 Angstrom'''
    wc_R1 = 6550.870; bandwith_R1 = 10.75
    wv_R1 = np.where((wc_R1-bandwith_R1/2 <= wv) & (wv <= wc_R1+bandwith_R1/2))
    flux_R1 = flux[wv_R1]
    mean_flux_R1 = np.mean(flux_R1)
    
    wc_R2 = 6580.310; bandwith_R2 = 8.75
    wv_R2 = np.where((wc_R2-bandwith_R2/2 <= wv) & (wv <= wc_R2+bandwith_R2/2))
    flux_R2 = flux[wv_R2]
    mean_flux_R2 = np.mean(flux_R2)   

    wc_L1 = 6562.808; bandwith_L1 = 0.6
    wv_L1 = np.where((wc_L1-bandwith_L1/2 <= wv) & (wv <= wc_L1+bandwith_L1/2))
    flux_L1 = flux[wv_L1]
    mean_flux_L1 = np.mean(flux_L1)  
    
    return mean_flux_L1/(mean_flux_R1 + mean_flux_R2)


def get_keys_spec(file):
    '''Getting spectral resolution and climate conditions (mean air mass, wind speed, relative humidty and temperature)'''
    spec = fits.open(file)
    spec_hdr = spec[0].header
    spectral_resolution = spec_hdr["SPEC_RES"]
    climate_conditions = {"Mean airmass":(spec_hdr["HIERARCH ESO TEL AIRM START"]+spec_hdr["HIERARCH ESO TEL AIRM END"])/2,
                          "Wind speed":spec_hdr["HIERARCH ESO TEL AMBI WINDSP"],
                          "Relative humidity":spec_hdr["HIERARCH ESO TEL AMBI RHUM"],
                          "Ambiente temperature":spec_hdr["HIERARCH ESO TEL AMBI TEMP"]}
    return spectral_resolution, climate_conditions

def add_columns_to_df(df):
    '''Add the spec_res and climate conditions columns to df'''
    spec_res = np.zeros((len(df["file"])))
    temperature = np.zeros((len(df["file"])))
    wind = np.zeros((len(df["file"])))
    humidity = np.zeros((len(df["file"]))) 
    airmass = np.zeros((len(df["file"]))) 
    for i,file in enumerate(df["file"]):
        spectral_resolution, climate_conditions = get_keys_spec(file)
        spec_res[i] = spectral_resolution
        temperature[i] = climate_conditions["Ambiente temperature"]
        wind[i] = climate_conditions["Wind speed"]
        humidity[i] = climate_conditions["Relative humidity"]
        airmass[i] = climate_conditions["Mean airmass"]

    df.loc[:, "spec_res"] = spec_res
    df.loc[:, "wind_speed"] = wind
    df.loc[:, "amb_temp"] = temperature
    df.loc[:, "rel_humidity"] = humidity
    df.loc[:, "mean_airmass"] = airmass

    return df

stars_dic = {
    'HD209100': {'up': 0, 'down': 0.14},
    'HD160691': {'up': 0.1012, 'down': 1},
    'HD115617': {'up': 0.106, 'down': 1},
    'HD46375': {'up': 1, 'down': 0},  # no points
    'HD22049': {'up': 1, 'down': 0},
    'HD102365': {'up': 0, 'down': 0.1},
    'HD1461': {'up': 0.103, 'down': 1},
    'HD16417': {'up': 0.103, 'down': 1},
    'HD10647': {'up': 1, 'down': 0},
    'HD13445': {'up': 1, 'down': 0},
    'HD142A': {'up': 1, 'down': 0},
    'HD108147': {'up': 1, 'down': 0},
    'HD16141': {'up': 0.101, 'down': 1},
    'HD179949': {'up': 1, 'down': 0},
    'HD47536': {'up': 1, 'down': 0},
    'HD20794': {'up': 0.11, 'down': 1},
    'HD85512': {'up': 1, 'down': 0},
    'HD192310': {'up': 0.125, 'down': 1}
}

stars = stars_dic.keys()
stars = ["HD16417"]
for star in stars:
    file_uves = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_UVES/", f"df_stats_{star}.fits"))
    if file_uves == []:
        continue
    df_uves, hdr = read_bintable(file_uves[0], print_info=False)

    df_uves = add_columns_to_df(df_uves)

    file_harps = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_HARPS/", f"df_stats_{star}.fits"))
    df_harps, hdr_harps = read_bintable(file_harps[0], print_info=False)

    df_harps = add_columns_to_df(df_harps)

    outlier_up = stars_dic[star]['up']
    outlier_down = stars_dic[star]['down']
    mask = (df_uves['I_Ha06'] < outlier_down) & (df_uves['I_Ha06'] > outlier_up)

    fig = plt.figure(2,figsize=(16, 8), constrained_layout=True)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    fig.suptitle(f"{star} climate conditions", fontsize=14)

    ax1.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'wind_speed'], label=f'I_Ha06 >= {outlier_up}')
    ax1.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'wind_speed'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax1.scatter(df_harps['bjd'] - 2450000, df_harps['wind_speed'], color='black', label='HARPS')
    ax1.set_xlabel("BJD $-$ 2450000 [days]"); ax1.set_ylabel("Wind speed (m/s)"); ax1.legend()

    ax2.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'amb_temp'], label=f'I_Ha06 >= {outlier_up}')
    ax2.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'amb_temp'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax2.scatter(df_harps['bjd'] - 2450000, df_harps['amb_temp'], color='black', label='HARPS')
    ax2.set_xlabel("BJD $-$ 2450000 [days]"); ax2.set_ylabel("Ambient Temperature (ºC)"); ax2.legend()

    ax3.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'rel_humidity'], label=f'I_Ha06 >= {outlier_up}')
    ax3.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'rel_humidity'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax3.scatter(df_harps['bjd'] - 2450000, df_harps['rel_humidity'], color='black', label='HARPS')
    ax3.set_xlabel("BJD $-$ 2450000 [days]"); ax3.set_ylabel("Relative Humidity (%)"); ax2.legend()

    ax4.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'mean_airmass'], label=f'I_Ha06 >= {outlier_up}')
    ax4.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'mean_airmass'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax4.scatter(df_harps['bjd'] - 2450000, df_harps['mean_airmass'], color='black', label='HARPS')
    ax4.set_xlabel("BJD $-$ 2450000 [days]"); ax4.set_ylabel("Mean Airmass"); ax4.legend()
    plt.tight_layout()
    #plt.savefig(f"uves_tests_fig/{star}_meteorology.pdf",overwrite=True, format = 'pdf', dpi=300)

    fig = plt.figure(1,figsize=(16, 8), constrained_layout=True)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    fig.suptitle(star, fontsize=14)

    print(df_uves[["RV_flag","I_Ha06","file","rv","spec_res"]])

    ax1.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'SNR'], label=f'I_Ha06 >= {outlier_up}')
    ax1.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'SNR'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax1.set_xlabel("BJD $-$ 2450000 [days]"); ax1.set_ylabel("SNR"); ax1.legend()

    ax2.scatter(df_uves.loc[~mask, 'SNR'], df_uves.loc[~mask, 'rv'], label=f'I_Ha06 >= {outlier_up}')
    ax2.scatter(df_uves.loc[mask, 'SNR'], df_uves.loc[mask, 'rv'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax2.set_xlabel("SNR"); ax2.set_ylabel("rv"); ax2.legend()

    ax3.errorbar(df_uves.loc[~mask, 'SNR'], df_uves.loc[~mask, 'I_Ha06'], df_uves.loc[~mask, 'I_Ha06_err'], fmt='b.', label=f'I_Ha06 >= {outlier_up}')
    ax3.errorbar(df_uves.loc[mask, 'SNR'], df_uves.loc[mask, 'I_Ha06'], df_uves.loc[mask, 'I_Ha06_err'], fmt='r.', label=f'I_Ha06 < {outlier_down}')
    ax3.set_xlabel("SNR"); ax3.set_ylabel("I_Ha06"); ax3.legend()

    ax4.errorbar(df_harps['bjd'] - 2450000, df_harps['I_Ha06'], df_harps['I_Ha06_err'], fmt='k.', label="HARPS")
    ax4.errorbar(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'I_Ha06'], df_uves.loc[~mask, 'I_Ha06_err'], fmt='b.', label='UVES')
    ax4.errorbar(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'I_Ha06'], df_uves.loc[mask, 'I_Ha06_err'], fmt='r.',label="UVES outlier")
    ax4.legend()
    ax4.set_ylabel("I_Ha06"); ax4.set_xlabel("BJD $-$ 2450000 [days]")

    ax5.scatter(df_uves.loc[~mask, 'bjd'] - 2450000, df_uves.loc[~mask, 'spec_res'], label=f'I_Ha06 >= {outlier_up}')
    ax5.scatter(df_uves.loc[mask, 'bjd'] - 2450000, df_uves.loc[mask, 'spec_res'], color='red', label=f'I_Ha06 < {outlier_down}')
    ax5.scatter(df_harps['bjd'] - 2450000, df_harps['spec_res'], color='black',label="HARPS")
    ax5.set_xlabel("BJD $-$ 2450000 [days]"); ax5.set_ylabel("Spectral Resolution"); ax5.legend()

    #outlier_point = df_uves.loc[mask, ['spec_res',"file","bjd"]].iloc[:1]
    #non_outlier_low_res = df_uves.loc[~mask, ['spec_res',"file","bjd"]].iloc[4:5]
    #non_outlier_high_res = df_uves.loc[~mask, ['spec_res',"file","bjd"]].iloc[:1]
    #harps_point = df_harps[['spec_res',"file","bjd"]].iloc[:1]

    outlier_point = df_uves.loc[mask, ['spec_res',"file","bjd"]].iloc[:1]
    non_outlier_high_res = df_uves.loc[~mask, ['spec_res',"file","bjd"]].iloc[:1]
    harps_point = df_harps[['spec_res',"file","bjd"]].iloc[:1]
    list_points = [outlier_point,non_outlier_high_res,harps_point]
    list_labels = ["outlier","non-outlier","HARPS"]

    plt.figure(3, figsize=(12,8))
    instr = "UVES"
    #labels_HD115617 = [f"Outlier, Res: {outlier_point['spec_res'].values[0]}", f"Non-outlier, Res: {non_outlier_low_res['spec_res'].values[0]}", 
    #          f"Non-outlier, Res: {non_outlier_high_res['spec_res'].values[0]}", f"HARPS, Res: {harps_point['spec_res'].values[0]}"]
    #labels_HD16141 = [f"Outlier, Res: {outlier_point['spec_res'].values[0]}, BJD: {round(outlier_point['bjd'].values[0]) - 2450000}", 
    #          f"Non-outlier, Res: {non_outlier_high_res['spec_res'].values[0]}, BJD: {round(non_outlier_high_res['bjd'].values[0]) - 2450000}", 
    #          f"HARPS, Res: {harps_point['spec_res'].values[0]}, BJD: {round(harps_point['bjd'].values[0]) - 2450000}"]
    labels_HD1461 = [f"Outlier, Res: {outlier_point['spec_res'].values[0]}, BJD: {round(outlier_point['bjd'].values[0]) - 2450000}", 
              f"Non-outlier, Res: {non_outlier_high_res['spec_res'].values[0]}, BJD: {round(non_outlier_high_res['bjd'].values[0]) - 2450000}", 
              f"HARPS, Res: {harps_point['spec_res'].values[0]}, BJD: {round(harps_point['bjd'].values[0]) - 2450000}"]
    offset = [0,0.2,-0.2]
    color_list = ["red","blue","black"]
    for i,spec in enumerate(list_points):
        if i == len(list_points): instr = "HARPS"
        file = spec["file"].values[0].replace('teste_download', 'teste_download_rv_corr')
        wv, flux, flux_err, hdr = read_fits(file,instrument=instr,mode="rv_corrected")
        plot_line(data=[(wv,flux)], line="Ha", offset=offset[i], normalize=True, line_color=color_list[i], line_legend=labels_HD1461[i], legend_plot = True, 
                  plot_continuum_vlines = False, plot_lines_vlines = False)
        #activity line bandpass
        plt.axvline(x=6562.808-0.6/2,ymin=0,ymax=1,ls="--",ms=0.1)
        plt.axvline(x=6562.808+0.6/2,ymin=0,ymax=1,ls="--",ms=0.1)
        #reference lines bandpass HaR1 and HaR2
        plt.axvline(x=6550.870-10.75/2,ymin=0,ymax=1,ls="--",ms=0.1)
        plt.axvline(x=6550.870+10.75/2,ymin=0,ymax=1,ls="--",ms=0.1)
        plt.axvline(x=6580.310-8.75/2,ymin=0,ymax=1,ls="--",ms=0.1)
        plt.axvline(x=6580.310+8.75/2,ymin=0,ymax=1,ls="--",ms=0.1)

        indice_act = actin_manual_Ha06(wv, flux)
        print(f"I_Halpha {list_labels[i]}: {indice_act}")        

    plt.title(f"{star}, Ha")
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"uves_tests_fig/{star}_Ha_overlapped.pdf",overwrite=True, format = 'pdf', dpi=300)

'''
HD1461: 1 outlier com RV perto de 0. o BJD é muito menor que os outros (8/2005)
HD16141: apenas 1 ponto não é outlier. outliers todos no mesmo dia (2004). RV simbad = -51, RV outlier = -22, não outlier = -76
HD102365: todos os outliers concentrados num dia, onde apenas 1 ponto não é outlier (6/2017, o penultimo dia)
HD115617: os outliers estão concentrados em 100<SNR<250, inconclusivo

todas as restantes não tem outliers (HD10647, HD13445, HD22049, HD47536, HD108147, HD179949), 
ou são todos outliers (HD102365, HD160691) e/ou é inconclusivo (HD209100, HD16417).

no caso de HD179949, os pontos estão no espaço de 2 dias

HD46375 não tem pontos nenhuns.

'''