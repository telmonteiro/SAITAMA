'''
This file contains general functions to be used by the pipeline function in the ACTINometer pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math
from astropy.io import fits
from astropy.table import Table

def read_fits(file_name,instrument,mode):
    '''
    Read fits file and get header and data. Varies if instrument is HARPS, ESPRESSO, UVES or FEROS. missing espresso
    '''
    hdul = fits.open(file_name)
    if instrument == "HARPS":
        if mode == "raw":
            if "s1d_A" in file_name:
                flux = hdul[0].data
                header = hdul[0].header
                wv = calc_fits_wv_1d(header)
                flux_err = np.zeros_like(flux)
            elif "ADP" in file_name:
                wv = hdul[1].data[0][0]
                flux = hdul[1].data[0][1]
                bjd = hdul[0].header["HIERARCH ESO DRS BJD"]
                header = hdul[0].header
                header["HIERARCH ESO DRS BJD"] = bjd
                if math.isnan(hdul[1].data["ERR"][0][0]):
                    flux_err = np.zeros_like(flux)
                else: 
                    flux_err = hdul[1].data["ERR"][0]
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header

    elif instrument == "UVES":
        if mode == "raw":
            header = hdul[0].header
            wv = hdul[1].data["WAVE"][0]
            try:
                flux = hdul[1].data["FLUX"][0]
                flux_err = hdul[1].data["ERR"][0]
            except:
                flux = hdul[1].data["FLUX_REDUCED"][0]
                flux_err = hdul[1].data["ERR_REDUCED"][0]
            bjd = hdul[0].header["MJD-OBS"]+2400000.5
            header["HIERARCH ESO DRS BJD"] = bjd
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header
    
    elif instrument == "ESPRESSO":
        if mode == "raw":
            if "s1d_A" in file_name:
                flux = hdul[0].data
                header = hdul[0].header
                wv = calc_fits_wv_1d(header)
                flux_err = np.zeros_like(flux)
            elif "ADP" in file_name:
                #print(hdul[0].data)
                header = hdul[0].header
                try:
                    wv = hdul[1].data["WAVE_AIR"][0]
                except:
                    wv = hdul[1].data["WAVE"][0]
                try:
                    flux = hdul[1].data["FLUX_EL"][0]
                    flux_err = hdul[1].data["ERR_EL"][0]
                    
                    valid_indices = ~np.isnan(flux)  #perigo de haver valores NaN neste tipo de flux
                    wv = wv[valid_indices]
                    flux = flux[valid_indices]
                    flux_err = flux_err[valid_indices]
                except:
                    flux = hdul[1].data["FLUX"][0]
                    flux_err = hdul[1].data["ERR"][0]
                bjd = hdul[0].header["MJD-OBS"]+2400000.5
                header["HIERARCH ESO DRS BJD"] = bjd
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            flux_err = hdul[0].data[2]
            header = hdul[0].header

    else:
        flux = hdul[0].data
        header = hdul[0].header
        wv = calc_fits_wv_1d(header)
        flux_err = np.zeros_like(flux)

    hdul.close()

    return wv, flux, flux_err, header

########################

def calc_fits_wv_1d(hdr, key_a='CRVAL1', key_b='CDELT1', key_c='NAXIS1'):
    '''
    Compute wavelength axis from keywords on spectrum header.
    '''
    try:
        a = hdr[key_a]; b = hdr[key_b]
    except KeyError:
        a = hdr["WAVELMIN"]*10; b = hdr["WAVELMAX"]*10
    try: 
        c = hdr[key_c]
    except KeyError:
        c = hdr["NELEM"]

    return a + b * np.arange(c)

########################

def sigma_clip(df, cols, sigma):
    '''
    Rough sigma clipping of a data frame.
    '''
    for col in cols:
        if math.isnan(list(df[col])[0]) == False:
            mean= df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - sigma * std) & (df[col] <= mean + sigma * std)]
    return df

########################

def plot_RV_indices(star,df,indices,save, path_save):
    """
    Plot RV and indices given as a function of time
    """
    plt.figure(figsize=(6, (len(indices)+1)*2))
    plt.suptitle(star, fontsize=14)
    plt.subplot(len(indices)+1, 1, 1)
    if "rv_err" not in df.columns: yerr = 0
    else: yerr = df.rv_err
    plt.errorbar(df.bjd - 2450000, df.rv, yerr, fmt='k.')
    plt.ylabel("RV [m/s]")
    #print(indices)
    for i, index in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        plt.ylabel(index)
        plt.errorbar(df.bjd - 2450000, df[index], df[index + "_err"], fmt='k.')
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.subplots_adjust(top=0.95)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

#########################
    
def stats_indice(star,cols,df):
    """
    Return pandas data frame with statistical data on the indice(s) given: max, min, mean, median, std and N (number of spectra)
    """
    df_stats = pd.DataFrame(columns=["star","indice","max","min","mean","median","std","time_span","N_spectra"])
    if len(cols) == 1:
            row = {"star":star,"column":cols,
                "max":max(df[cols]),"min":min(df[cols]),
                "mean":np.mean(df[cols]),"median":np.median(df[cols]),
                "std":np.std(df[cols]),"time_span":max(df["bjd"])-min(df["bjd"]),
                "N_spectra":len(df[cols])}
            df_stats.loc[len(df_stats)] = row
    elif len(cols) > 1:
        for i in cols:
            if i != "rv":
                indices = df[df[i+"_Rneg"] < 0.01].index
                data = df.loc[indices, i]
            else: data = df["rv"]
            if len(data) != 0:
                row = {"star": star, "indice": i,
                    "max": max(data), "min": min(data),
                    "mean": np.mean(data), "median": np.median(data),
                    "std": np.std(data), "time_span": max(df["bjd"]) - min(df["bjd"]),
                    "N_spectra": len(data)}
                df_stats.loc[len(df_stats)] = row
            else: 
                row = {"star": star, "indice": i,
                    "max": 0, "min": 0,
                    "mean": 0, "median": 0,
                    "std": 0, "time_span": max(df["bjd"]) - min(df["bjd"]),
                    "N_spectra": len(data)}
                df_stats.loc[len(df_stats)] = row

    else:
        print("ERROR: No columns given")
        df_stats = None
    
    return df_stats

########################

def plot_line(data, line, line_color=None,offset=0, line_legend="", lstyle = "-", normalize=True, legend_plot = False, plot_continuum_vlines = True, plot_lines_vlines = True):
    '''
    Plots the spectra used in the position of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,
             "NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 22
    else: window = 0.7

    for array in data:
        wv = array[0]; flux = array[1]
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))

        if wv_array[0].shape == (0,):
            continue

        wv = wv[wv_array]
        flux = flux[wv_array]
        #flux_normalized = flux/np.linalg.norm(flux)
        if normalize == True:
            flux_normalized = flux/np.median(flux)#(flux-np.min(flux))/(np.max(flux)-np.min(flux))
        else: flux_normalized = flux
        plt.plot(wv, flux_normalized+offset, lstyle, label=line_legend, color=line_color)
        if len(flux_normalized) < 50:
            lim = 5
        else: lim = 20
        if plot_continuum_vlines == True:
            plt.axvline(x=wv[lim],ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=wv[-lim],ymin=0,ymax=1,ls="--",ms=0.1)
        if plot_lines_vlines == True:
            plt.axvline(x=line_wv-window/30,ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=line_wv+window/30,ymin=0,ymax=1,ls="--",ms=0.1)

    plt.axvline(x=line_wv,ymin=0,ymax=1,ls="-",ms=0.2)
    if legend_plot == True: plt.legend()
    
    if normalize == True: ylab = "Normalized Flux"
    else: ylab = "Flux"
    plt.xlabel(r"Wavelength ($\AA$)"); plt.ylabel(ylab)
    plt.title(f"{line} line")

#################################
    
def general_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err, flag_period, flag_rv_ratio):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    if math.isnan(float(period_err)): period_err = 0

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]; #N_spectra = stats_df["N_spectra"][0]
    dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                "INSTR":[instr,"Instrument used"],
                "TIME_SPAN":[time_span, 'Time span in days between first and last observations used'],
                #"N_SPECTRA":[N_spectra,"Number of spectra used"],
                "SNR_MIN":[min_snr,"Minimum SNR"],
                "SNR_MAX":[max_snr,"Maximum SNR"],
                "PERIOD_I_CaII":[period,"Period of CaII activity index"],
                "PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
                "FLAG_PERIOD":[flag_period,"Goodness of periodogram fit flag. Color based."],
                "FLAG_RV":[flag_rv_ratio,"Goodness of RV correction indicador. 1 = all good"],
                "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                "COMMENT1":["RV obtained from CCF measure (m/s)","Comment"],
                "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}

    indices = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
    stats = ["max","min","mean","median","std","N_spectra"]

    for i,ind in enumerate(indices):
        for col in stats:
            stat = stats_df[col][i]
            if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
            elif col == "N_spectra": comment = f"Nr of spectra used in {ind}"
            else: comment = f"{col} of {ind}"
            dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]
    
    for keyword in dict_hdr.keys():
        hdr.append(("HIERARCH "+keyword, dict_hdr[keyword][0], dict_hdr[keyword][1]), end=True)

    df.columns = df.columns.astype(str)
    table = Table.from_pandas(df)
    hdul = fits.BinTableHDU(data=table, header=hdr)

    hdul.writeto(file_path, overwrite=True)

#################################

def read_bintable(file,print_info=False):
    '''
    Simple function to read the header and BinTable from the fits file that contains the statistics in the header and the data frame as BinTable
    '''
    hdul = fits.open(file)
    if print_info == True: hdul.info()
    hdr = hdul[1].header
    table = hdul[1].data
    astropy_table = Table(table)
    data_dict = astropy_table.to_pandas().to_dict()
    df = pd.DataFrame(data_dict)
    return df, hdr