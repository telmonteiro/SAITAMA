'''
This file contains general functions to be used by the pipeline function in the ACTINometer pipeline
'''

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math
from astropy.io import fits
from astropy.table import Table
from pipeline_functions.pyrhk.pyrhk import calc_smw, get_bv, calc_rhk, calc_prot_age
from PyAstronomy import pyasl # type: ignore
import urllib.request



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
    Rough sigma clipping of a data frame, allowing for NaN values.
    '''
    for col in cols:
        # Ignore NaN values for the mean and standard deviation calculation
        mean = df[col].dropna().mean()
        std = df[col].dropna().std()
        
        # Apply sigma clipping, retaining NaN values
        df = df[((df[col].isna()) | ((df[col] >= mean - sigma * std) & (df[col] <= mean + sigma * std)))]
        
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
    for i, index in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        plt.ylabel(index)
        plt.errorbar(df.bjd - 2450000, df[index], df[index + "_err"], fmt='k.')
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.subplots_adjust(top=0.95)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

#########################

def stats_indice(star, cols, df):
    """
    Return pandas DataFrame with statistical data on the indice(s) given: max, min, mean, median, std, weighted mean and N (number of spectra)
    Used for I_CaII, I_Ha06, I_NaI, rv, S_MW, log_Rhk, prot_n84, prot_m08
    """
    df_stats = pd.DataFrame(columns=["star", "indice", "max", "min", "mean", "median", "std", "weighted_mean", "time_span", "N_spectra"])

    if len(cols) < 1:
        print("ERROR: No columns given")
        return None

    for col in cols:
        if col not in df.columns:
            print(f"WARNING: Column {col} not found in DataFrame")
            continue

        data = pd.to_numeric(df[col], errors='coerce')  # Ensure data is numeric, coercing errors to NaN
        data = data.dropna()  # Remove NaN values

        bad_ind = [i for i,d in enumerate(data) if d == 1e+20] #when converting log Rhk this can happen
        data = data.drop(df.index[bad_ind])
        
        if col != "rv":
            data_err = pd.to_numeric(df[col+"_err"], errors='coerce')  # Ensure data is numeric, coercing errors to NaN
            data_err = data_err.dropna()  # Remove NaN values
            data_err = data_err.drop(df.index[bad_ind])

        if len(data) == 0:
            row = {"star": star, "indice": col,
                "max": np.nan, "min": np.nan,
                "mean": np.nan, "median": np.nan,
                "std": np.nan, "weighted_mean": np.nan,
                "time_span": np.nan,
                "N_spectra": 0}
        else:
            if col == "rv":
                weighted_mean = np.nan
            else:
                weights = 1/(data_err)**2
                weighted_mean = np.sum(weights * data) / np.sum(weights)
            
            row = {"star": star, "indice": col,
                "max": data.max(), "min": data.min(),
                "mean": data.mean(), "median": data.median(),
                "std": data.std(), "weighted_mean": weighted_mean,
                "time_span": df["bjd"].max() - df["bjd"].min(),
                "N_spectra": len(data)}
        
        df_stats.loc[len(df_stats)] = row

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
        if normalize == True:
            flux_normalized = flux/np.median(flux)
        else: flux_normalized = flux

        plt.plot(wv, flux_normalized+offset, lstyle, label=line_legend, color=line_color)
        
        if len(flux_normalized) < 50:
            lim = 5
        else: lim = 20
        
        if plot_continuum_vlines == True:
            plt.axvline(x=wv[lim],ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=wv[-lim],ymin=0,ymax=1,ls="--",ms=0.1)
        if plot_lines_vlines == True:
            plt.axvline(x=line_wv-0.05,ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=line_wv+0.05,ymin=0,ymax=1,ls="--",ms=0.1)

    plt.axvline(x=line_wv,ymin=0,ymax=1,ls="-",ms=0.2)
    if legend_plot == True: plt.legend()
    
    if normalize == True: ylab = "Normalized Flux"
    else: ylab = "Flux"
    plt.xlabel(r"Wavelength ($\AA$)"); plt.ylabel(ylab)
    plt.title(f"{line} line")

#################################
    
def instrument_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err, flag_period, flag_rv_ratio):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    if math.isnan(float(period)): period = 0
    if math.isnan(float(period_err)): period_err = 0

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]
    dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                "INSTR":[instr,"Instrument used"],
                "TIME_SPAN":[time_span, 'Time span in days between first and last observations used'],
                "SNR_MIN":[min_snr,"Minimum SNR"],
                "SNR_MAX":[max_snr,"Maximum SNR"],
                "PERIOD_I_CaII":[period,"Period of CaII activity index"],
                "PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
                "FLAG_PERIOD":[flag_period,"Goodness of periodogram fit flag. Color based."],
                "beta_RV":[flag_rv_ratio,"Goodness of RV correction indicador. 1 = all good"],
                "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                "COMMENT1":["RV obtained from CCF measure (m/s)","Comment"],
                "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}

    indices = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv', "S_MW", "log_Rhk", "prot_n84", "prot_m08","age_m08"]
    stats = ["max","min","mean","median","std","weighted_mean","N_spectra"]

    for i,ind in enumerate(indices):
        for col in stats:
            stat = stats_df[col][i]
            if math.isnan(float(stat)): stat = 0
            if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
            elif col == "N_spectra": comment = f"Nr of spectra used in {ind}"
            else: comment = f"{col} of {ind}"
            dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]

    for keyword in dict_hdr.keys():
        value = dict_hdr[keyword][0]
        if isinstance(value, float) and np.isnan(value):
            # Handle NaN values here, for example, replace them with 0
            value = 0
        hdr.append(("HIERARCH "+keyword, value, dict_hdr[keyword][1]), end=True)

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
    
#################################

def plot_RV_indices_diff_instr(star, df, indices, save, path_save):
    """
    Plot RV and indices given as a function of time.
    """

    instruments = df["instr"].unique()  # Get unique instruments

    plt.figure(figsize=(6.5, (len(indices) + 1.5) * 2))
    plt.suptitle(star, fontsize=14)

    for i, instrument in enumerate(instruments):
        instr_df = df[df["instr"] == instrument]

        for j, index in enumerate(indices):
            plt.subplot(len(indices) + 1, 1, j + 1)
            plt.ylabel(index)
            plt.errorbar(instr_df["bjd"] - 2450000, instr_df[index], instr_df[index + "_err"], fmt='.', label=instrument)
            plt.legend()

        plt.subplot(len(indices) + 1, 1, len(indices) + 1)
        if "rv_err" not in instr_df.columns:
            yerr = 0
        else:
            yerr = instr_df["rv_err"]
        plt.errorbar(instr_df["bjd"] - 2450000, instr_df["rv"], yerr, fmt='.', label=instrument)
        plt.ylabel("RV [m/s]")

    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.legend()
    plt.subplots_adjust(top=0.95)

    if save:
        plt.savefig(path_save, bbox_inches="tight")

##################################

def get_calibrations_CaII(star_gaia_dr3, instrument, I_CaII,I_CaII_err):
    '''Function to convert the I_CaII indice to Mount-Wilson index S_MW and to log R'_HK using the B-V color.
    Also computes if possible the rotation period with two different calibrations and the chromospheric age.
    Uses functions from the https://github.com/gomesdasilva/pyrhk/tree/master repository.
    
    Input parameters:
    - star_gaia_dr3: str in the format "GAIA DR3 ######", as only a part of the stars in SWEET-Cat has HD identifier.
    - I_CaII: activity indice from ACTIN2 for CaII H&K line.
    - I_CaII_err: error of the activity indice.
    
    Output parameters:
    - log_rhk, log_rhk_err: activity indice for CaII H&K line in log R'_HK scale and respective error
    - prot_n84, prot_n84_err: rotation period and respective error with the Noyes et al. (1984) calibration
    - prot_m08, prot_m08_err: rotation period and respective error with the Mamajek & Hillenbrand (2008) calibration
    - age_m08, age_m08_err : chromospheric age and respective error with the Mamajek & Hillenbrand (2008) calibration
    '''

    #converting to Mount-Wilson index S_MW
    if instrument == "ESPRESSO": instr = "ESPRESSO"
    else: instr="HARPS_GDS21" #UVES doesn't retrieve CaII indice
    smw, smw_err = calc_smw(caii=I_CaII, caii_err=I_CaII_err, instr=instr)
    smw = smw.astype(float); smw_err = smw_err.astype(float)

    #get Teff from SWEET-Cat
    sweetCat_table_url = "http://sweetcat.iastro.pt/catalog/SWEETCAT_Dataframe.csv"
    dtype_SW = dtype={'gaia_dr2':'int64','gaia_dr3':'int64'}
    SC = pd.read_csv(urllib.request.urlopen(sweetCat_table_url), dtype=dtype_SW)
    teff = SC[SC["gaia_dr3"]==int(star_gaia_dr3[9:])]["Teff"].values[0] 

    #get B-V from SIMBAD, if none get from Ballesteros calibration
    try:
        bv, bv_err, bv_ref = get_bv(star_gaia_dr3, alerts=True) 
    except:
        b = pyasl.BallesterosBV_T()
        bv = b.t2bv(T=teff) 

    if math.isnan(bv) == True:
        b = pyasl.BallesterosBV_T()
        bv = b.t2bv(T=teff) 

    #converting to log R'_HK using B-V. if star is M type (Teff < 3700 K) use the mascareno calibration, otherwise the rutten
    if teff < 3700:
        method_to_use = "mascareno"
    else: method_to_use = "rutten"

    log_rhk, log_rhk_err, rhk, rhk_err = calc_rhk(smw, smw_err, bv, method=method_to_use) #what about the evstage?

    #computing rotation period of star using 2 different calibrations, as well as the age of the star if possible
    prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err = calc_prot_age(log_rhk, bv)
    #for HARPS and ESPRESSO this may work well, but for UVES the calibrations may be bad or non existent?

    return smw, smw_err, log_rhk, log_rhk_err, prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err