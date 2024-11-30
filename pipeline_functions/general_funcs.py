'''
This file contains general functions to be used by the pipeline function in the SAITAMA pipeline
'''
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np, pandas as pd, matplotlib.pylab as plt, math
from astropy.io import fits
from astropy.table import Table
from pyrhk.pyrhk import calc_smw, get_bv, calc_rhk, calc_prot_age
from PyAstronomy import pyasl # type: ignore
import urllib.request

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpecFunc:

    def __init__(self):
        self.name = "SpecFunc"

    def _read_fits(self, file_name, instrument, mode):
        '''
        Read fits file and get header and data. Varies if instrument is HARPS, ESPRESSO or UVES.
        '''
        hdul = fits.open(file_name)

        if instrument == "HARPS":

            if mode == "raw":
                wv = hdul[1].data[0][0]
                flux = hdul[1].data[0][1]
                flux_err = hdul[1].data["ERR"][0]
                header = hdul[0].header
                bjd = header["HIERARCH ESO DRS BJD"]
                header["HIERARCH ESO DRS BJD"] = bjd
            elif mode == "rv_corrected":
                wv = hdul[0].data[0]
                flux = hdul[0].data[1]; #flux_err = hdul[0].data[2]
                header = hdul[0].header

        elif instrument == "UVES":

            if mode == "raw":
                header = hdul[0].header
                wv = hdul[1].data["WAVE"][0]
                try:
                    flux = hdul[1].data["FLUX"][0]; #flux_err = hdul[1].data["ERR"][0]
                except:
                    flux = hdul[1].data["FLUX_REDUCED"][0]; #flux_err = hdul[1].data["ERR_REDUCED"][0]
                bjd = header["MJD-OBS"]+2400000.5
                header["HIERARCH ESO DRS BJD"] = bjd
            elif mode == "rv_corrected":
                wv = hdul[0].data[0]
                flux = hdul[0].data[1]; #flux_err = hdul[0].data[2]
                header = hdul[0].header
        
        elif instrument == "ESPRESSO":

            if mode == "raw":
                header = hdul[0].header
                try:
                    wv = hdul[1].data["WAVE_AIR"][0]
                except:
                    wv = hdul[1].data["WAVE"][0]
                try:
                    flux = hdul[1].data["FLUX_EL"][0]; #flux_err = hdul[1].data["ERR_EL"][0]
                except:
                    flux = hdul[1].data["FLUX"][0]; #flux_err = hdul[1].data["ERR"][0]
                bjd = header["MJD-OBS"]+2400000.5
                header["HIERARCH ESO DRS BJD"] = bjd
            elif mode == "rv_corrected":
                wv = hdul[0].data[0]
                flux = hdul[0].data[1]; #flux_err = hdul[0].data[2]
                header = hdul[0].header

        else:
            flux = hdul[0].data; #flux_err = np.zeros_like(flux)
            header = hdul[0].header
            wv = self._calc_fits_wv_1d(header)

        hdul.close()

        spectrum = {'wave':wv, 'flux':flux, }#'flux_error':flux_err}

        return spectrum, header

    ########################

    def _calc_fits_wv_1d(self, hdr, key_a='CRVAL1', key_b='CDELT1', key_c='NAXIS1'):
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



class dfFuncs:

    def __init__(self):
        self.info = "Functions to apply to the data frame, whether it's plots or cleaning."

    def _plot_time_series(self, star, df, indices, path_save):
        """
        Plot RV and indices given as a function of time (time series).

        Args:
            star      (str): star ID.
            df        (pandas data frame): data frame with the RV, activity indices and other information.
            indices   (list): list containing the activity indices names.
            path_save (str): path to save the plot.

        Returns:
            plt: plot of time series.
        """

        plt.figure(figsize=(6, (len(indices)+1)*2))
        plt.suptitle(star, fontsize=14)

        for i, index in enumerate(indices):
            plt.subplot(len(indices), 1, i+1)
            plt.ylabel(index)
            if index + "_err" not in list(df.columns):
                plt.scatter(df[f"bjd"] - 2450000, df[index],marker=".",color="k")
            else:
                plt.errorbar(df[f"bjd"] - 2450000, df[index], df[index + "_err"], fmt='k.')

        plt.xlabel("BJD $-$ 2450000 [days]")
        plt.subplots_adjust(top=0.95)
        
        if path_save:
            plt.savefig(path_save, bbox_inches="tight")
            plt.close()

    def _seq_sigma_clip(self, df, key, sigma=3, show_plot=False):
        """Sequencial sigma clip of given 'key' for a full DataFrame, 'df'.

        The time series of 'key' will be sequencially sigma clipped until no more values are above the 'sigma' from the median.

        Args:
            df (pandas DataFrame): DataFrame with time series to be clipped
            key (str): DataFrame column on which to apply sigma clip
            sigma (int): Sigma (number of standard deviations) value
            show_plot (bool): Show diagnostic plot

        Returns:
            DataFrame: Sigma clipped DataFrame
        """
        mask = (df[key] >= np.nanmedian(df[key]) - sigma*np.nanstd(df[key]))
        mask &= (df[key] <= np.nanmedian(df[key]) + sigma*np.nanstd(df[key]))

        if show_plot:
            plt.figure()
            plt.plot(df.bjd, df[key], 'k.', label=key)
            plt.axhline(df[key].median(), color='k', ls='--', lw=0.7)
            plt.axhline(df[key].median() + sigma*df[key].std(), color='k', ls=':', lw=0.7)
            plt.axhline(df[key].median() - sigma*df[key].std(), color='k', ls=':', lw=0.7)

        while len(df[key].dropna()[~mask]) != 0:
            df.loc[~mask, key] = np.nan

            mask = (df[key] >= np.nanmedian(df[key]) - sigma*df[key].std())
            mask &= (df[key] <= np.nanmedian(df[key]) + sigma*df[key].std())

        if show_plot:  
            plt.plot(df.bjd, df[key], color='none', marker='o', mec='r', ls='')
            plt.legend()
            plt.show()
            plt.close()

        return df    


    def _stats(self, star, cols, df):
        """
        Return pandas DataFrame with statistical data on the indice(s) given: max, min, mean, median, std, weighted mean and N (number of spectra)
        
        Args:
            star      (str): star ID.
            cols      (list): columns to compute the statistics on.
            df        (pandas data frame): data frame with the RV, activity indices and other information.

        Returns:
            df_stats  (pandas data frame): data frame with the statistical information.
        
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
            data_err = pd.to_numeric(df[col+"_err"], errors='coerce')  # Ensure data is numeric, coercing errors to NaN
            data_err = data_err.dropna()  # Remove NaN values
            
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
    elif line in ["Ha","NaID1","NaID2"]: window = 14
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
        
        if plot_continuum_vlines == True: #mostly for CaI
            plt.axvline(x=line_wv - window + 0.2,ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=line_wv + window - 0.2,ymin=0,ymax=1,ls="--",ms=0.1)
        if plot_lines_vlines == True:
            plt.axvline(x=line_wv-0.03,ymin=0,ymax=1,ls="--",ms=0.1)
            plt.axvline(x=line_wv+0.03,ymin=0,ymax=1,ls="--",ms=0.1)

    plt.axvline(x=line_wv,ymin=0,ymax=1,ls="-",ms=0.2)
    if legend_plot == True: plt.legend()
    
    if normalize == True: ylab = "Normalized Flux"
    else: ylab = "Flux"
    plt.xlabel(r"Wavelength ($\AA$)"); plt.ylabel(ylab)
    plt.title(f"{line} line")

#################################
    
def instrument_fits_file(indices, stats_df, df, file_path, min_snr, max_snr, instr, gls_list, flag_rv_ratio):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]
    
    dict_hdr = {
            "STAR_ID": [star_id, 'Star ID in HD catalogue'],
            "INSTR": [instr, "Instrument used"],
            "TIME_SPAN": [time_span, 'Time span in days between first and last observations used'],
            "SNR_MIN": [min_snr, "Minimum SNR"],
            "SNR_MAX": [max_snr, "Maximum SNR"],
            "FLAG_RV": [flag_rv_ratio, "Goodness of RV correction indicator. 1 = all good"],
            "COMMENT": ["Spectra based on SNR - time span trade-off", "Comment"],
            "COMMENT1": ["RV obtained from CCF measure (m/s)", "Comment"],
            "COMMENT2": ["3D data of wv (Angs) and flux of each spectrum", "Comment"],
        }
    
    for i,ind in enumerate(indices):
        dict_hdr[f"PERIOD_{ind}"] = [gls_list[i]["period"][0],f"Period of {ind} activity index"]
        dict_hdr[f"PERIOD_{ind}_err"] = [gls_list[i]["period_err"][0],f"Error of period of {ind} activity index"]
        dict_hdr[f"FLAG_PERIOD_{ind}"] = [gls_list[i]["period"][0],f"Period of {ind} activity index"]

    cols = ['I_CaII', 'I_Ha06', 'I_NaI', "S_MW", "log_Rhk", "prot_n84", "prot_m08","age_m08"]
    stats = ["max","min","mean","median","std","weighted_mean","N_spectra"]

    for i,ind in enumerate(cols):
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

def plot_indices_diff_instr(star, df, indices, save, path_save):
    """
    Plot indices given as a function of time
    """

    instruments = df["instr"].unique()  # Get unique instruments

    plt.figure(figsize=(6.5, (len(indices) + 1.5) * 2))
    plt.suptitle(star, fontsize=14)

    for i, instrument in enumerate(instruments):
        instr_df = df[df["instr"] == instrument]

        for j, index in enumerate(indices):
            plt.subplot(len(indices)+1, 1, j+1)
            plt.ylabel(index)
            plt.errorbar(instr_df["bjd"] - 2450000, instr_df[index], instr_df[index + "_err"], fmt='.', label=instrument)
            plt.legend()

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
    bv, bv_err, bv_ref = get_bv(star_gaia_dr3, alerts=True) 
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

    return smw, smw_err, log_rhk, log_rhk_err, prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err



def bin_data(x, y, err=None, bsize=1, stats="median", n_vals=2, estats="quad", show_plot=False):
    """Bin time series data.

    Args:
        x (list, array): Time domain values.
        y (list, array): Y-coordinate values.
        err (list, array): Uncertainties of the y-coordinate values. If no errors set to 'None'.
        bsize (int): Size of the bins in terms of time values. If time is in
            days, bsize = 1 will bin per day.
        stats (str): The statistic to apply to y-coordinate inside each bin.
            Options are 'mean', 'median', 'std' for standard deviation, 'min' for minimum and 
            'max' for maximum.
        n_vals (int): The number of points to be included in the 'min' or 'max' stats option.
        estats (int): The statistic to apply to the uncertainties, 'err'. Options are 
            'quad' for quadratically added errors and 'SEM' for standard error on the
            mean.
        show_plot (bool): If 'True' shows a plot of the procedure (for diagnostic).
    
    Returns:
        (array): Binned time domain
        (array): Binned y-coordinate
        (array): Errors of the bin statistic
    """
    # need to order the two arrays by ascending x

    x, y = zip(*sorted(zip(x, y)))

    buffer = 10e20

    x = np.asarray(x) * buffer
    y = np.asarray(y)
    bsize = bsize * buffer

    # Set errors:
    if isinstance(err, type(None)):
        err = np.zeros_like(x)
    else:
        err = np.asarray(err)

    # Create grid of bins:
    start = np.nanmin(x)
    end = np.nanmax(x) + bsize
    bins = np.arange(start, end, bsize)

    # Initialize lists:
    x_stat = np.zeros_like(bins)
    y_stat = np.zeros_like(bins)
    y_stat_err = np.zeros_like(bins)

    def calc_mean(x, err):
        if err.any():
            x_stat = np.average(x, weights=1/err**2)
        else:
            x_stat = np.average(x)
        return x_stat

    for i in range(bins.size):
        mask = (x >= bins[i]) & (x < bins[i] + bsize)

        # Dealing with no data in bin:
        if not x[mask].size:
            x_stat[i] = np.nan
            y_stat[i] = np.nan
            y_stat_err[i] = np.nan
            continue

        if stats == "mean":
            x_stat[i] = calc_mean(x[mask], err[mask])
            y_stat[i] = calc_mean(y[mask], err[mask])
        elif stats == "median":
            x_stat[i] = np.median(x[mask])
            y_stat[i] = np.median(y[mask])
        elif stats == "std":
            x_stat[i] = np.average(x[mask])
            y_stat[i] = np.std(y[mask])
        elif stats == "min":
            indices = y[mask].argsort()
            x_stat[i] = np.average(x[mask][indices][:n_vals])
            sorted_y = y[mask][y[mask].argsort()][:n_vals]
            y_stat[i] = np.average(sorted_y)
        elif stats == "max":
            indices = y[mask].argsort()
            x_stat[i] = np.average(x[mask][indices][-n_vals:])
            sorted_y = y[mask][y[mask].argsort()][-n_vals:]
            y_stat[i] = np.average(sorted_y)
        else:
            raise Exception("*** Error: 'stats' needs to be one of: 'mean', 'median', 'std', 'min', 'max'.")

        if estats == "quad":
            # quadratically added errors
            y_stat_err[i] = np.sqrt(np.sum(err[mask]**2))/y[mask].size
        elif estats == "SEM":
            # standard error on the mean
            y_stat_err[i] = np.std(y[mask])/np.sqrt(y[mask].size)
        else:
            raise Exception("*** Error: 'estats' needs to be one of: 'quad', 'SEM'.")

    remove_nan = (~np.isnan(x_stat))
    x_stat = x_stat[remove_nan]
    y_stat = y_stat[remove_nan]
    y_stat_err = y_stat_err[remove_nan]

    x_stat = np.asarray(x_stat)
    y_stat = np.asarray(y_stat)
    y_stat_err = np.asarray(y_stat_err)

    if show_plot:
        plt.figure("bin_data diagnostic")
        plt.title(f"Binning result: bsize = {bsize/buffer}")
        plt.errorbar(x/buffer, y, err, fmt='.r')
        plt.errorbar(x_stat/buffer, y_stat, y_stat_err, color='none', ecolor='k', markeredgecolor='k', marker='o', ls='')
        for step in bins:
            plt.axvline(step/buffer, color='k', ls=':', lw=0.7)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

    return x_stat/buffer, y_stat, y_stat_err