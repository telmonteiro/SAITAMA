r'''
SAITAMA - a pipeline for the derivation of spectral activity indices for stars with exoplanets

This pipeline uses the ACTIN tool (2018JOSS....3..667G and 2021A&A...646A..77G) that computes the spectral activity indices for a stellar spectrum.
The objective is to apply ACTIN on the most stars possible from the SWEET-Cat catalogue (2021A&A...656A..53S), which contains the stars with exoplanets and respective
stellar parameters. The spectral activity indices chosen in this pipeline are the CaII H&K, the H$\alpha$ 0.6 A and the NaI lines.

- The pipeline starts by quering in ESO data base the spectra taken with the given instrument for the given object.
For each instrument:
- Checks if there is data to neglect and cuts the data sets by the minimum and maximum SNR chosen.
- Chooses the best SNR spectra that cover a big time span, binned by month. Making a trade-off between SNR and time spread.
- Checks if the folder for the instrument exists or not. If not creates the necessary folders.
- According to the instrument, retrieves the ADP spectra from ESO and untars the ancillary data (optional). For non-HARPS instruments, it just retrieves the ADP spectra.
- Corrects the spectra by RV comparing the spectra to a spectrum of the Sun with a CCF.
- Computes a quality indicator of the RV correction.
- Runs ACTIN2 on the spectra and obtains activity indices for CaII H&K, H$\alpha$ at 0.6 A and NaI.
- Computes a periodogram using GLS to retrieve the period of CaII H&K, and flags it according to a quality indicator.
- Converts the S_CaII to S_MW and then to log R'_HK and uses this last one to estimate rotation period and chromospheric age through calibrations
- Saves plots of the spectra in lines of interest to check if everything is alright, as well as plots of the activity indices and the statistics related to them.
- Saves the statistics and data frame in a fits file.
Then it combines the fits file per instrument into a master fits file, recomputing the statistics and the periodogram with all the data points.

Input parameters:
- stars: object identifiers, preferentially in the HD catalogue. Must be in a list format
- instruments: name of the spectrograph to retrieve the spectra from. Must be in a list format
- indices: spectral lines to compute the activity indices. Must be in a list format
- max_spectra: maximum number of spectra to be downloaded
- min_snr: minimum SNR to select the spectra. Must be in a float-like format
- download: boolean to decide if the spectra are downloaded or not (if not, it is assumed that the spectra is already inside the folders where 
the pipeline would download the spectra to)
- neglect_data: spectra to be manually neglected for whatever reason
- username_eso: username in the ESO data base to download the data
- download_path: folder to where download spectra
- final_path: folder to save products of the pipeline

Returns:
- creates a folder "{final_path}/{star}" where all the produced files are stored
- For each instrument:
    - in the subfolder ADP/ the corrected fits files of the spectra are stored
    - {star}_{instrument}_{line}.pdf: plot of the spectra in said line (CaI, CaII H, CaII K, H$\alpha$, HeI, NaID1, NaID2 and FeII)
    - stats_{star}.csv: csv file with a data frame including statistical information like the star name, the indice name and the respective maximum, minimum, mean, median, 
    standard deviation, weighted mean, time span and number of spectra used for that indice
    - df_{star}_{instrument}.csv: csv file with the data frame including the columns_df given as input before any processing
    - {star}_GLS.pdf and {star}_WF.pdf: plots of the periodogram of the CaII H&K indice and the respective Window Function
    - df_stats_{star}.fits: FITS file that includes the data frame given by df_{star}_{instrument}.csv in a BINTable format and includes the statistics given in stats_{star}.csv
    and informations regarding the periodogram in the header
    - report_periodogram_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics
- master_df_{target_save_name}.fits: FITS file that contains the information for each instrument separately plus a BINTable + Header for the combined data
- {star}_GLS.pdf and {star}_WF.pdf: plots of the periodogram of the CaII H&K indice and the respective Window Function for the combined data
- report_periodogram_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics for the combined data
'''
import os, glob, logging, numpy as np, matplotlib.pyplot as plt, pandas as pd, tqdm, math, time, requests

from astroquery.eso import Eso # type: ignore
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table

from pipeline_functions.get_spec_funcs import get_gaiadr3, choose_snr, select_best_spectra
from pipeline_functions.general_funcs import SpecFunc, dfFuncs, plot_line, instrument_fits_file, read_bintable, plot_indices_diff_instr, get_calibrations_CaII, bin_data
from pipeline_functions.RV_correction_funcs import RV_correction
from pipeline_functions.periodogram_funcs import gls_periodogram

from actin2 import ACTIN # type: ignore

import warnings
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

SpecFunc = SpecFunc()
dfFuncs = dfFuncs()
RV_correction = RV_correction()
actin = ACTIN()

# Setup logging logger
global logger
eso = Eso()

def download_spectra(eso, datasets, destination, max_retries=5):
    '''
    Downloads the spectra and safeguards against connection errors.
    '''
    retries = 0
    while retries < max_retries:
        try:
            #download the data
            eso.retrieve_data(datasets=datasets, destination=destination)
            print("Download completed successfully.")
            return
        except requests.exceptions.ConnectionError as e:
            retries += 1
            print(f"Connection error: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(2) #waits before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    print("Max retries reached. Please check which files were downloaded.")

def get_adp_spec(eso, search_name, name_target, neglect_data, instrument="HARPS", min_snr=10, max_snr=550, box=0.07, path_download_base="tmpdir_download/", max_spectra=250):
    """
    Downloads and processes spectra from ESO database.

    Args:
    eso: Eso instance from astroquery.
    search_name: Name to search in ESO database.
    name_target: Name of the target.
    neglect_data: Dictionary containing data to neglect.
    instrument: Instrument name.
    min_snr: Minimum SNR.
    max_snr: Maximum SNR.
    box: Search box.
    path_download_base: Base path for downloading.
    max_spectra: Maximum number of spectra to select.

    Returns:
    List of paths where the downloaded spectra are stored.
    """
    name_target = name_target.replace(" ", "")
    print("Searching: ", search_name)

    tbl_search = eso.query_surveys(surveys=instrument, target=search_name, box=box)

    if tbl_search is None:
        return "None"

    if instrument == "UVES":  # to ensure that at least the Halpha line is included
        wav_lim_min = np.array([int(float(wav.split("..")[0])) * 10 for wav in tbl_search["Wavelength"]])
        wav_lim_max = np.array([int(float(wav.split("..")[1])) * 10 for wav in tbl_search["Wavelength"]])

        ind_wav_lim = np.where((wav_lim_min < 6400) & (wav_lim_max > 6600))
        tbl_search = tbl_search[ind_wav_lim]

    # Removing eventually detected manually bad data
    print(tbl_search["ARCFILE"])
    if search_name in neglect_data.keys():
        i_clean = [i for i in range(len(tbl_search)) if tbl_search["ARCFILE"][i] not in neglect_data[search_name]]
        tbl_search = tbl_search[i_clean]

    print("Found %d datasets." % (len(tbl_search)))

    paths_download = []

    # cuts the lowest SNR by a minimum and a maximum
    icut = choose_snr(tbl_search["SNR"], min_snr=min_snr, max_snr=max_snr)
    if len(icut[0]) < 1:
        return "None"
    tbl_search = tbl_search[icut]
    print(f"{len(tbl_search)} datasets obey the SNR cut")

    # selects the best SNR spectra while keeping a time span significant
    tbl_search = select_best_spectra(tbl_search, max_spectra=max_spectra)
    print(f"{len(tbl_search)} datasets obey the selection process")

    # Download
    path_download = path_download_base + name_target + "_" + instrument + "/"
    if not os.path.isdir(path_download):
        os.mkdir(path_download)
        os.mkdir(path_download + "ADP/")
        paths_download.append(path_download)

    table = list(tbl_search["ARCFILE"])
    download_spectra(eso, datasets=table, destination=path_download + "ADP/", max_retries=5)
        
    return paths_download


def SAITAMA(stars, instruments, indices, max_spectra, min_snr,download, neglect_data, username_eso, download_path, final_path):
    '''Main pipeline function that runs the entire procedure, covering data retrieval, correction, activity indices computation
    and all the post-processing like periodogram analysis and statistical information.'''

    max_snr_instr = {"HARPS": 550,"ESPRESSO": 1000,"UVES": 550}  # max SNR to be used to avoid saturation

    for star_name in stars:

        gaiadr3 = get_gaiadr3(star_name)

        if gaiadr3 == -1:
            print("GAIA DR3 Not found automatically:")
            gaiadr3 = input("Insert GAIADR3 > ")
            star_gaia_dr3 = star_name
        else:
            star_gaia_dr3 = "GAIA DR3 " + gaiadr3

        print(star_name, star_gaia_dr3)

        # creates folders for the download and then the correct spectra
        if not os.path.isdir(download_path):
            os.mkdir(download_path)
        star_folder = f"{download_path}/{star_name}/"
        if not os.path.isdir(star_folder):
            os.mkdir(star_folder)

        if not os.path.isdir(final_path):
            os.mkdir(final_path)
        star_folder_rv = f"{final_path}/{star_name}/"
        if not os.path.isdir(star_folder_rv):
            os.mkdir(star_folder_rv)

        for instr in instruments:

            print(f"Processing and correcting the spectra by RV for {instr} instrument...")
            df = pd.DataFrame()

            max_snr = max_snr_instr[instr]
            if download == True:
                print(f"Downloading spectra from {instr} instrument...")

                # to login into the ESO data base
                eso.login(username=username_eso, store_password=True)
                eso.ROW_LIMIT = -1

                logger = logging.getLogger("individual Run")
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler("test.log", mode="w")
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s : %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logger.info("Started")
                fh.close()

                path_downloads = get_adp_spec(eso, star_gaia_dr3, star_name, neglect_data,
                    min_snr=min_snr,max_snr=max_snr, instrument=instr, path_download_base=star_folder, max_spectra=max_spectra)

                if path_downloads == None:
                    pass  # skip to next instrument if there are no observations with the present one

            # rounding up the spectra downloaded
            files = glob.glob(os.path.join(f"{download_path}/{star_name}/{star_name}_{instr}/ADP","ADP*.fits"))
            if len(files) == 0:
                continue
            files_tqdm = tqdm.tqdm(files) #to make a progress bar
            # template spectrum for RV correction
            spectrum_sun, sun_header = SpecFunc._read_fits(file_name="pipeline_functions/Sun1000.fits", instrument=None, mode=None)
            sun_template_flux = spectrum_sun["flux"]

            # creates folders to save the rv corrected fits files
            folder_path = (f"{final_path}/{star_name}/{star_name}_{instr}/")
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                os.mkdir(folder_path + "ADP/")
            if not os.path.isdir(folder_path+"gls/"):
                os.mkdir(folder_path+"gls/")
            if not os.path.isdir(f"{final_path}/{star_name}/gls/"):
                os.mkdir(f"{final_path}/{star_name}/gls/")
            
            data_array = []  # to plot the lines later
            drv = 0.5  # step to search rv

            for file in files_tqdm:

                spectrum, hdr = SpecFunc._read_fits(file, instr, mode="raw")
                wv, f = spectrum["wave"], spectrum["flux"]
                #f_err[np.isnan(f_err)] = 0 #some spectra have the error = nan, others do not

                bjd, rv_ccf, _, _, _, w, f = RV_correction._get_rv_ccf(star = star_name,
                                                                                stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                                                                                template_hdr = sun_header, template_spec = sun_template_flux,
                                                                                drv = drv, units = "m/s",
                                                                                instrument = instr, quick_RV = True)
                wv_corr = RV_correction._correct_spec_rv(w, rv_ccf, units="m/s")

                #creating the fits file of the corrected spectra
                data = np.vstack((wv_corr, f)) #consider only photon noise because some spectra have nan in the flux errors, and replacing by 0 sets the indicator error to 0
                hdu = fits.PrimaryHDU(data, header=hdr)
                hdul = fits.HDUList([hdu])
                file_path = file.replace(download_path, final_path)
                hdul.writeto(f"{file_path}", overwrite=True)
                hdul.close()
                data_array.append(data)

                #run ACTIN2
                spectrum = dict(wave=wv_corr, flux=f)#, flux_err=f_err)
                SNR = hdr["SNR"]
                # flag for goodness of RV correction
                beta_RV, gamma_RV_arr = RV_correction._get_betaRV([file_path], instr)
                gamma_RV = gamma_RV_arr[0]
                headers = {"bjd": bjd,"file": file,"instr": instr,"rv": rv_ccf,"obj": star_name,"SNR": SNR,"RV_flag": gamma_RV}
                df_ind = actin.CalcIndices(spectrum, headers, indices).indices
                df = df._append(pd.DataFrame([{**df_ind, **headers}]),ignore_index=True,sort=True)

            for ind in indices: #UVES does not cover CaII H&K so need to add empty columns for that indice
                if ind not in df.columns:
                    nan_array = np.empty((len(df),))
                    nan_array[:] = np.nan
                    df.insert(0, ind+"_err", nan_array)
                    df.insert(0, ind+"_Rneg", nan_array)
                    df.insert(0, ind, nan_array)

            df.to_csv(folder_path + f"df_{star_name}_{instr}_raw.csv", index=False) 

            print("Processing the data...")

            # if there are a lot of zero/negative fluxes in the window (>2%), discard for whatever indice
            for ind in indices:
                if ind not in df.columns:
                    continue
                #if Rneg or Rzero >2%, replace by NaN
                indice_Rneg = np.array(df[f"{ind}_Rneg"]) 
                Rneg_bad_ind = np.where(indice_Rneg > 0.02)[0]
                df.loc[Rneg_bad_ind, f"{ind}"] = np.nan

            # flag for goodness of RV correction
            gamma_RV_col = np.array(df["RV_flag"])
            good_spec_ind = np.where(gamma_RV_col == 0)
            N_good_spec = len(gamma_RV_col[good_spec_ind]) #number of well corrected spectra
            beta_RV = N_good_spec / len(gamma_RV_col)
            bad_spec_indices = np.where(gamma_RV_col == 1)[0]
            df = df.drop(list(bad_spec_indices)).reset_index(drop=True) #dropping the badly corrected spectra. the only instance where we really need to drop rows

            # plot the wavelength of known lines to check if RV correction is good
            for line in ["Ha", "CaIIH", "CaIIK", "FeII", "NaID1", "NaID2", "HeI", "CaI"]:
                if not os.path.isdir(folder_path+"spectral_lines/"):
                    os.mkdir(folder_path+"spectral_lines/")
                plt.figure(2)
                plot_line(data=data_array, line=line, line_color=None,plot_lines_vlines=False,plot_continuum_vlines=False, normalize=True)
                plt.savefig(folder_path+"spectral_lines/"+f"{star_name}_{instr}_{line}.pdf",bbox_inches="tight",)
                plt.clf()

            if beta_RV > 0:  # if there is at least one spectrum nicely corrected
                cols = ["I_CaII", "I_Ha06", "I_NaI"]
                if len(df) > 10:  # perform a sigma-clip only if it has at least 10 rows
                    #sequential 3-sigma clip
                    for col in cols:
                        df = dfFuncs._seq_sigma_clip(df, col, sigma=3, show_plot=False) #indices values
                        #df = dfFuncs._seq_sigma_clip(df, col+"_err", sigma=3, show_plot=False) #indices error values

                #binning the data to days
                table = pd.DataFrame()
                bin_bjd = None

                for i, column in enumerate(df.columns):
                    if pd.api.types.is_numeric_dtype(df[column]):
                        #if the column contains numeric values, apply binning
                        bin_bjd, bin_column_values, _ = bin_data(df["bjd"], df[column])
                        table[column] = bin_column_values
                        if bin_bjd is not None and i == 0:
                            table["bjd"] = bin_bjd
                    else:
                        table[column] = df[column]

                df = table.apply(pd.to_numeric, errors='ignore')
                df.to_csv(folder_path + f"df_{star_name}_{instr}_clean.csv", index=False) 

                plt.figure(3)
                dfFuncs._plot_time_series(star_name, df, indices, path_save=folder_path + f"{star_name}_{instr}.pdf")
                plt.clf()

                gls_list = []    
                fig, axes = plt.subplots(nrows=len(cols),ncols=2,figsize=(14, 0.5+len(cols)*2), sharex="col")

                for i,ind in enumerate(cols):

                    df_ind = df[["bjd",ind,ind+"_err"]]
                    df_ind = df_ind.dropna()
                    yerr = np.asarray(df_ind[ind+"_err"])

                    x = np.asarray(df_ind["bjd"])
                    y = np.asarray(df_ind[ind])

                    if len(df_ind["bjd"]) < 3:
                        gls_dic = pd.DataFrame({f"period":[0], f"period_err":[0], f"flag_period":["white"]})
                        gls_list.append(gls_dic)
                        continue

                    t_span = max(x) - min(x)
                    gls = gls_periodogram(star = star_name, ind = ind, x = x, y = y, y_err = yerr, 
                                            pmin=1.5, pmax=t_span, steps=1e6, verb = False, save=True, folder_path=folder_path+"gls/")
                    results = gls.run()

                    axes[i, 0].errorbar(df_ind[f"bjd"] - 2450000, df_ind[ind], df_ind[ind + "_err"], fmt="k.")
                    ylabel = rf"I$_{{{ind[2:]}}}$"
                    axes[i, 0].set_ylabel(ylabel, fontsize=13)
                    if i == len(cols) - 1:
                        axes[i, 0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)
                    axes[i, 0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

                    axes[i, 1].semilogx(results["period"], results["power"], "k-")
                    axes[i, 1].plot(
                        [min(results["period"]), max(results["period"])],
                        [results["fap_01"]] * 2,
                        "--",color="black",lw=0.7)
                    axes[i, 1].plot(
                        [min(results["period"]), max(results["period"])],
                        [results["fap_1"]] * 2,
                        "--",color="black",lw=0.7)
                    axes[i, 1].plot(
                        [min(results["period"]), max(results["period"])],
                        [results["fap_5"]] * 2,
                        "--",color="black",lw=0.7)
                    axes[i, 1].set_ylabel("Norm. Power", fontsize=13)
                    if i == len(cols) - 1:
                        axes[i, 1].set_xlabel("Period [d]", fontsize=13)
                    axes[i, 1].tick_params(axis="both", direction="in", top=True, right=True, which='both')
                    if results['period_best'] > 0:
                        axes[i, 1].text(0.05, 0.95,
                            f"P = {np.around(results['period_best'], 1)} ± {np.around(results['period_best_err'], 1)} d ({results['flag']})",
                            fontsize=13,
                            transform=axes[i, 1].transAxes,verticalalignment="top",bbox={"facecolor":'white'})

                    gls_dic = pd.DataFrame({f"period":[results["period_best"]],
                                f"period_err":[results["period_best_err"]],
                                f"flag_period":[results["flag"]]})
                    
                    gls_list.append(gls_dic)

                fig.subplots_adjust(hspace=0.0)
                fig.text(0.13, 0.89, f"{star_name} - {instr}", fontsize=17)
                fig.savefig(folder_path+f"{star_name}_{instr}_GLS.pdf",bbox_inches="tight",dpi=1000)
                plt.close('all')

                if gls_list:
                    all_gls_df = pd.concat(gls_list, axis=0, ignore_index=True)

                smw, smw_err, log_rhk, log_rhk_err, prot_n84, prot_n84_err, prot_m08, prot_m08_err, age_m08, age_m08_err = get_calibrations_CaII(star_gaia_dr3,instr,df["I_CaII"],df["I_CaII_err"])

                new_cols_name = ["S_MW","S_MW_err","log_Rhk","log_Rhk_err","prot_n84","prot_n84_err","prot_m08","prot_m08_err","age_m08","age_m08_err"]
                new_cols = [smw,smw_err,log_rhk,log_rhk_err,prot_n84,prot_n84_err,prot_m08,prot_m08_err,age_m08,age_m08_err]
                
                for i,col_name in enumerate(new_cols_name):
                    new_col = np.where(new_cols[i] == 1.000000e+20, np.nan, new_cols[i])
                    df.insert(len(df.columns), col_name, new_col)

                cols += ["S_MW","log_Rhk","prot_n84","prot_m08","age_m08"]
                stats_df = dfFuncs._stats(star_name, cols, df)
                stats_df = pd.concat([stats_df, all_gls_df], axis=1)
                stats_df.to_csv(folder_path + f"stats_{star_name}_{instr}.csv")

                file_path = folder_path + f"df_stats_{star_name}_{instr}.fits" #save into the final fits file
                instrument_fits_file(indices, stats_df, df, file_path, min_snr, max_snr, instr, gls_list, beta_RV)

                # shutil.rmtree(f"teste_download/{target_save_name}/")  # remove original fits files


        print("Making the final data frame...")
        master_df = pd.DataFrame() #initialize an empty DataFrame to hold all the data for the current star
        master_header = fits.Header() 
        hdulist = [fits.PrimaryHDU()]  #initialize HDU list with a primary HDU

        list_instruments = []
        for instr in instruments:

            folder_path = os.path.join(final_path, f"{star_name}/{star_name}_{instr}/")
            file_path = folder_path + f"df_stats_{star_name}.fits"
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            list_instruments.append(instr)

            df, hdr = read_bintable(file_path)
            master_df = pd.concat([master_df, df], ignore_index=True)

            # Convert the individual DataFrame to a FITS HDU and add it to the HDU list
            for col in df.columns:
                if col not in ["file","instr","obj"]:
                    df[col] = df[col].astype(str)
            table = Table.from_pandas(df)
            hdulist.append(fits.BinTableHDU(data=table, header=hdr, name=f"{instr}"))

        # Ensure all columns are compatible with FITS
        for col in master_df.columns:
            if col not in ["file","instr","obj"]:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

        cols = ["I_CaII", "I_Ha06", "I_NaI"]
        if len(master_df) > 10:  # perform a sigma-clip
            for col in cols:
                master_df = dfFuncs._seq_sigma_clip(master_df, col, sigma=3)
                #master_df = dfFuncs._seq_sigma_clip(master_df, col+"_err", sigma=3) 

        print(master_df)

        plt.figure(5)
        #dfFuncs._plot_time_series(star_name, master_df, indices, path_save=f"{final_path}/{star_name}/{star_name}.pdf")
        plot_indices_diff_instr(star=star_name, df=master_df, indices=indices, save=True, path_save=f"{final_path}/{star_name}/{star_name}.pdf")
        plt.clf()
        
        snr_min = np.min(master_df["SNR"]); snr_max = np.max(master_df["SNR"])

        gls_list = []        
        fig, axes = plt.subplots(nrows=len(cols),ncols=2,figsize=(14, 0.5+len(cols)*2), sharex="col")

        for i,ind in enumerate(cols):

            df_ind = df[["bjd",ind,ind+"_err"]]
            df_ind = df_ind.dropna()
            
            x = np.asarray(df_ind["bjd"], dtype=float)
            y = np.asarray(df_ind[ind], dtype=float)
            yerr = np.asarray(df_ind[ind+"_err"], dtype=float)

            if len(df_ind["bjd"]) < 3:
                gls_dic = pd.DataFrame({f"period":[0], f"period_err":[0], f"flag_period":[0]})
                gls_list.append(gls_dic)
                continue
                
            t_span = max(x) - min(x)
            gls = gls_periodogram(star = star_name, ind = ind, x = x, y = y, y_err = yerr, 
                                    pmin=1.5, pmax=t_span, steps=1e6, verb = False, save=True, folder_path=f"{final_path}/{star_name}/gls/")
            results = gls.run()

            axes[i, 0].errorbar(x - 2450000, y, yerr, fmt="k.")
            ylabel = rf"I$_{{{ind[2:]}}}$"
            axes[i, 0].set_ylabel(ylabel, fontsize=13)
            if i == len(cols) - 1:
                axes[i, 0].set_xlabel("BJD $-$ 2450000 [d]", fontsize=13)
            axes[i, 0].tick_params(axis="both", direction="in", top=True, right=True, which='both')

            axes[i, 1].semilogx(results["period"], results["power"], "k-")
            axes[i, 1].plot(
                [min(results["period"]), max(results["period"])],
                [results["fap_01"]] * 2,
                "--",color="black",lw=0.7)
            axes[i, 1].plot(
                [min(results["period"]), max(results["period"])],
                [results["fap_1"]] * 2,
                "--",color="black",lw=0.7)
            axes[i, 1].plot(
                [min(results["period"]), max(results["period"])],
                [results["fap_5"]] * 2,
                "--",color="black",lw=0.7)
            axes[i, 1].set_ylabel("Norm. Power", fontsize=13)
            if i == len(cols) - 1:
                axes[i, 1].set_xlabel("Period [d]", fontsize=13)
            axes[i, 1].tick_params(axis="both", direction="in", top=True, right=True, which='both')
            if results['period_best'] > 0:
                axes[i, 1].text(0.05, 0.95,
                    f"P = {np.around(results['period_best'], 1)} ± {np.around(results['period_best_err'], 1)} d ({results['flag']})",
                    fontsize=13,
                    transform=axes[i, 1].transAxes,verticalalignment="top",bbox={"facecolor":'white'})

            gls_dic = pd.DataFrame({f"period":[results["period_best"]],
                        f"period_err":[results["period_best_err"]],
                        f"flag_period":[results["flag"]]})
            gls_list.append(gls_dic)

        fig.subplots_adjust(hspace=0.0)
        fig.text(0.13, 0.89, f"{star_name}", fontsize=17)
        fig.savefig(f"{final_path}/{star_name}/"+f"{star_name}_GLS.pdf",bbox_inches="tight",dpi=1000)
        plt.close('all')

        if gls_list:
            all_gls_df = pd.concat(gls_list, axis=0, ignore_index=True)

        cols += ["S_MW","log_Rhk","prot_n84","prot_m08","age_m08"]
        stats_master_df = dfFuncs._stats(star_name, cols, master_df) #computes the statistics. include errors or not?
        stats_master_df = pd.concat([stats_master_df, all_gls_df], axis=1)
        stats_master_df.to_csv(f"{final_path}/{star_name}/" + f"stats_{star_name}.csv")

        # Ensure all columns are compatible with FITS
        for col in df.columns:
            df[col] = df[col].astype(str)

        list_instruments_str = ""
        for ins in list_instruments:
            list_instruments_str += ins + ","

        dict_hdr = {"STAR_ID":[star_name,'Star ID in HD catalogue'],
                    "INSTRUMENTS":[list_instruments_str,"Instruments used"],
                    "TIME_SPAN":[t_span, 'Time span in days between first and last observations used'],
                    "SNR_MIN":[snr_min,"Minimum SNR"],
                    "SNR_MAX":[snr_max,"Maximum SNR"],
                    "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                    "COMMENT1":["RV obtained from CCF measure (m/s)","Comment"],
                    "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}
        
        for i,ind in enumerate(indices):
            dict_hdr[f"PERIOD_{ind}"] = [gls_list[i]["period"][0],f"Period of {ind} activity index"]
            dict_hdr[f"PERIOD_{ind}_err"] = [gls_list[i]["period_err"][0],f"Error of period of {ind} activity index"]
            dict_hdr[f"FLAG_PERIOD_{ind}"] = [gls_list[i]["period"][0],f"Period of {ind} activity index"]

        values = ['I_CaII', 'I_Ha06', 'I_NaI', "S_MW", "log_Rhk", "prot_n84", "prot_m08", "age_m08"]
        stats = ["max","min","mean","median","std","weighted_mean","N_spectra"]

        for i,ind in enumerate(values):
            for col in stats:
                stat = stats_master_df[col][i]
                if math.isnan(float(stat)): stat = 0
                if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
                elif col == "N_spectra": comment = f"Nr of spectra used in {ind}"
                else: comment = f"{col} of {ind}"
                dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]
        
        for keyword in dict_hdr.keys():
            master_header.append(("HIERARCH "+keyword, dict_hdr[keyword][0], dict_hdr[keyword][1]), end=True)
        
        master_table = Table.from_pandas(master_df)
        master_hdu = fits.BinTableHDU(data=master_table, header=master_header, name="MASTER")
        
        # Add the master HDU to the HDU list
        hdulist.append(master_hdu)

        # Write all HDUs to a single FITS file
        output_file = os.path.join(final_path, f"{star_name}/master_df_{star_name}.fits")
        fits.HDUList(hdulist).writeto(output_file, overwrite=True)
        print(f"Master FITS file for {star_name} created successfully.")