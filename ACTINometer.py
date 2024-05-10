r'''
ACTINometer - a pipeline for the derivation of spectral activity indices for stars with exoplanets

This pipeline uses the ACTIN tool (2018JOSS....3..667G and 2021A&A...646A..77G) that computes the spectral activity indices for a stellar spectrum.
The objective is to apply ACTIN on the most stars possible from the SWEET-Cat catalogue (2021A&A...656A..53S), which contains the stars with exoplanets and respective
stellar parameters. The spectral activity indices chosen in this pipeline are the CaII H&K, the H$\alpha$ 0.6 A and the NaI lines.

- The pipeline starts by quering in ESO data base the spectra taken with the given instrument for the given object.
- Checks if there is data to neglect and cuts the data sets by the minimum and maximum SNR chosen.
- Chooses the best SNR spectra that cover a big time span, binned by month. Making a trade-off between SNR and time spread.
- Checks if the folder for the instrument exists or not. If not creates the necessary folders.
- According to the instrument, retrieves the ADP spectra from ESO and untars the ancillary data (optional). For non-HARPS instruments, it just retrieves the ADP spectra.
- Corrects the spectra by RV comparing the spectra to a spectrum of the Sun with a CCF.
- Computes a quality indicator of the RV correction.
- Runs ACTIN2 on the spectra and obtains activity indices for CaII H&K, H$\alpha$ at 0.6 A and NaI.
- Computes a periodogram using GLS to retrieve the period of CaII H&K, and flags it according to a quality indicator.
- Saves plots of the spectra in lines of interest to check if everything is alright, as well as plots of the activity indices and the statistics related to them.
- Saves the statistics and data frame in a fits file.

Input parameters:
- stars: object identifiers, preferentially in the HD catalogue. Must be in a list format
- instruments: name of the spectrograph to retrieve the spectra from. Must be in a list format
- columns_df: columns to include in the final data frame. Must be in a list format
- indices: spectral lines to compute the activity indices. Must be in a list format
- max_spectra: maximum number of spectra to be downloaded
- min_snr: minimum SNR to select the spectra. Must be in a float-like format
- download: boolean to decide if the spectra are downloaded or not (if not, it is assumed that the spectra is already inside the folders where 
the pipeline would download the spectra to)
- neglect_data: spectra to be manually neglected for whatever reason
- username_eso: username in the ESO data base to download the data

Returns:
- creates a folder teste_download_rv_corr/{star} where all the produced files are stored
- in the subfolder ADP/ the corrected fits files of the spectra are stored
- {star}_{instrument}_{line}.pdf: plot of the spectra in said line (CaI, CaII H, CaII K, H$\alpha$, HeI, NaID1, NaID2 and FeII)
- stats_{star}.csv: csv file with a data frame including statistical information like the star name, the indice name and the respective maximum, minimum, mean, median, 
standard deviation, time span and number of spectra used for that indice
- df_{star}_{instrument}.csv: csv file with the data frame including the columns_df given as input
- {star}_GLS.pdf and {star}_WF.pdf: plots of the periodogram of the CaII H&K indice and the respective Window Function
- df_stats_{star}.fits: FITS file that includes the data frame given by df_{star}_{instrument}.csv in a BINTable format and includes the statistics given in stats_{star}.csv
and informations regarding the periodogram in the header
- report_periodogram_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics
- flag_ratios_{instrument}.txt: adds to a txt file the names of the badly corrected spectra and the RV flag ratio

Notes:
- to run the RV correction a file containing a reference spectra is needed. I recommend using the Sun spectrum given along this pipeline, because the reading of this spectra is 
adapted to it
'''
import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import time

from astroquery.eso import Eso # type: ignore
from astropy.io import fits
from astroquery.simbad import Simbad # type: ignore
from util_funcs_1 import (get_gaiadr3,choose_snr,check_downloaded_data,select_best_spectra)
from util_funcs_2 import (
    read_fits,plot_line,
    get_rv_ccf,correct_spec_rv,flag_ratio_RV_corr,
    stats_indice,plot_RV_indices,
    sigma_clip,gls_periodogram,
    general_fits_file)

from actin2 import ACTIN # type: ignore

actin = ACTIN()

# Setup logging logger
global logger
eso = Eso()

def get_adp_spec(
    eso,
    search_name,name_target,
    neglect_data,
    instrument="HARPS",
    min_snr=10,max_snr=550,
    box=0.07,
    path_download_base="tmpdir_download4/",
    max_spectra=250,
):
    """
    Downloads and processes spectra from ESO database.

    Parameters:
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

    tbl_search = eso.query_surveys(instrument, target=search_name, box=box)

    if tbl_search is None:
        return "None"

    if instrument == "UVES":  # to ensure that at least the Halpha line is included
        wav_lim_min = np.array([int(float(wav.split("..")[0])) * 10 for wav in tbl_search["Wavelength"]])
        wav_lim_max = np.array([int(float(wav.split("..")[1])) * 10 for wav in tbl_search["Wavelength"]])

        ind_wav_lim = np.where((wav_lim_min < 6400) & (wav_lim_max > 6600))
        tbl_search = tbl_search[ind_wav_lim]

    # Removing eventually detected manually bad data
    if name_target in neglect_data.keys():
        i_clean = [i for i in range(len(tbl_search)) if tbl_search["ARCFILE"][i] not in neglect_data[name_target]]
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

    # This part is to get the ancillary data, the original s1D and CCF files
    if instrument == "HARPS":
        tbl_adp = list(tbl_search["ARCFILE"])
        tbl_ret = []
        for f in tbl_adp:
            fs = f.split(":")
            x = (float(fs[-1]) * 1000.0 + 1.0) / 1000.0
            f_new = fs[0] + ":" + fs[1] + ":" + f"{x:06.3f}"
            tbl_ret.append(f)
            tbl_ret.append(f_new)
        eso.retrieve_data(tbl_ret, destination=path_download + "ADP/")
    else:
        # pass
        table = tbl_search["ARCFILE"]
        eso.retrieve_data(table, destination=path_download + "ADP/")

    #snr_arr = check_downloaded_data(path_download + "ADP/")

    return paths_download


def pipeline(stars, instruments, 
             columns_df, indices, 
             max_spectra, min_snr,
             download, neglect_data, username_eso):

    max_snr_instr = {"HARPS": 550,"ESPRESSO": 1000,"FEROS": 1000,"UVES": 550}  # max SNR to be used to avoid saturation

    # to login into the ESO data base
    eso.login(username_eso)
    eso.ROW_LIMIT = -1

    for target_save_name in stars:

        gaiadr3 = get_gaiadr3(target_save_name)

        if gaiadr3 == -1:
            print("GAIA DR3 Not found automatically:")
            gaiadr3 = input("Insert GAIADR3 > ")
            target_search_name = target_save_name
        else:
            target_search_name = "GAIA DR3 " + gaiadr3

        print(target_save_name, target_search_name)

        # creates folders for the download and then the correct spectra
        if not os.path.isdir("teste_download"):
            os.mkdir("teste_download")
        star_folder = f"teste_download/{target_save_name}/"
        if not os.path.isdir(star_folder):
            os.mkdir(star_folder)

        if not os.path.isdir("teste_download_rv_corr"):
            os.mkdir("teste_download_rv_corr")
        star_folder_rv = f"teste_download_rv_corr/{target_save_name}/"
        if not os.path.isdir(star_folder_rv):
            os.mkdir(star_folder_rv)

        for instr in instruments:

            print(f"Processing spectra from {instr} instrument...")
            df = pd.DataFrame(columns=columns_df)

            max_snr = max_snr_instr[instr]
            if download == True:
                print(f"Downloading spectra from {instr} instrument...")

                logger = logging.getLogger("individual Run")
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler("test.log", mode="w")
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    "%(asctime)s : %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
                )
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logger.info("Started")
                fh.close()

                path_downloads = get_adp_spec(
                    eso,
                    target_search_name,target_save_name,
                    neglect_data,
                    min_snr=min_snr,max_snr=max_snr,
                    instrument=instr,
                    path_download_base=star_folder,
                    max_spectra=max_spectra)

                if path_downloads == None:
                    pass  # skip to next instrument if there are no observations with the present one

            # rounding up the spectra downloaded
            files = glob.glob(os.path.join(f"teste_download/{target_save_name}/{target_save_name}_{instr}/ADP","ADP*.fits"))
            if len(files) == 0:
                continue
            files_tqdm = tqdm.tqdm(files) #to make a progress bar
            sun_template_wv, sun_template_flux, sun_template_flux_err, sun_header = read_fits(
                file_name="Sun1000.fits", instrument=None, mode=None
            )  # template spectrum for RV correction

            # creates folders to save the rv corrected fits files
            folder_path = (f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/")
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                os.mkdir(folder_path + "ADP/")

            data_array = []  # to plot the lines later
            drv = 0.1  # step to search rv

            for file in files_tqdm:
                time.sleep(0.01)

                wv, f, f_err, hdr = read_fits(file, instr, mode="raw")
                # value = negative_flux_treatment(f, method="skip") #method can be "zero_pad"

                bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(
                    star = target_save_name,
                    stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                    template_hdr = sun_header, template_spec=sun_template_flux,
                    drv = drv, units = "m/s",
                    instrument = instr)
                #radial_velocity is the RV value, rv is the array of radial velocities given by the CCF
                wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units="m/s")

                #creating the fits file of the corrected spectra
                data = np.vstack((wv_corr, f, f_err))
                hdu = fits.PrimaryHDU(data, header=hdr)
                hdul = fits.HDUList([hdu])
                file_path = file.replace("teste_download", "teste_download_rv_corr")
                hdul.writeto(f"{file_path}", overwrite=True)
                hdul.close()

                data_array.append(data)

                # run ACTIN2
                #this part must be corrected, but is included to avoid the case where the first points of flux_error array is zero
                if f_err[10000] != 0:
                    spectrum = dict(wave=wv_corr, flux=f, flux_err=f_err)
                else:
                    spectrum = dict(wave=wv_corr, flux=f)
                SNR = hdr["SNR"]
                # flag for goodness of RV correction
                flag_ratio, flag_list = flag_ratio_RV_corr([file_path], instr)
                flag = flag_list[0]

                headers = {"bjd": bjd,"file": file,"instr": instr,"rv": radial_velocity,"obj": target_save_name,"SNR": SNR,"RV_flag": flag}
                df_ind = actin.CalcIndices(spectrum, headers, indices).indices
                df = df.append(pd.DataFrame([{**df_ind, **headers}]),ignore_index=True,sort=True)

            df.to_csv(folder_path + f"df_{target_save_name}_{instr}.csv", index=False)  # save before sigma clip

            # flag for goodness of RV correction
            flag_col = np.array(df["RV_flag"])
            good_spec_ind = np.where(flag_col == 0)
            N_good_spec = len(flag_col[good_spec_ind]) #number of well corrected spectra
            flag_rv_ratio = N_good_spec / len(flag_col)
            bad_spec_indices = np.where(flag_col == 1)[0]
            df = df.drop(labels=list(bad_spec_indices)) #dropping the badly corrected spectra

            with open(f"flag_ratios_{instr}.txt", "w") as f:  #write in a text file the names of the badly corrected spectra and RV quality indicator flag
                f.write("##########################\n")
                f.write(f"Star: {target_save_name}\n")
                f.write(f"Flag ratio: {flag_rv_ratio}\n")
                if flag_rv_ratio != 1:
                    for index in bad_spec_indices:
                        f.write(f"Bad spectrum: {files[index]}\n")

            # plot the wavelength of known lines to check if RV correction is good
            for line in ["Ha", "CaIIH", "CaIIK", "FeII", "NaID1", "NaID2", "HeI", "CaI"]:
                plt.figure(2)
                plot_line(data=data_array, line=line, line_color=None)
                plt.savefig(folder_path+ f"{target_save_name}_{instr}_{line}.pdf",bbox_inches="tight",)
                plt.clf()

            if flag_rv_ratio > 0:  # if there is at least one spectrum nicely corrected
                cols = ["I_CaII", "I_Ha06", "I_NaI", "rv"]
                if len(df) > 10:  # perform a sigma-clip
                    df = sigma_clip(df, cols, sigma=3)

                plt.figure(3)
                plot_RV_indices(target_save_name,df,indices,save=True,path_save=folder_path + f"{target_save_name}_{instr}.pdf")

                ind_I_CaII_Rneg = df[df["I_CaII_Rneg"] < 0.01].index  # if there are a lot of zero fluxes in the window (>1%), discard
                I_CaII_extracted = df.loc[ind_I_CaII_Rneg, "I_CaII"]
                I_CaII_err_extracted = df.loc[ind_I_CaII_Rneg, "I_CaII_err"]
                bjd_extracted = df.loc[ind_I_CaII_Rneg, "bjd"]
                try:
                    t_span = max(bjd_extracted) - min(bjd_extracted)
                    n_spec = len(I_CaII_err_extracted)
                except:
                    n_spec = 0

                if n_spec >= 30 and t_span >= 2 * 365:
                    # only compute periodogram if star has at least 30 spectra in a time span of at least 2 years
                    period, period_err, flag_period = gls_periodogram(
                        target_save_name,
                        I_CaII_extracted, I_CaII_err_extracted,
                        bjd_extracted,
                        print_info=False,
                        mode="Period",
                        save=True, path_save=folder_path + f"{target_save_name}_GLS.pdf")
                    plt.clf()
                    print(f"Period of I_CaII: {period} +/- {period_err} days")
                    print(f"Flag of periodogram: {flag_period}")
                else:
                    period = 0
                    period_err = 0
                    flag_period = "white"

                stats_df = stats_indice(target_save_name, cols, df) #computes the statistics
                print(stats_df)
                stats_df.to_csv(folder_path + f"stats_{target_save_name}.csv")

                file_path = folder_path + f"df_stats_{target_save_name}.fits" #save into the final fits file
                general_fits_file(
                    stats_df, df,
                    file_path,
                    min_snr, max_snr,
                    instr,
                    period, period_err, flag_period,
                    flag_rv_ratio)

                # shutil.rmtree(f"teste_download/{target_save_name}/")  # remove original fits files


### Main program:
def main(stars, instruments, columns_df, indices, max_spectra, min_snr, download, neglect_data, username_eso):

    pipeline(stars, instruments, columns_df, indices, max_spectra, min_snr, download, neglect_data, username_eso)

if __name__ == "__main__":

    stars = ["HD209100","HD160691","HD115617","HD46375","HD22049","HD102365","HD1461","HD16417","HD10647","HD13445","HD142A",
        "HD108147","HD16141","HD179949","HD47536","HD20794","HD85512","HD192310"]
    #stars = ["HD47536"]
    instruments = ["UVES"]
    columns_df = [
        "I_CaII","I_CaII_err","I_CaII_Rneg",
        "I_Ha06","I_Ha06_err","I_Ha06_Rneg",
        "I_NaI","I_NaI_err","I_NaI_Rneg",
        "bjd","file","instr","rv","obj","SNR"]  # for df
    indices = ["I_CaII", "I_Ha06", "I_NaI"]  # indices for activity
    
    max_spectra = 150 # maximum spectra to be selected
    min_snr = 15 #minimum overall SNR

    download = True #download from ESO database?

    username_eso = "telmonteiro"

    # specific data to neglect for any reason
    neglect_data = {"HD20794": "ADP.2020-08-10T15:36:05.310"}

    main(stars, instruments, columns_df, indices, max_spectra, min_snr, download, neglect_data, username_eso)