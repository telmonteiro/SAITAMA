import os, glob, logging, tarfile, numpy as np, matplotlib.pyplot as plt, pandas as pd, tqdm, time, shutil
from astroquery.eso import Eso
from astropy.io import fits
from astroquery.simbad import Simbad
from util_funcs import get_gaiadr3, choose_snr, check_downloaded_data, get_rv_ccf, read_fits, correct_spec_rv, general_fits_file, flag_ratio_RV_corr
from util_funcs import stats_indice, plot_RV_indices, sigma_clip, plot_line, select_best_spectra, line_ratio_indice, gls_periodogram
from actin2.actin2 import ACTIN
actin = ACTIN()

# Setup logging logger
global logger

eso = Eso()
username_eso = "telmonteiro"
eso.login(username_eso)
eso.ROW_LIMIT = -1

neglect_data = {"HD20794":"ADP.2020-08-10T15:36:05.310"}

def get_adp_spec(eso, search_name, name_target, neglect_data, instrument="HARPS", min_snr=10, max_snr=550, box= 0.07, path_download_base="tmpdir_download4/", max_spectra=250):
    """
    Starts by quering in ESO data base the object and the instrument. Then checks if there is data to neglect.
    Then it cuts the data sets by the minimum SNR chosen. From that it chooses the best SNR spectra that cover a big time span.
    Checks if the folder for the instrument exists or not.
    Then according to the instrument (for now only HARPS is defined) it retrieves the data from ESO and untars the ancillary data.
    For non-HARPS instruments, it just retrieves the data in a simple way.
    Then corrects the spectra by RV and runs ACTIN2. Saves plots of the spectra in lines of interest to check if everything is alright, as well
    as plots of the activity indices and the statistics related to them.
    """
    name_target = name_target.replace(" ","")
    print("Searching: ", search_name)

    tbl_search = eso.query_surveys(instrument, target=search_name, box=box)

    if tbl_search is None:
        return "None"
    
    if instrument == "UVES":
        wav_lim_min = np.array([int(float(wav.split("..")[0]))*10 for wav in tbl_search["Wavelength"]])
        wav_lim_max = np.array([int(float(wav.split("..")[1]))*10 for wav in tbl_search["Wavelength"]])
        ind_wav_lim = np.where((wav_lim_min < 6400) & (wav_lim_max > 6600))
        tbl_search = tbl_search[ind_wav_lim]

    # Removing eventually detected manually bad data
    if name_target in neglect_data.keys():
        i_clean = [i for i in range(len(tbl_search)) if tbl_search['ARCFILE'][i] not in neglect_data[name_target]]
        tbl_search = tbl_search[i_clean]

    print("Found %d datasets." % (len(tbl_search)))

    paths_download = []

    #cuts the lowest SNR by a minimum
    icut = choose_snr(tbl_search['SNR'], min_snr = min_snr, max_snr = max_snr)
    if len(icut[0])<1:
        return "None"
    tbl_search = tbl_search[icut]
    print(f"{len(tbl_search)} datasets obey the SNR cut")

    #selects the best SNR spectra while keeping a time span significant
    tbl_search = select_best_spectra(tbl_search, max_spectra=max_spectra)
    print(tbl_search)
    print(f"{len(tbl_search)} datasets obey the selection process")

    #Download
    path_download = path_download_base+name_target+'_'+instrument+"/"
    if not os.path.isdir(path_download):
        os.mkdir(path_download)
        os.mkdir(path_download+"ADP/")
        paths_download.append(path_download)

    # This part is to get the ancillary data, the original s1D and CCF files
    if instrument == "HARPS":
        tbl_adp = list(tbl_search['ARCFILE'])
        tbl_ret = []
        for f in tbl_adp:
            fs = f.split(":")
            x = (float(fs[-1])*1000.+1.)/1000.
            f_new = fs[0]+":"+fs[1]+":"+f"{x:06.3f}"
            tbl_ret.append(f)
            tbl_ret.append(f_new)
        eso.retrieve_data(tbl_ret, destination = path_download+"ADP/")
        #untar_ancillary_harps(path_download+"ADP/")
    else:
        #pass
        table = tbl_search['ARCFILE']
        eso.retrieve_data(table, destination = path_download+"ADP/")

    snr_arr = check_downloaded_data(path_download+"ADP/")

    return paths_download

#stars = ["HD20794","HD85512","HD192310"]

stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
        'HD20794',"HD85512","HD192310"]

instruments = ["HARPS","ESPRESSO","UVES"] # the problem is with BJD in UVES
columns = ['I_CaII', 'I_CaII_err', 'I_CaII_Rneg', 'I_Ha06', 'I_Ha06_err', 'I_Ha06_Rneg', 'I_NaI', 'I_NaI_err', 'I_NaI_Rneg',
           'bjd', 'file', 'instr', 'rv', 'obj', 'SNR'] #for df
indices= ['I_CaII', 'I_Ha06', 'I_NaI'] #indices for activity
max_snr_instr = {"HARPS":550,"ESPRESSO":1000,"FEROS":1000,"UVES":550} #max SNR to be used to avoid saturation

#maximum spectra to be selected, as well as the minimum overall SNR
max_spectra = 150
min_snr = 15

download = True

### Main program:
def main():

    for target_save_name in stars:

        gaiadr3 = get_gaiadr3(target_save_name)
    
        if gaiadr3 == -1:
            print("GAIA DR3 Not found automatically:")
            gaiadr3 = input('Insert GAIADR3 > ')
            target_search_name = target_save_name
        else:
            target_search_name = "GAIA DR3 " + gaiadr3

        print(target_save_name, target_search_name)

        #creates folders for the download and then the correct spectra
        star_folder = f"teste_download/{target_save_name}/"
        if not os.path.isdir(star_folder):
            os.mkdir(star_folder)
        star_folder_rv = f"teste_download_rv_corr/{target_save_name}/"
        if not os.path.isdir(star_folder_rv):
            os.mkdir(star_folder_rv)

        for instr in instruments:
            print(f"Downloading and processing spectra from {instr} instrument")
            df = pd.DataFrame(columns=columns)

            max_snr = max_snr_instr[instr]
            if download == True:

                logger = logging.getLogger("individual Run")
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler("test.log", mode='w')
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s : %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logger.info('Started')
                fh.close()
                
                path_downloads = get_adp_spec(eso,target_search_name, target_save_name, neglect_data, min_snr=min_snr, max_snr=max_snr, 
                                                  instrument=instr, path_download_base=star_folder, max_spectra=max_spectra)
                
                if path_downloads == None:
                    pass #skip to next instrument if there are no observations with the present one

            files = glob.glob(os.path.join(f"teste_download/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))
            if len(files) == 0:
                continue
            files_tqdm = tqdm.tqdm(files)
            sun_template_wv, sun_template_flux, sun_template_flux_err, sun_header = read_fits(file_name="Sun1000.fits",instrument=None, mode=None) #template spectrum for RV correction
            
            #creates folders to save the rv corrected fits files
            folder_path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                os.mkdir(folder_path+"ADP/")

            data_array = [] #to plot the lines later, if too many spectra may cause memory problems
            drv = .1 #step to search rv

            for i,file in enumerate(files_tqdm):
                time.sleep(0.01)

                wv, f, f_err, hdr = read_fits(file,instr,mode="raw")

                #value = negative_flux_treatment(f, method="skip") #method can be "zero_pad"

                bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star = target_save_name, stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                                                        template_hdr = sun_header, template_spec = sun_template_flux, 
                                                        drv = drv, units = "m/s", instrument=instr)
                
                wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")

                data = np.vstack((wv_corr, f, f_err))
                hdu = fits.PrimaryHDU(data, header=hdr)
                hdul = fits.HDUList([hdu])
                file_path = file.replace('teste_download', 'teste_download_rv_corr')
                hdul.writeto(f"{file_path}", overwrite=True)
                hdul.close()

                data_array.append(data)

                #run ACTIN2
                if f_err[0] != 0:
                    spectrum = dict(wave=wv_corr, flux=f, flux_err = f_err)
                else: 
                    spectrum = dict(wave=wv_corr, flux=f)
                SNR = hdr["SNR"]
                #flag for goodness of RV correction  
                flag_ratio, flag_list = flag_ratio_RV_corr([file_path],instr)
                flag = flag_list[0]

                headers = {"bjd":bjd,"file":file,"instr":instr,"rv":radial_velocity,"obj":target_save_name,"SNR":SNR, "RV_flag":flag}
                df_ind = actin.CalcIndices(spectrum, headers, indices).indices
                df = df.append(pd.DataFrame([{**df_ind, **headers}]), ignore_index=True,sort=True)
                          
            #print(df)
            df.to_csv(folder_path+f"df_{target_save_name}_{instr}.csv",index=False) #save before sigma clip

            #flag for goodness of RV correction     
            flag_col = np.array(df["RV_flag"])
            good_spec_ind = np.where((flag_col == 0))
            N_good_spec = len(flag_col[good_spec_ind])
            flag_rv_ratio = N_good_spec / len(flag_col)  
            bad_spec_indices = np.where(flag_col == 1)[0]
            df = df.drop(labels=list(bad_spec_indices))
            #print(df)
            #print(len(df["RV_flag"]))
            with open(f"flag_ratios_{instr}.txt", "a") as f: # Write to a text file
                f.write("##########################\n")
                f.write(f"Star: {target_save_name}\n")
                f.write(f"Flag ratio: {flag_rv_ratio}\n")
                if flag_rv_ratio != 1:
                    for index in bad_spec_indices:
                        f.write(f"Bad spectrum: {files[index]}\n")

            #plot the wavelength of known lines to check if RV correction is good
            for line in ["Ha","CaIIH","CaIIK","FeII","NaID1","NaID2","HeI","CaI"]:
                plt.figure(2)
                plot_line(data=data_array, line=line)
                plt.savefig(folder_path+f"{target_save_name}_{instr}_{line}.pdf", bbox_inches="tight")
                plt.clf()  

            if flag_rv_ratio > 0:
                #print(df)
                cols = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
                df = sigma_clip(df, cols, sigma=3)

                plt.figure(3)
                plot_RV_indices(target_save_name, df, indices, save=True, 
                path_save = folder_path+f"{target_save_name}_{instr}.pdf")
                #plt.show()

                ind_I_CaII_Rneg = df[df["I_CaII_Rneg"] < 0.01].index
                I_CaII_extracted = df.loc[ind_I_CaII_Rneg, "I_CaII"]
                I_CaII_err_extracted = df.loc[ind_I_CaII_Rneg, "I_CaII_err"]
                bjd_extracted = df.loc[ind_I_CaII_Rneg, "bjd"]
                try:
                    t_span = max(bjd_extracted)-min(bjd_extracted)
                    n_spec = len(I_CaII_err_extracted)
                except:
                    n_spec = 0
                if n_spec >= 30 and t_span >= 2*365:  #only compute periodogram if star has at least 30 spectra in a time span of at least 2 years
                    period, period_err, flag_period = gls_periodogram(target_save_name, I_CaII_extracted, I_CaII_err_extracted, bjd_extracted, 
                                                        print_info = False, mode = "Period",
                                            save=True, path_save=folder_path+f"{target_save_name}_GLS.pdf")
                    plt.clf()
                    print(f"Period of I_CaII: {period} +/- {period_err} days")
                    print(f"Flag of periodogram: {flag_period}")
                else: period = 0; period_err = 0; flag_period = "white"

                stats_df = stats_indice(target_save_name,cols,df)
                print(stats_df)
                stats_df.to_csv(folder_path+f"stats_{target_save_name}.csv")          

                file_path = folder_path+f"df_stats_{target_save_name}.fits"
                general_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err, flag_period, flag_rv_ratio)
                
                #shutil.rmtree(f"teste_download/{target_save_name}/")  #remove original fits files

if __name__ == "__main__":
    main()

'''
Problems with the program:
- to compute the rotation period:
        - if many points and spread in time: GLS periodogram
        - if few points: use calibration. convert I_CaII to log Rhk and use the mean. 
                                          get Teff and [Fe/H] from Sweet-Cat to compute B-V 
                                          through another calibration (may not work well with M stars) (alternative: get from GAIA)
- with rotation period and gyrochronology the age

- ESPRESSO and UVES spectrographs are not yet configured
- some way of reducing running time, as well as supressing verbose of ESO download
'''