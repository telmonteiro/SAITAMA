import os, glob, logging, tarfile, numpy as np, matplotlib.pyplot as plt, pandas as pd, tqdm, time
from astroquery.eso import Eso
from astropy.io import fits
from astroquery.simbad import Simbad
from util_funcs import get_gaiadr3, choose_snr, untar_ancillary_harps, check_downloaded_data, get_rv_ccf, read_fits, correct_spec_rv 
from util_funcs import stats_indice, plot_RV_indices, sigma_clip, plot_line, select_best_spectra

import gc
gc.collect()

from actin2.actin2 import ACTIN
actin = ACTIN()

# Setup logging logger
global logger

eso = Eso()
username_eso = "telmonteiro"
eso.login(username_eso)
eso.ROW_LIMIT = -1

neglect_data = {"HD20794":["ADP.2014-09-18T12:17:43.923","ADP.2014-10-06T10:03:52.550","ADP.2014-12-18T22:55:22.517",
                           "ADP.2014-10-06T10:06:35.710","ADP.2014-12-18T22:55:22.773"]} # espetros todos cegos

def get_adp_spec(search_name, name_target, neglect_data, instrument="HARPS", min_snr=10, box= 0.07, path_download_base="tmpdir_download4/", max_spectra=250):
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

    # Removing eventually detected manually bad data
    if name_target in neglect_data.keys():
        i_clean = [i for i in range(len(tbl_search)) if tbl_search['ARCFILE'][i] not in neglect_data[name_target]]
        tbl_search = tbl_search[i_clean]

    print("Found %d datasets." % (len(tbl_search)))

    paths_download = []

    #cuts the lowest SNR by a minimum
    icut = choose_snr(tbl_search['SNR'], min_snr = min_snr)
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

stars = ["HD20794", "HD85512", "HD192310"]
instruments = ["HARPS"] # ["HARPS","ESPRESSO","FEROS","UVES"]
columns = ['I_CaII', 'I_CaII_err', 'I_CaII_Rneg', 'I_Ha06', 'I_Ha06_err', 'I_Ha06_Rneg', 'I_NaI', 'I_NaI_err', 'I_NaI_Rneg',
           'bjd', 'file', 'instr', 'rv', 'obj', 'SNR'] #for df
indices= ['I_CaII', 'I_Ha06', 'I_NaI'] #indices for activity

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

        logger = logging.getLogger("individual Run")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler("test.log", mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s : %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info('Started')
        fh.close()

        #creates folders for the download and then the correct spectra
        star_folder = f"teste_download/{target_save_name}/"
        if not os.path.isdir(star_folder):
            os.mkdir(star_folder)
        star_folder_rv = f"teste_download_rv_corr/{target_save_name}/"
        if not os.path.isdir(star_folder_rv):
            os.mkdir(star_folder_rv)

        #maximum spectra to be selected, as well as the minimum overall SNR
        max_spectra = 150
        min_snr = 40

        for instr in instruments:
            print(f"Downloading and processing spectra from {instr} instrument")
            df = pd.DataFrame(columns=columns)

            download = True
            if download == True:
                if instr == "HARPS":
                    path_downloads = get_adp_spec(target_search_name, target_save_name, neglect_data, min_snr=min_snr, instrument=instr, 
                                                  path_download_base=star_folder, max_spectra=max_spectra)
                else: 
                    path_downloads = get_adp_spec(target_search_name, target_save_name, neglect_data, min_snr=min_snr, instrument=instr, 
                                                  path_download_base=star_folder, max_spectra=max_spectra)

            files = glob.glob(os.path.join(f"teste_download/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))
            files = tqdm.tqdm(files)
            sun_template_wv, sun_template_flux, sun_header = read_fits(file_name="Sun1000.fits",instrument=None) #template spectrum for RV correction
            
            #creates folders to save the rv corrected fits files
            folder_path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/"
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                os.mkdir(folder_path+"ADP/")

            data_array = [] #to plot the lines later, if too many spectra may cause memory problems
            drv = .1 #step to search rv

            for i,file in enumerate(files):
                time.sleep(0.01)

                wv, f, hdr = read_fits(file,instr)

                skip_0_flux = False #to do nothing and use the spectrum as it is just comment this line
                if skip_0_flux == True:
                    # skips the spectra that have negative fluxes           
                    neg_flux_position = np.where(f < 0)[0]
                    if neg_flux_position.shape[0] != 0:
                        print(f"File {file}, number {i}, has negative flux at positions {neg_flux_position}")
                        continue
                elif skip_0_flux == False:
                    #replace negative flux by zero flux
                    f = np.where(f < 0, 1e-5, f)

                bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star = target_save_name, stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                                                        template_hdr = sun_header, template_spec = sun_template_flux, 
                                                        drv = drv, units = "m/s", instrument=instr)
                
                wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")

                data = np.vstack((wv_corr, f))
                hdu = fits.PrimaryHDU(data, header=hdr)
                hdul = fits.HDUList([hdu])
                file_path = file.replace('teste_download', 'teste_download_rv_corr')
                hdul.writeto(f"{file_path}", overwrite=True)
                hdul.close()

                data_array.append(data)

                #run ACTIN2
                spectrum = dict(wave=wv_corr, flux=f)
                SNR = hdr["SNR"]
                headers = {"bjd":bjd,"file":file,"instr":instr,"rv":radial_velocity,"obj":target_save_name,"SNR":SNR}
                df_ind = actin.CalcIndices(spectrum, headers, indices).indices
                df = df.append(pd.DataFrame([{**df_ind, **headers}]), ignore_index=True,sort=True)

            #plot the wavelength of known lines to check if RV correction is good
            for line in ["Ha","CaIIH","CaIIK","FeII","NaID1","NaID2","HeI","CaI"]:
                plt.figure(1)
                plot_line(data=data_array, line=line)
                plt.savefig(folder_path+f"{target_save_name}_{instr}_{line}.png", bbox_inches="tight")
                plt.clf()
            
            cols = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
            df = sigma_clip(df, cols, sigma=3)

            print(df)
            df.to_csv(folder_path+f"df_{target_save_name}_{instr}.csv") #save after sigma clip

            plt.figure(2)
            plot_RV_indices(target_save_name, df, indices, save=True, 
            path_save = folder_path+f"{target_save_name}_{instr}.png")
            #plt.show()

            stats_df = stats_indice(target_save_name,cols,df)
            print(stats_df)
            stats_df.to_csv(folder_path+f"stats_{target_save_name}.csv")           

if __name__ == "__main__":
    main()

'''
Problems with the program:

- ESPRESSO and UVES spectrographs are not yet configured
- crashes if the number of spectra is too high because of the lines plot
- missing an automatic quality index, for now only way of checking bad spectra is by outliers and visual inspection of spectral lines plots
- maybe a better way of dealing with negative values in the flux, instead of just ignoring the spectra or replacing the point by 0
- some way of reducing running time, as well as supressing verbose of ESO download

- crashes if the star has no spectra by the instrument asked (future work)
- does not delete the downloaded spectra after processing, leading to shortage of memory when processing several stars (future work)
'''