from general_funcs import read_bintable
from astropy.io import fits
import pandas as pd
import numpy as np
import os
import math
from astropy.table import Table
from periodogram_funcs import (gls)
from general_funcs import (stats_indice, sigma_clip)
from periodogram_funcs import get_report_periodogram
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)

def read_and_merge_fits_files(stars, instruments, base_folder="teste_download_rv_corr"):
    """
    Reads each FITS file for each instrument and joins them in a master FITS file,
    adding a column that indicates the instrument used and updating the headers.
    
    Parameters:
    stars: list of star names
    instruments: list of instrument names
    base_folder: base folder where the FITS files are stored
    
    Returns:
    None
    """
    
    for star_id in stars:
        # Initialize an empty DataFrame to hold all the data for the current star
        master_df = pd.DataFrame()
        master_header = fits.Header() 
        hdulist = [fits.PrimaryHDU()]  # Initialize HDU list with a primary HDU

        list_instruments = []
        for instr in instruments:
            folder_path = os.path.join(base_folder, f"{star_id}/{star_id}_{instr}/")
            file_path = folder_path + f"df_stats_{star_id}.fits"
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            list_instruments.append(instr)

            df, hdr = read_bintable(file_path)

            master_df = pd.concat([master_df, df], ignore_index=True)
            print(master_df["instr"])

            new_header = {}
            hdr_keys_to_use = ["TIME_SPAN","SNR_MIN","SNR_MAX","PERIOD_I_CaII","PERIOD_I_CaII_ERR","FLAG_PERIOD","FLAG_RV"]
            for key in hdr_keys_to_use:
                new_header[f"{key}_{instr}"] = hdr[key]
                
            # Save updated headers in the master header
            master_header.update(new_header)

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

        cols = ["I_CaII", "I_Ha06", "I_NaI", "rv"]
        if len(master_df) > 10:  # perform a sigma-clip
            master_df = sigma_clip(master_df, cols, sigma=3)

        I_CaII = master_df["I_CaII"]; I_CaII_err = master_df["I_CaII_err"]; bjd = master_df["bjd"]
        try:
            t_span = max(bjd) - min(bjd)
            n_spec_CaII = len(master_df["I_CaII"].dropna())
        except:
            n_spec_CaII = 0
        if n_spec_CaII >= 30 and t_span >= 2 * 365 and math.isnan(I_CaII[0]) == False:
            # only compute periodogram if star has at least 30 spectra in a time span of at least 2 years
            #results, gaps, flag, period, period_err, harmonics_list, amplitude, amplitude_err = gls(star_id, instr, bjd-2450000, I_CaII, y_err=I_CaII_err, 
            #                                                                        pmin=1.5, pmax=1e4, steps=1e5, print_info = False, save=True, folder_path=os.path.join(base_folder, f"{star_id}"))
            #report_periodogram = get_report_periodogram(hdr,gaps,period,period_err,amplitude,amplitude_err,flag_period,harmonics_list,folder_path=os.path.join(base_folder, f"{star_id}"))
            snr_min = np.min(master_df["SNR"]); snr_max = np.max(master_df["SNR"])
            #print(f"Period of I_CaII: {period} +/- {period_err} days")
            #print(f"Flag of periodogram: {flag_period}")
        else:
            period = 0
            period_err = 0
            flag_period = "white"

        cols += ["S_MW","log_Rhk","prot_n84","prot_m08","age_m08","age_m08_err"]
        stats_master_df = stats_indice(star_id, cols, master_df) #computes the statistics. include errors or not?

        # Ensure all columns are compatible with FITS
        for col in master_df.select_dtypes(include=['object']).columns:
            master_df[col] = master_df[col].astype(str)

        list_instruments_str = ""
        for ins in list_instruments:
            list_instruments_str += ins + ","

        dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                    "INSTRUMENTS":[list_instruments_str,"Instruments used"],
                    "TIME_SPAN":[t_span, 'Time span in days between first and last observations used'],
                    #"N_SPECTRA":[N_spectra,"Number of spectra used"],
                    "SNR_MIN":[snr_min,"Minimum SNR"],
                    "SNR_MAX":[snr_max,"Maximum SNR"],
                    #"PERIOD_I_CaII":[period,"Period of CaII activity index"],
                    #"PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
                    #"FLAG_PERIOD":[flag_period,"Goodness of periodogram fit flag. Color based."],
                    "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                    "COMMENT1":["RV obtained from CCF measure (m/s)","Comment"],
                    "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}

        indices = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv', "S_MW", "log_Rhk", "prot_n84", "prot_m08"]#,"age_m08", "age_m08_err"]
        stats = ["max","min","mean","median","std","N_spectra"]

        for i,ind in enumerate(indices):
            for col in stats:
                stat = stats_master_df[col][i]
                if math.isnan(float(stat)): stat = 0
                if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
                elif col == "N_spectra": comment = f"Nr of spectra used in {ind}"
                else: comment = f"{col} of {ind}"
                dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]
        
        for keyword in dict_hdr.keys():
            master_header.append(("HIERARCH "+keyword+"_ALL", dict_hdr[keyword][0], dict_hdr[keyword][1]), end=True)
        
        print(master_df["instr"])

        master_table = Table.from_pandas(master_df)
        master_hdu = fits.BinTableHDU(data=master_table, header=master_header, name="MASTER")
        
        # Add the master HDU to the HDU list
        hdulist.append(master_hdu)

        # Write all HDUs to a single FITS file
        output_file = os.path.join(base_folder, f"{star_id}/master_df_{star_id}.fits")
        fits.HDUList(hdulist).writeto(output_file, overwrite=True)
        print(f"Master FITS file for {star_id} created successfully.")
    

# Example usage
stars = ["HD209100"]
instruments = ["HARPS", "ESPRESSO", "UVES"]

read_and_merge_fits_files(stars, instruments)

# Reading the merged FITS file
hdul = fits.open("teste_download_rv_corr/HD209100/master_df_HD209100.fits")
hdul.info()
