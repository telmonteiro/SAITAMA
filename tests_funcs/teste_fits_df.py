import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from astropy.io import fits
from astropy.table import Table, vstack

def general_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]; #N_spectra = stats_df["N_total_spectra"][0]
    dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                "INSTR":[instr,"Instrument used"],
                "TIME_SPAN":[time_span, 'Time span in days between first and last observations used'],
                #"N__TOTAL_SPECTRA":[N_spectra,"Number of spectra used before indice calculation"],
                "SNR_MIN":[min_snr,"Minimum SNR"],
                "SNR_MAX":[max_snr,"Maximum SNR"],
                "PERIOD_I_CaII":[period,"Period of CaII activity index"],
                "PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
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

def read_bintable(file,print_info=False):
    hdul = fits.open(file)
    if print_info == True: hdul.info()
    hdr = hdul[1].header
    table = hdul[1].data
    astropy_table = Table(table)
    data_dict = astropy_table.to_pandas().to_dict()
    df = pd.DataFrame(data_dict)
    return df, hdr

target_save_name = "HD192310"
instr = "HARPS"
folder_path = f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/"
file_path = folder_path+f"df_stats_{target_save_name}.fits"

period = 0; period_err = 0; min_snr = 20; max_snr = 550

df = pd.read_csv(folder_path+f"df_{target_save_name}_{instr}.csv")
df = df.drop(columns=['Unnamed: 0']) #this should be solved in the main script

stats_df = pd.read_csv(folder_path+f"stats_{target_save_name}.csv")

general_fits_file(stats_df, df, file_path, min_snr, max_snr, instr, period, period_err)

df, hdr = read_bintable(file=file_path,print_info=False)
print(df)
print(df.columns)
print(hdr)