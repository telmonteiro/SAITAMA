import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os, glob
from util_funcs import read_bintable

stars = ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
         'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536',
         "HD20794","HD85512","HD192310"]

keys = ['TIME_SPAN', 'N_SPECTRA', 'SNR_MIN', 'SNR_MAX', 'FLAG_RV', 'PERIOD_I_CaII', 'PERIOD_I_CaII_ERR', 'FLAG_PERIOD',
        'I_CAII_MAX', 'I_CAII_MIN', 'I_CAII_MEAN', 'I_CAII_MEDIAN', 'I_CAII_STD', "I_CAII_N_SPECTRA", 
        'I_HA06_MAX', 'I_HA06_MIN', 'I_HA06_MEAN', 'I_HA06_MEDIAN', 'I_HA06_STD', "I_HA06_N_SPECTRA",
        "RV_MEAN"] #'STAR_ID','INSTR', 'I_NAI_MAX', 'I_NAI_MIN', 'I_NAI_MEAN', 'I_NAI_MEDIAN', 'I_NAI_STD', 'RV_MAX', 'RV_MIN', 'RV_MEAN','RV_MEDIAN','RV_STD'

# Create an empty DataFrame to store the results
final_df = pd.DataFrame()

for star in stars:
    file = glob.glob(os.path.join(f"teste_download_rv_corr/{star}/{star}_ESPRESSO/", f"df_stats_{star}.fits"))
    if file == []: continue
    print(file)
    df, hdr = read_bintable(file[0], print_info=False)

    values = {}
    for key, value in hdr.items():
        if key in keys:
            if type(value) != str:
                values[key] = round(value,5)
            else: values[key] = value

    # Convert the dictionary to a DataFrame and transpose it
    star_df = pd.DataFrame(values, index=[star])
    
    # Append the transposed DataFrame to the final DataFrame
    final_df = pd.concat([final_df, star_df], sort=False)

# Reset the index to make "star" a column
final_df.reset_index(inplace=True)
final_df.rename(columns={"index": "STAR_ID"}, inplace=True)
print(final_df)

final_df.to_csv("df_15_stars_stats_ESPRESSO.csv")

'''
stars where GLS periodogram is not reliable: 
HD160691: period too low, data is not periodic
HD46375: period too low, data is well spread and majority of points are in a short time span
HD22049: period too high, spread of points is not good
HD102365: spread of points is not well defined
HD13445: period too low, spread is not defined enough, but there peaks in more believable periods
HD142A: no peak is higher than FAP
HD108147: spread is not good enough so there are a ton of peaks
HD16141: spread of points is not well defined
HD179949: not enough good spectra to compute GLS. remark: CaI line is a caos but Ha and CaII H&K seem ok
HD47536: not enough spectra overall

stars where GLS seem ok:
HD209100
HD115617
HD1461
HD16417
HD10647

Some remarks:
- in HD179949 the CaI line is a caos so most of the spectra were discarded by the indicator. Nevertheless, in the lines where activity
is computed the spectra seem ok...
- some constraints should be made for the computed period, for example must be higher than the FAP levels (HD142A), and a realistic
value should be attained (periods lower than 1.5 years (HD160691, HD46375, HD13445, HD16141) or higher than say 100 years (HD22049))
- the SNR-time span trade-off choice that is made before the download of the spectra should be rethinked in order to consider if
there is too many spectra in the space of one day. for example, if a given day has 15 spectra that have good SNR, they may be chosen
in detriment of good time span, inserting a bias in the time distribution of spectra and leading to no good period computed
(HD108147, HD102365, for example)

For UVES:
Only HD13445 didnt have all spectra negged, and the values of I_Ha06 and I_NaI seem consistent with the HARPS values...
very weird that for the other stars ALL the spectra gave RV_Flag = 1

'''