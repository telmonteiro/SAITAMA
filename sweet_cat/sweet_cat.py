import urllib, logging, pandas as pd, numpy as np, math
from astroquery.eso import Eso
from astroquery.simbad import Simbad
from util_funcs import get_gaiadr3, get_gaiadr2, choose_snr

#loading as a pandas dataframe:
sweetCat_table_url = "http://sweetcat.iastro.pt/catalog/SWEETCAT_Dataframe.csv"
dtype_SW = dtype={'gaia_dr2':'int64','gaia_dr3':'int64'}
SC = pd.read_csv(urllib.request.urlopen(sweetCat_table_url), dtype=dtype_SW)
#print(SC.columns)

# Extract the sign and the first two characters from the 'DEC' column and convert them to integers
SC['DEC_degrees'] = SC['DEC'].str[:3].astype(int)

# Filter the DataFrame based on the sign and 'DEC_degrees'
filtered_SC = SC[(SC['Vmag'] < 8) & ((SC['DEC'].str[0] == '-') | (SC['DEC_degrees'] < 30)) & (SC['SWFlag'] == 1)]
filtered_SC = filtered_SC.sort_values(by=["Teff"])

# Drop the temporary column 'DEC_degrees'
filtered_SC = filtered_SC.drop(columns=['DEC_degrees'])

#print(filtered_SC[["Name", "hd", "RA", "DEC", "Vmag", "Teff", "SWFlag"]])

stars_studied = ["20794","85512","192310"]
for star in stars_studied:
    print(filtered_SC[filtered_SC["hd"]==star][["Name", "hd", "RA", "DEC", "Vmag", "Teff", "SWFlag"]])

# Setup logging logger
global logger

eso = Eso()
username_eso = "telmonteiro"
eso.login(username_eso)
eso.ROW_LIMIT = -1
logger = logging.getLogger("individual Run")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("test.log", mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('Started')
fh.close()

stars = ["HD"+ str(x) for x in list(filtered_SC["hd"])]
instruments = ["HARPS", "UVES"]

# Add columns for storing number of spectra and SNR statistics for each instrument
filtered_SC['N_HARPS'] = 0
filtered_SC['SNR_MIN_HARPS'] = np.nan
filtered_SC['SNR_MAX_HARPS'] = np.nan
filtered_SC['N_UVES'] = 0
filtered_SC['SNR_MIN_UVES'] = np.nan
filtered_SC['SNR_MAX_UVES'] = np.nan

for _, row in filtered_SC.iterrows():
    target_nr_name = row["hd"]

    if target_nr_name != "nan":
        target_name = "HD" + str(target_nr_name)
    else:
        target_name = row["Name"]

    gaiadr3 = get_gaiadr3(target_name)

    if gaiadr3 == -1:
        gaiadr2 = get_gaiadr2(target_name)
        target_search_name = "GAIA DR2 " + str(gaiadr2) if gaiadr2 != -1 else target_name
    else:
        target_search_name = "GAIA DR3 " + str(gaiadr3)

    print(target_name, target_search_name)

    for instr in instruments:          

        tbl_search = eso.query_surveys(instr, target=target_search_name, box=0.07)

        try:
            min_snr = 15
            max_snr = 550
            icut = choose_snr(tbl_search['SNR'], min_snr = min_snr, max_snr = max_snr)
            tbl_search = tbl_search[icut]

            num_datasets = len(tbl_search)
            min_snr = tbl_search['SNR'].min()
            max_snr = tbl_search['SNR'].max()

            # Update filtered_SC with the statistics
            filtered_SC.loc[filtered_SC['hd'] == target_nr_name, f'N_{instr}'] = num_datasets
            filtered_SC.loc[filtered_SC['hd'] == target_nr_name, f'SNR_MIN_{instr}'] = min_snr
            filtered_SC.loc[filtered_SC['hd'] == target_nr_name, f'SNR_MAX_{instr}'] = max_snr

        except:
            continue

#for star in stars_studied:
#    print(filtered_SC[filtered_SC["hd"]==star][["Name", "hd", "RA", "DEC", "Vmag", "Teff", "SWFlag",  "N_HARPS", "SNR_MIN_HARPS", "SNR_MAX_HARPS", "N_UVES", "SNR_MIN_UVES", "SNR_MAX_UVES"]])

print(filtered_SC[["Name", "hd", "RA", "DEC", "Vmag", "Teff", "SWFlag", "N_HARPS", "SNR_MIN_HARPS", "SNR_MAX_HARPS", "N_UVES", "SNR_MIN_UVES", "SNR_MAX_UVES"]])

filtered_SC.to_csv("sweet_cat_stars.csv",index=False)   