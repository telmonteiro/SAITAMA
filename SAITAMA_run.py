'''
This script allows the user to easily run the SAITAMA pipeline, given the input parameters:
- stars: object identifiers, preferentially in the HD catalogue. Must be in a list format (list)
- instruments: name of the spectrograph to retrieve the spectra from. Must be in a list format (list)
- indices: spectral lines to compute the activity indices. Must be in a list format (list)
- max_spectra: maximum number of spectra to be downloaded (float or int)
- min_snr: minimum SNR to select the spectra. Must be in a float-like format
- download: boolean to decide if the spectra are downloaded or not (if not, it is assumed that the spectra is already inside the folders where 
the pipeline would download the spectra to) (bool)
- neglect_data: spectra to be manually neglected for whatever reason (dic)
- username_eso: username in the ESO data base to download the data (str)
- download_path: folder to where download spectra (str)
- final_path: folder to save products of the pipeline (str)
'''
from SAITAMA import SAITAMA

stars = ["HD209100"]#"HD209100","HD160691","HD115617","HD46375","HD22049","HD102365","HD1461","HD16417","HD10647","HD13445","HD142A",
      #  "HD108147","HD16141","HD179949","HD47536","HD20794","HD85512","HD192310"] 

instruments = ["HARPS","ESPRESSO","UVES"] #instruments

indices = ["I_CaII", "I_Ha06", "I_NaI"]  # indices for activity

max_spectra = 150 # maximum spectra to be selected
min_snr = 15 #minimum overall SNR

download = True #download from ESO database?
download_path = "download_spectra" #folder to download spectra
final_path = "pipeline_products" #folder to save products
username_eso = "telmonteiro" #ESO database username

# specific data to neglect for any reason
neglect_data = {"HD20794": "ADP.2020-08-10T15:36:05.310"}

SAITAMA(stars, instruments, indices, max_spectra, min_snr, download, neglect_data, username_eso, download_path, final_path)