from ACTINometer import ACTINometer

#stars = ["HD209100","HD160691","HD115617","HD46375","HD22049","HD102365","HD1461","HD16417","HD10647","HD13445","HD142A",
#        "HD108147","HD16141","HD179949","HD47536","HD20794","HD85512","HD192310"]

stars = ["HD209100"]

instruments = ["HARPS","ESPRESSO","UVES"] #instruments

indices = ["I_CaII", "I_Ha06", "I_NaI"]  # indices for activity

max_spectra = 150 # maximum spectra to be selected
min_snr = 15 #minimum overall SNR

download = False #download from ESO database?
download_path = "download_spectra" #folder to download spectra
final_path = "pipeline_products" #folder to save products
username_eso = "telmonteiro" #ESO database username

# specific data to neglect for any reason
neglect_data = {"HD20794": "ADP.2020-08-10T15:36:05.310"}

ACTINometer(stars, instruments, indices, max_spectra, min_snr, download, neglect_data, username_eso, download_path, final_path)