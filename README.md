# ACTINometer - a pipeline for the derivation of spectral activity indices for stars with exoplanets

This repository's aim is to track the work done in the PEEC "The stellar activity in stars with exoplanets", made in 2024.

This pipeline uses the ACTIN tool (2018JOSS....3..667G and 2021A&A...646A..77G) that computes the spectral activity indices for a stellar spectrum. The objective is to apply ACTIN on the most stars possible from the SWEET-Cat catalogue (2021A&A...656A..53S), which contains the stars with exoplanets and respective stellar parameters. The spectral activity indices chosen in this pipeline are the CaII H&K, the H$\alpha$ 0.6 A and the NaI lines.

## Description of the pipeline

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

A schematic flow-chart of the pipeline can be seen in ACTINometer_FlowChart.pdf.

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
    - {star}\_{instrument}\_{line}.pdf: plot of the spectra in said line (CaI, CaII H, CaII K, H$\alpha$, HeI, NaID1, NaID2 and FeII)
    - stats_{star}.csv: csv file with a data frame including statistical information like the star name, the indice name and the respective maximum, minimum, mean, median, 
    standard deviation, weighted mean, time span and number of spectra used for that indice
    - df_{star}_{instrument}.csv: csv file with the data frame including the columns_df given as input before any processing
    - {star}\_GLS.pdf and {star}_WF.pdf: plots of the periodogram of the CaII H&K indice and the respective Window Function
    - df_stats\_{star}.fits: FITS file that includes the data frame given by df_{star}_{instrument}.csv in a BINTable format and includes the statistics given in stats_{star}.csv
    and informations regarding the periodogram in the header
    - report\_periodogram\_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics
- master_df_{target_save_name}.fits: FITS file that contains the information for each instrument separately plus a BINTable + Header for the combined data
- {star}\_GLS.pdf and {star}\_WF.pdf: plots of the periodogram of the CaII H&K indice and the respective Window Function for the combined data
- report_periodogram\_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics for the combined data

The testing phase of this pipeline includes the 3 stars from Pepe et al. 2011 and 15 other stars chosen from SWEET-Cat, that agree with
- Vmag < 8
- Declination under +30º
- SWFlag = 1 (homogeneus parameters)

The resulting list of stars is: 'HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461','HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536'.
HD47536 was particularly chosen because it has a low number of spectra available, so it acts as a test.

## Running the pipeline

- To run this pipeline, the user first needs to download and install the ACTIN2 tool (https://github.com/gomesdasilva/ACTIN2), so it can be accessed as a package. Other dependencies include:
  - numpy==1.26.4, pandas==1.5.3, astropy==6.1.0, matplotlib==3.9.0, scipy==1.13.1, astroquery==0.4.7, PyAstronomy==0.21.0, tqdm==4.66.1.
- The versions above are the ones used to code and run the pipeline, not mandatory. The Python version used was 3.10.12, ran inside WSL Ubuntu with VsCode.
- Then, in the directory of ACTINometer, the user runs
  
pip install -r requirements.txt

sudo python setup.py install

- Finally, to run ACTINometer, one can use the file ACTINometer_run.py, where the user can change the input parameters.

## Problems and future work

- correct eventual bugs
- the sometimes odd results from UVES may deserve a more thorough analysis
- in the near future a full report describing ACTINometer will be included
- no UVES calibration to obtain S_MW
