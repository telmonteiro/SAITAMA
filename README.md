# ACTINometer

This repository's aim is to track the work done in the PEEC "The stellar activity in stars with exoplanets", made in 2024.

## First task: case-study of Pepe et al. 2011

The first task was a case-study comparing the results of $\log R'_{HK}$ for three stars in the paper Pepe et al 2011 (DOI: 10.1051/0004-6361/201117055) with the values obtained using the ACTIN2 tool (https://github.com/gomesdasilva/ACTIN2).

## Second task: ACTINometer - a pipeline for the derivation of spectral activity indices for stars with exoplanets

This pipeline uses the ACTIN tool (2018JOSS....3..667G and 2021A&A...646A..77G) that computes the spectral activity indices for a stellar spectrum. The objective is to apply ACTIN on the most stars possible from the SWEET-Cat catalogue (2021A&A...656A..53S), which contains the stars with exoplanets and respective stellar parameters. The spectral activity indices chosen in this pipeline are the CaII H&K, the H$\alpha$ 0.6 A and the NaI lines.

### Description of the pipeline

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
- df_stats_{star}.fits: FITS file that includes the data frame given by df\_{star}\_{instrument}.csv in a BINTable format and includes the statistics given in stats_{star}.csv
and informations regarding the periodogram in the header
- report_periodogram_{star}.txt: txt file that contains a small report on the periodogram computed and the WF, as well as the harmonics
- flag_ratios_{instrument}.txt: adds to a txt file the names of the badly corrected spectra and the RV flag ratio

For now, the testing phase of this pipeline includes the 3 stars from Pepe et al. 2011 and 15 other stars chosen from SWEET-Cat, that agree with
- Vmag < 8
- Declination under +30º
- SWFlag = 1 (homogeneus parameters)

The resulting list of stars is:  'HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461','HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536'.
HD47536 was particularly chosen because it has a low number of spectra available, so it acts as a test.

### Running the pipeline

- For now, running the pipeline is a bit complicated. First, one has to clone the ACTIN2 repository (https://github.com/gomesdasilva/ACTIN2). Then, clone this repository inside the folder of ACTIN2, so that ACTINometer.py is in the same folder as actin2 folder. This is made this way to ease the import of the ACTIN2 module.

- Then one runs the ACTINometer by the terminal. To change the input parameters, for now one has to edit them manually inside the script of ACTINometer.py

## Problems and future work

- the periodogram computation is under study
- a comparison between the HARPS and UVES activity indices is under way
- use calibrations to compute the age and rotation of the stars
