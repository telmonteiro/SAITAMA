# PEEC-24

This repository's aim is to track the work done in the PEEC "The stellar activity in stars with exoplanets".

## First task: case-study of Pepe et al. 2011

The first task was a case-study comparing the results of $\log R'_{HK}$ for three stars in the paper Pepe et al 2011 (DOI: 10.1051/0004-6361/201117055) with the values obtained using the ACTIN2 tool (https://github.com/gomesdasilva/ACTIN2). Inside the folder "Pepe2011" are six Python scripts:

- average_spec_night.py: gets average of spectra observations available in ESO archive. Fetches the paths and files, average by night of observation and saves them.
- RV_CCF_measure.py: computes the RV from CCF of the spectra, a.k.a. "manual" way, and corrects the spectra by Doppler shift.
- actin_manual_rv.py: runs ACTIN2 on the manual RV corrected spectra, plots and save results.
- test_instrument.py: contains a test/mock class for instrument, to store the RV, BJD and spectra to use in the above file.
- compare_manual_pepe.py: compares the results of actin_manual_rv.py with the values of $\log R'_{HK}$ of Pepe et al, 2011, in a 1 to 1 plot.
- pepe2011.py: same as actin_manual_rv.py, but uses the RV in the CCF files from ESO to correct the spectra "automatically".

An additional script, in the main folder, was used:

- util_funcs_py: contains several utilitary functions used in the other scripts.

## Second task: building a program to process any star

The second task was to write a program, get_eso_spectra.py, that fetches the spectra available for a given instrument (for now HARPS, but in the future ESPRESSO and UVES) in ESO data base, filters them and downloads them. Then the program corrects the spectra by the RV, computes the activity indices with ACTIN2 and stores the results. This program also makes use of util_funcs.py. In a more comprehensive way:

- Starts by quering in ESO data base the object and the instrument.
- Checks if there is data to neglect.
- Cuts the data sets by the minimum and maximum SNR chosen.
- Chooses the best SNR spectra that cover a big time span, binned by month.
- Checks if the folder for the instrument exists or not. If not creates the necessary folders.
- According to the instrument (for now only HARPS is defined) it retrieves the data from ESO and untars the ancillary data (optional). For non-HARPS instruments, it just retrieves the data in a simple way.
- Corrects the spectra by RV comparing the spectra to a spectrum of the Sun.
- Computes a quality indicator of the RV correction.
- Runs ACTIN2, obtains activity indices for CaII H\&K, H $\alpha$ at 0.6 $\mathring A$ and NaI and store the results in a data frame.
- Computes a periodogram using GLS to retrieve the period of $S_\text{CaII}$.
- Saves plots of the spectra in lines of interest to check if everything is alright, as well as plots of the activity indices and the statistics related to them.
- Saves the statistics and data frame in a fits file.

For now, only the 3 stars from Pepe et al. 2011 were studied. The next step is to experiment with 15 new stars chosen from SWEET-Cat, that agree with
- Vmag < 8
- Declination under +30º
- SWFlag = 1 (homogeneus parameters)

The resulting list of stars is:  ['HD209100', 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461','HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536']
HD47536 was particularly chosen because it has a low number of spectra available, so it acts as a test. For now the experimentation is ongoing with HARPS spectra, but the next step is with UVES.

## Problems and future work

Some problems with the get_eso_spectra program are:
- some way of reducing running time, as well as supressing verbose of ESO download

Future work:
- ESPRESSO and UVES spectrographs are not yet configured
- does not delete the downloaded spectra after processing, leading to shortage of memory when processing several stars
- use calibrations to compute the age and rotation of the stars
