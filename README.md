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

The second task was to write a program, get_eso_spectra.py, that fetches the spectra available for a given instrument (for now HARPS, but in the future ESPRESSO and UVES) in ESO data base, filters them and downloads them. Then the program corrects the spectra by the RV, computes the activity indices with ACTIN2 and stores the results. In a more comprehensive way:

- Starts by quering in ESO data base the object and the instrument.
- Checks if there is data to neglect.
- Cuts the data sets by the minimum SNR chosen.
- Chooses the best SNR spectra that cover a big time span, binned by month.
- Checks if the folder for the instrument exists or not. If not creates the necessary folders.
- According to the instrument (for now only HARPS is defined) it retrieves the data from ESO and untars the ancillary data (optional). For non-HARPS instruments, it just retrieves the data in a simple way.
- Corrects the spectra by RV comparing the spectra to a spectrum of the Sun.
- Runs ACTIN2, obtains activity indices for CaII H\&K, H $\alpha$ at 0.6 $\mathring A$ and NaI and store the results in a data frame.
- Saves plots of the spectra in lines of interest to check if everything is alright, as well as plots of the activity indices and the statistics related to them.

This last program also makes use of util_funcs.py.

Some problems with the get_eso_spectra program are:

- crashes if the number of spectra is too high because of the lines plot (too much memory used)
- maybe a better way of dealing with negative values in the flux, instead of just ignoring the spectra or replacing the point by 0
- some way of reducing running time, as well as supressing verbose of ESO download

Future work:

- ESPRESSO and UVES spectrographs are not yet configured
- missing an automatic quality index, for now only way of checking bad spectra is by outliers and visual inspection of spectral lines plots
- crashes if the star has no spectra by the instrument asked (future work)
- does not delete the downloaded spectra after processing, leading to shortage of memory when processing several stars (future work)
