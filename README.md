# PEEC-24

This repository's aim is to track the work done in the PEEC "The stellar activity in stars with exoplanets".

For now, there is a study comparing the results of $\log R'_{HK}$ for three stars in the paper Pepe et al 2011 (DOI: 10.1051/0004-6361/201117055) with the values obtained using the ACTIN2 tool (https://github.com/gomesdasilva/ACTIN2). There are seven Python scripts:

- average_spec_night.py: gets average of spectra observations available in ESO archive. Fetches the paths and files, average by night of observation and saves them.
- RV_CCF_measure.py: computes the RV from CCF of the spectra, a.k.a. "manual" way, and corrects the spectra by Doppler shift.
- actin_manual_rv.py: runs ACTIN2 on the manual RV corrected spectra, plots and save results.
- test_instrument.py: contains a test/mock class for instrument, to store the RV, BJD and spectra to use in the above file.
- compare_manual_pepe.py: compares the results of actin_manual_rv.py with the values of $\log R'_{HK}$ of Pepe et al, 2011, in a 1 to 1 plot.
- pepe2011.py: same as actin_manual_rv.py, but uses the RV in the CCF files from ESO to correct the spectra "automatically".
- util_funcs: contains several utilitary functions used in the other scripts.