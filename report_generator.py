import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from astropy.io import fits
from util_funcs import read_bintable
from PyAstronomy.pyTiming import pyPeriod

def get_report_periodogram(hdr,gaps,period,period_err,flag_period,harmonics_list,period_WF,period_err_WF,folder_path):
    instr = hdr["INSTR"]; star = hdr["STAR_ID"]
    t_span = hdr["TIME_SPAN"]
    snr_min = hdr["SNR_MIN"]; snr_max = hdr["SNR_MAX"]
    n_spec = hdr["I_CAII_N_SPECTRA"]
    flag_rv = hdr["FLAG_RV"]

    name_file = folder_path+f"report_periodogram_{star}.txt"
    with open(name_file, "w") as f:
        f.write("###"*30+"\n")
        f.write("\n")
        f.write("Periodogram Report\n")
        f.write("---"*30+"\n")
        f.write(f"Star: {star}\n")
        f.write(f"Instrument: {instr}\n")
        f.write(f"SNR: {snr_min} - {snr_max}\n")
        f.write(f"RV flag: {flag_rv}\n")
        f.write(f"Time Span: {t_span}\n")
        f.write(f"Number of spectra: {n_spec}\n")
        f.write("---"*30+"\n")
        f.write(f"Period I_CaII: {period} +/- {period_err} days\n")
        f.write(f"Period flag: {flag_period}\n")
        f.write(f"Harmonics: {harmonics_list}\n")
        f.write(f"Time gaps between data points: {gaps}\n")
        f.write(f"Window Function Period: {period_WF} +/- {period_err_WF} days\n")
        f.write("\n")
        f.write("###"*30+"\n")
    
    f = open(name_file, 'r')
    file_contents = f.read()
    f.close()

    return file_contents

        

