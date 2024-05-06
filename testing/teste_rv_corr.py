from __future__ import print_function, division
import numpy as np, pandas as pd, matplotlib.pylab as plt, os, tarfile, glob, tqdm, time
from astropy.io import fits
from PyAstronomy import pyasl
from util_funcs import calc_fits_wv_1d, _get_simbad_data, read_fits, correct_spec_rv, flag_ratio_RV_corr, plot_line

def get_rv_ccf(star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument):
    '''
    Uses crosscorrRV function from PyAstronomy to get the CCF of the star comparing with a spectrum of the Sun.
    Returns the BJD in days, the RV, the max value of the CCF, the list of possible RV and CC, as well as the flux and raw wavelength.
    To maximize the search for RV and avoid errors, the script searches the RV in SIMBAD and makes a reasonable interval. 
    '''
    
    if instrument == "HARPS" or instrument == "UVES" or instrument == "ESPRESSO":
        bjd = stellar_header["HIERARCH ESO DRS BJD"]
    else: bjd = None

    #get wavelength and flux of both sun (template) and star
    w = stellar_wv; f = stellar_flux
    tw = calc_fits_wv_1d(template_hdr); tf = template_spec

    w_ind_common = np.where((w < np.max(tw)) & (w > np.min(tw)))
    w_cut = w[w_ind_common]
    f_cut = f[w_ind_common]

    try:
        rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
        rvmin = rv_simbad - 10; rvmax = rv_simbad + 100
    except:
        rvmin = -150; rvmax = 150

    #get the cross-correlation
    skipedge_values = [0, 100, 500, 1000, 5000, 20000, 50000]

    for skipedge in skipedge_values:
        try:
            rv, cc = pyasl.crosscorrRV(w=w_cut, f=f_cut, tw=tw,
                                    tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
            break  # Break out of the loop if successful
        except Exception as e:
            #print(f"Error with skipedge={skipedge}: {e}")
            print(f"Error with skipedge={skipedge}")
            # Continue to the next iteration

    #index of the maximum cross-correlation function
    maxind = np.argmax(cc)
    radial_velocity = rv[maxind]

    if units == "m/s":
        radial_velocity *= 1000
        rv *= 1000
    
    return bjd, radial_velocity, cc[maxind], np.around(rv,0), cc, w, f

stars = ['HD46375']#, 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        # 'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536']

instr = "ESPRESSO"

for target_save_name in stars:
    files = glob.glob(os.path.join(f"teste_download/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))
    sun_template_wv, sun_template_flux, sun_template_flux_err, sun_header = read_fits(file_name="Sun1000.fits",instrument=None, mode=None) #template spectrum for RV correction
    drv = .1 #step to search rv
    for i,file in enumerate(files):
        print(file)
        wv, f, f_err, hdr = read_fits(file,instr,mode="raw")

        bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star = target_save_name, stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                                                template_hdr = sun_header, template_spec = sun_template_flux, 
                                                drv = drv, units = "m/s", instrument=instr)
        
        wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")

        plt.figure(1)
        plot_line(data=[(wv_corr,f)], line="CaI")

        #flag for goodness of RV correction  
        file_path = file.replace('teste_download', 'teste_download_rv_corr')
        flag_ratio, flag_list = flag_ratio_RV_corr([file_path],instr)
        flag = flag_list[0]
        print(flag)

        print(radial_velocity)

        plt.figure(2)
        plt.plot(rv,cc)
        plt.show()