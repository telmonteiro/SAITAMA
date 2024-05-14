import numpy as np, pandas as pd, matplotlib.pylab as plt, os, tarfile, glob, tqdm, time
from astropy.io import fits
from PyAstronomy import pyasl # type: ignore
from general_funcs import (calc_fits_wv_1d, _get_simbad_data, read_fits, plot_line, read_bintable)
from RV_correction_funcs import (correct_spec_rv, flag_ratio_RV_corr)

def get_rv_ccf(star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument, quick_RV):
    '''
    Uses crosscorrRV function from PyAstronomy to get the CCF of the star comparing with a spectrum of the Sun.
    Returns the BJD in days, the RV, the max value of the CCF, the list of possible RV and CC, as well as the flux and raw wavelength.
    To maximize the search for RV and avoid errors, the script searches the RV in SIMBAD and makes a reasonable interval. 
    '''
    
    if instrument == "HARPS" or instrument == "UVES" or instrument == "ESPRESSO":
        bjd = stellar_header["HIERARCH ESO DRS BJD"] #may change with instrument
    else: bjd = None

    #get wavelength and flux of both sun (template) and star
    w = stellar_wv; f = stellar_flux
    tw = calc_fits_wv_1d(template_hdr); tf = template_spec

    #to make the two spectra compatible, cut the stellar one
    w_ind_common = np.where((w < np.max(tw)) & (w > np.min(tw)))
    w_cut = w[w_ind_common]
    f_cut = f[w_ind_common]

    #only uses two 250 Angstrom intervals, that change according to the instrument
    if quick_RV == True: 
        if instrument == "UVES": #doesn't include CaII H&K lines
            w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5500 > w) & (w > 5250)) #Halpha region and leftmost side of spectrum
        else: #HARPS or ESPRESSO
            w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5250 > w) & (w > 4900)) #Halpha region and CaII H&K region
        w_cut = w_cut[w_ind_tiny]
        f_cut = f_cut[w_ind_tiny]   

    try:
        rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
        if instrument == "HARPS":
            rvmin = rv_simbad - 5; rvmax = rv_simbad + 5
        elif instrument == "UVES":
            rvmin = rv_simbad - 80; rvmax = rv_simbad + 80
        elif instrument == "ESPRESSO":
            rvmin = rv_simbad - 80; rvmax = rv_simbad + 80
    except:
        rvmin = -150; rvmax = 150

    #get the cross-correlation
    skipedge_values = [0, 100, 500, 1000, 5000, 20000]

    for skipedge in skipedge_values:
        try:
            rv, cc = pyasl.crosscorrRV(w=w_cut, f=f_cut, tw=tw, tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
            break  # Break out of the loop if successful
        except Exception as e:
            #print(f"Error with skipedge={skipedge}: {e}")
            continue
            # Continue to the next iteration

    #index of the maximum cross-correlation function
    maxind = np.argmax(cc)
    radial_velocity = rv[maxind]

    if units == "m/s":
        radial_velocity *= 1000
        rv *= 1000
    
    return bjd, radial_velocity, cc[maxind], np.around(rv,0), cc, w, f


stars = ['HD209100']#, 'HD160691', 'HD115617', 'HD46375', 'HD22049', 'HD102365', 'HD1461', 
        # 'HD16417', 'HD10647', 'HD13445', 'HD142A', 'HD108147', 'HD16141', 'HD179949', 'HD47536']

instr = "HARPS"

for target_save_name in stars:
    file_directory = glob.glob(os.path.join(f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/", f"df_stats_{target_save_name}.fits"))
    df_instr, hdr = read_bintable(file_directory[0], print_info=False)
    files = list(df_instr["file"])
    #files = glob.glob(os.path.join(f"teste_download/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))
    sun_template_wv, sun_template_flux, sun_template_flux_err, sun_header = read_fits(file_name="Sun1000.fits",instrument=None, mode=None) #template spectrum for RV correction
    drv = 0.5 #step to search rv 0.5 km/s
    files = tqdm.tqdm(files)
    for i,file in enumerate(files):
        #print(file)
        wv, f, f_err, hdr = read_fits(file,instr,mode="raw")

        bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star = target_save_name, stellar_wv = wv, stellar_flux = f, stellar_header = hdr,
                                                template_hdr = sun_header, template_spec = sun_template_flux, 
                                                drv = drv, units = "m/s", instrument=instr, quick_RV = True) 
        
        wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")

        plt.figure(1)
        plot_line(data=[(wv_corr,f)], line="Ha")

        plt.figure(2)
        plot_line(data=[(wv_corr,f)], line="CaI")

        #flag for goodness of RV correction  
        file_path = file.replace('teste_download', 'teste_download_rv_corr')
        flag_ratio, flag_list = flag_ratio_RV_corr([file_path],instr)
        flag = flag_list[0]
        print(flag)

        print(radial_velocity)

        plt.figure(3)
        plt.plot(rv,cc)

        plt.figure(4)
        plt.plot(wv_corr,f)

        #plt.show()

#for HD209100 UVES (62 spectra):
#using drv = 0.1 and full spectra: 16:06 -22635.000000005533
#using drv = 0.5 and full spectra: 3:36 -22534.999999999996
#using drv = 0.1 and 2 intervals of wv: 10:01  -22535.000000005537
#using drv = 0.5 and 2 intervals of wv: 2:18  -22534.999999999996

#for HD20911 HARPS (143 spectra)
#drv = 0.5 and 2 intervals: 1:52 -40035.0
#drv = 0.1 and full spectra: 6:17 -39934.99999999993