'''
This file contains functions to be used to correct and flag the spectra by the RV in the SAITAMA pipeline
'''
import numpy as np
from PyAstronomy import pyasl # type: ignore
from general_funcs import SpecFunc
SpecFunc = SpecFunc()
from get_spec_funcs import _get_simbad_data

class RV_correction:

    def __init__(self):
        self.name = "RV_correction"

    def _get_rv_ccf(self, star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument, quick_RV):
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
        tw = SpecFunc._calc_fits_wv_1d(template_hdr); tf = template_spec

        #to make the two spectra compatible, cut the stellar one
        w_ind_common = np.where((w < np.max(tw)) & (w > np.min(tw)))
        w_cut = w[w_ind_common]; f_cut = f[w_ind_common]

        #only uses two 250 Angstrom intervals, that change according to the instrument
        if quick_RV == True: 
            if instrument == "UVES": #doesn't include CaII H&K lines
                w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5500 > w) & (w > 5250)) #Halpha region and leftmost side of spectrum
            else: #HARPS or ESPRESSO
                w_ind_tiny = np.where((6400 < w) & (w < 6700) | (5250 > w) & (w > 4900)) #Halpha region and CaII H&K region
            w_cut = w_cut[w_ind_tiny]; f_cut = f_cut[w_ind_tiny]   

        try:
            rv_interval_instr = {"HARPS":5,"UVES":110,"ESPRESSO":110}
            rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
            rvmin = rv_simbad - rv_interval_instr[instrument]; rvmax = rv_simbad + rv_interval_instr[instrument]
        except:
            rvmin = -150; rvmax = 150

        #get the cross-correlation
        skipedge_values = [0, 100, 500, 1000, 5000, 20000]

        for skipedge in skipedge_values:
            try:
                rv_array, cc = pyasl.crosscorrRV(w=w_cut, f=f_cut, tw=tw, tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
                break  # Break out of the loop if successful
            except Exception as e:
                #print(f"Error with skipedge={skipedge}: {e}")
                continue

        #index of the maximum cross-correlation function
        maxind = np.argmax(cc)
        rv_ccf = rv_array[maxind]

        if units == "m/s":
            rv_ccf *= 1000
            rv_array *= 1000
        
        return bjd, rv_ccf, cc[maxind], np.around(rv_array,0), cc, w, f

    def _correct_spec_rv(self, wv, rv, units):
        '''
        Correct wavelength of spectrum by the RV of the star with a Doppler shift.
        '''
        c = 299792.458 #km/s

        if units == "m/s":
            c *= 1000
        
        return wv / (1+rv/c)

    ################################

    def _get_alphaRV(self, data, line="CaI"):
        '''
        Computes alphaRV, the ratio between the continuum and the center of the line fluxes of a reference line.
        The line chosen is CaI.
        '''
        lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,"NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
        line_wv = lines_list[line]
        
        if line in ["CaIIK","CaIIH"]: window = 12
        elif line in ["Ha","NaID1","NaID2"]: window = 22
        else: window = 0.7

        alpha_RV = np.zeros(len(data)); center_flux_line_arr = np.zeros(len(data)); flux_continuum_arr = np.zeros(len(data)) 

        for i in range(len(data)):
            wv = data[i][0]; flux = data[i][1]

            wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
            wv = wv[wv_array]
            flux = flux[wv_array]

            flux_normalized = flux/np.median(flux)

            flux_left_ind = np.where((wv > line_wv-window) & (wv < line_wv - window + 0.2))
            flux_right_ind = np.where((wv < line_wv+window) & (wv > line_wv + window - 0.2))
            flux_left = np.median(flux_normalized[flux_left_ind])
            flux_right = np.median(flux_normalized[flux_right_ind])

            flux_continuum = np.median([flux_left,flux_right])
            
            wv_center_line = np.where((line_wv-0.03 < wv) & (wv < line_wv+0.03))
            flux_line = flux_normalized[wv_center_line]
            center_flux_line = np.median(flux_line)

            ratio_fluxes = center_flux_line/flux_continuum 

            alpha_RV[i] = ratio_fluxes
            center_flux_line_arr[i] = center_flux_line
            flux_continuum_arr[i] = flux_continuum
            
        return alpha_RV, center_flux_line_arr, flux_continuum_arr


    def _get_betaRV(self,files,instr):
        '''
        Computes beta_RV, as an overall quality indicator of the RV correction. 
        For each spectrum, the algorithm runs an interval of offsets and if the minimum alpha_RV (ratio between the continuum and the center
        of the line fluxes) is not in the interval [-0.03,0.03] Angstrom, then gamma_RV (binary flag) = 1 for that spectrum. 
        Otherwise, gamma_RV = 0.
        beta_RV is then the ratio between the number of spectra with flag = 0 and the total number of spectra:
        - beta_RV = 1 means that all of the spectra was well corrected.
        - beta_RV = 0 means that none of the spectra was well corrected.
        '''
        offset_list = np.linspace(-1,1,501)
        gamma_RV_arr = np.zeros((len(files)))

        for i,file in enumerate(files):
            spectrum, header = SpecFunc._read_fits(file,instrument=instr,mode="rv_corrected")
            wv, flux = spectrum["wave"], spectrum["flux"]

            alpha_RV_arr = np.zeros_like(offset_list)
            for j,offset in enumerate(offset_list):
                alpha_RV, _, _ = self._get_alphaRV([(wv+offset,flux)], line="CaI")
                alpha_RV_arr[j]=alpha_RV[0]

            min_ratio_ind = np.argmin(alpha_RV_arr)
            offset_min = offset_list[min_ratio_ind]

            if offset_min < -0.03 or offset_min > 0.03:
                gamma_RV_arr[i] = 1
            else: gamma_RV_arr[i] = 0

        good_spec_ind = np.where((gamma_RV_arr == 0))
        N_good_spec = len(gamma_RV_arr[good_spec_ind])
        beta_RV = N_good_spec / len(gamma_RV_arr)

        return beta_RV, gamma_RV_arr