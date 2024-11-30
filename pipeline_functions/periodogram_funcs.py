'''
This file contains functions to be used to compute the activity periodogram in the SAITAMA pipeline
'''
import numpy as np, matplotlib.pylab as plt, gc
from matplotlib.lines import Line2D
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

class gls_periodogram:
    """Generalised Lomb-Scargle Periodogram using the `astropy` implementation.
    
    Args:
        star (str): Star ID.
        p_rot_lit (list): Literature rotation period and its error for comparison.
        ind (str): Activity indicator name.
        x (array-like): Time coordinate.
        y (array-like): Y-coordinate (signal).
        y_err (array-like, optional): Y error if provided.
        pmin (float): Minimum period to compute periodogram.
        pmax (float): Maximum period to compute periodogram.
        steps (int): Number of steps in the frequency grid.
        verb (bool): Whether to print information or not.
        save (bool): Whether to save the plots or not.
        folder_path (str, optional): Path to save the plots.

    Returns:
        dict: Dictionary containing periodogram results.
    """

    def __init__(self, star, ind, x, y, y_err=None, pmin=1.5, pmax=1e4, steps=1e5, verb=True, save=False, folder_path=None):
        self.star = star
        self.ind = ind
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.y_err = y_err
        self.t = max(x) - min(x)
        self.n = len(x)
        self.pmin = pmin
        self.pmax = pmax
        self.steps = steps
        self.verb = verb
        self.save = save
        self.folder_path = folder_path
        self.results = {}


    def compute_periodogram(self):
        """Compute the periodogram and window function."""
        #Nyquist frequencies computation
        k1 = self.pmax/self.t; fmin = 1./(k1 * self.t)
        k2 = self.pmin * self.n/self.t; fmax = self.n/(k2 * self.t)

        freq = np.linspace(fmin, fmax, int(self.steps))
        period = 1./freq
        #get power from GLS
        if int(self.y_err[0]) == 0:
            gls = LombScargle(self.x, self.y)
        else:
            gls = LombScargle(self.x, self.y, self.y_err)
        power = gls.power(freq)

        #detect peaks
        peaks, _ = find_peaks(power)
        sorted_peak_indices = np.argsort(power[peaks])[::-1]  #sort in descending order of power
        sorted_peaks = peaks[sorted_peak_indices]
        peaks_power = power[sorted_peaks]
        peaks_period = period[sorted_peaks]
        
        #False alarm probabilities (FAP)
        fap_max_power = gls.false_alarm_probability(np.nanmax(power))
        faps = gls.false_alarm_probability(power)
        fap_levels = np.array([0.05, 0.01, 0.001])
        fap5, fap1, fap01 = gls.false_alarm_level(fap_levels)

        self.results.update({'freq': freq,'period': period,'power': power,'fap_maxp': fap_max_power,'fap_01': fap01,'fap_1': fap1,'fap_5': fap5,'FAPS': faps,})

        #Window function
        y_win = np.ones_like(self.y)
        power_win = LombScargle(self.x, y_win, fit_mean=False, center_data=False).power(freq)
        period_best_WF = period[np.argmax(power_win)]
        self.results.update({'power_win': power_win, 'period_best_WF': period_best_WF})

        peaks_WF, _ = find_peaks(power_win)
        sorted_peak_indices = np.argsort(power_win[peaks_WF])[::-1]
        sorted_peaks = peaks_WF[sorted_peak_indices]
        peaks_power_win = power_win[sorted_peaks]
        peaks_period_win = period[sorted_peaks]

        #Get significant peaks
        dict_peaks = {"peaks_period":peaks_period,"peaks_power":peaks_power}
        dict_peaks_WF = {"peaks_period_win":peaks_period_win,"peaks_power_win":peaks_power_win}
        gaps = self._gaps_time(self.x)
        sel_peaks_dict = self._get_sign_gls_peaks(dict_peaks, dict_peaks_WF, gaps, fap5, atol_frac=0.1, verb=False, evaluate_gaps=False)
        
        self.results.update({'gaps': gaps,'sel_peaks_period': sel_peaks_dict["sel_peaks_period"],'sel_peaks_power': sel_peaks_dict["sel_peaks_power"]})

        try:
            plevels = [fap5,fap1,fap01]
            period_best = sel_peaks_dict["sel_peaks_period"][0]
            power_best = sel_peaks_dict["sel_peaks_power"][0]
            period_err = self._get_period_error(self.x,self.y,power,power_best,freq)
            harmonics_list = self._get_harmonic_list(sel_peaks_dict["sel_peaks_period"], self.verb)
            flag = self._periodogram_flagging(harmonics_list, period_best, period_err, sel_peaks_dict["sel_peaks_power"], plevels)
   
        except: #there is no significant peaks in the periodogram
            period_best = 0
            power_best = 0
            period_err = 0
            harmonics_list = []
            flag = 0

        self.results.update({'period_best': round(period_best,3), 'power_best': power_best, 'period_best_err': round(period_err,3), 'flag':flag, 
                             'harmonics_list':harmonics_list})


    def plot_periodogram(self, period, power, sel_peaks, fap5, fap1, fap01, save=False):
        """Generate and save the periodogram plot."""
        fig, axes = plt.subplots(figsize=(10, 4))
        axes.text(0.13, 0.89, f"{self.star} GLS {self.ind}", fontsize=12, transform=plt.gcf().transFigure)
        axes.set_xlabel("Period [d]")
        axes.set_ylabel("Normalized Power")
        axes.semilogx(period, power, 'k-')

        plevels = [fap5,fap1,fap01]
        fap_levels = np.array([0.05, 0.01, 0.001])
        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        for peak in sel_peaks:
            axes.axvline(peak, ls='--', lw=0.8, color='red')

        vline_legend = Line2D([0], [0], color='red', linestyle='--', label='Significant peaks')
        handles, _ = axes.get_legend_handles_labels()
        handles.append(vline_legend)
        axes.legend(handles=handles, loc="best")

        axes.text(0.13, 0.6, f"P = {np.around(self.results['period_best'],1)} ± {np.around(self.results['period_best_err'],1)} d ({self.results['flag']})", fontsize=15, 
                  bbox={"facecolor":'white'},transform=plt.gcf().transFigure)

        if save and self.folder_path:
            plt.savefig(f"{self.folder_path}{self.star}_{self.ind}_GLS.pdf", bbox_inches="tight")
        plt.close('all')
        gc.collect()


    def plot_window_function(self, period, power_win, fap5, fap1, fap01, save=False):
        """Generate and save the window function periodogram plot."""
        fig, axes = plt.subplots(figsize=(10, 4))
        axes.set_xlabel("Period [days]"); axes.set_ylabel("Power")
        axes.semilogx(period, power_win, 'b-',label="WF")
        axes.semilogx(period, self.results["power"], 'r-',lw=0.7,label="data")
        axes.text(0.13, 0.89, f"{self.star} WF GLS {self.ind}", fontsize=12, transform=plt.gcf().transFigure)

        plevels = [fap5,fap1,fap01]
        fap_levels = np.array([0.05, 0.01, 0.001])
        for i in range(len(fap_levels)):
            axes.plot([min(period), max(period)], [plevels[i]]*2, '--', label="FAP = %4.1f%%" % (fap_levels[i]*100))

        for gap in self.results["gaps"]:
            axes.axvline(gap, ls='--', lw=0.7, color='green')
        axes.legend(loc="best")

        if save and self.folder_path:
            plt.savefig(f"{self.folder_path}{self.star}_{self.ind}_GLS_WF.pdf", bbox_inches="tight")
        plt.close('all')


    def run(self):
        """Run the GLS analysis."""
        self.compute_periodogram()

        if self.save:
            self.plot_periodogram(self.results["period"], self.results["power"], self.results["sel_peaks_period"], self.results['fap_5'], self.results['fap_1'], self.results['fap_01'], save=self.save)
            self.plot_window_function(self.results["period"], self.results["power_win"], self.results['fap_5'], self.results['fap_1'], self.results['fap_01'], save=self.save)

        if self.verb:
            print("Gaps in BJD:", self.results["gaps"])
            print("Significant Peaks: ", self.results["sel_peaks_period"])
            print("Period flag: ",self.results['flag'])

        return self.results


    def _get_sign_gls_peaks(self, dict_peaks, dict_peaks_WF, gaps, fap5, atol_frac=0.05, verb=False, evaluate_gaps=False):
        """Get GLS significant peaks and excludes peaks close to window function peaks and to gaps in BJD.
        Args:
            df_peaks (dict): star ID.
            df_peaks_WF (dict): columns to compute the statistics on.
            gaps (ndarray): time gaps in the time series.
            fap1 (float): 1% false alarm probability.
            atol_frac (float):
            verb (boolean): print or not extra information.
            evaluate_gaps (boolean): whether to remove peaks close to time gaps.

        Returns:
            sel_peaks_dict (dict): period and power of significant peaks.
        """
        sign_peaks = [per for per, power in zip(dict_peaks['peaks_period'], dict_peaks['peaks_power']) if power > fap5]
        sign_peaks_power = [power for per, power in zip(dict_peaks['peaks_period'], dict_peaks['peaks_power']) if power > fap5]
        sign_peaks_win = [per for per, power in zip(dict_peaks_WF['peaks_period_win'], dict_peaks_WF['peaks_power_win']) if power > fap5]

        # exclude peaks close to win peaks
        exc_peaks = []; exc_peaks_power = []
        for ind,peak in enumerate(sign_peaks):
            atol = peak * atol_frac
            for peak_win in sign_peaks_win:
                if np.isclose(peak, peak_win, atol=atol):
                    exc_peaks.append(peak)
                    exc_peaks_power.append(ind)
                    if verb:
                        print(f"{peak:.2f} is close to win peak {peak_win:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")
            if evaluate_gaps == True:
                for gap in gaps:
                    if np.isclose(peak, gap, atol=atol):
                        exc_peaks.append(peak)
                        exc_peaks_power.append(ind)
                        if verb:
                            print(f"{peak:.2f} is close to gap {gap:.2f} for the tolerance {atol:.2f} ({int(atol_frac*100)} %)")               

        sel_peaks_period = [peak for peak in sign_peaks if peak not in exc_peaks]  
        sel_peaks_power = [pow for ind, pow in enumerate(sign_peaks_power) if ind not in exc_peaks_power]

        sel_peaks_dict = {"sel_peaks_period":sel_peaks_period,"sel_peaks_power":sel_peaks_power}

        return sel_peaks_dict
    

    def _gaps_time(self, BJD):
        '''Takes the BJD array and returns the 3 biggest gaps in time.
        Args:
            BJD (ndarray): time array.
        Returns:
            gaps (ndarray): time gaps in the time series.
        '''
        time_sorted = BJD[np.argsort(BJD)] #sorting time
        gaps = np.diff(time_sorted)
        gaps = gaps[np.argsort(gaps)][-3:]

        return gaps
    

    def _periodogram_flagging(self, harmonics_list, period, period_err, peaks_power, plevels):
        '''
        Color-based quality indicator of the period obtained from the GLS periodogram.
            Green/4: error < 20% and no harmonics in significant peaks involving the max peak
            Yellow/3: 30% > error > 20% and no harmonics in significant peaks involving the max peak
            Orange/2: harmonics involving the max peak, no error or error > 30%
            Red/1: many periods with power close to each under: if number of periods over 85% of the max > 0
            Black/0: discarded - period under 1 yr, below FAP 1% level, no significant peak obtained
        '''
        harmonics_list = list(np.unique(np.array(harmonics_list)))
        try: error = period_err/period * 100
        except: error = 0
        powers_close_max = [n for n in peaks_power if n > 0.99*np.max(peaks_power) and n != np.max(peaks_power)]

        if np.max(peaks_power) < plevels[-1] or period < 365:
            flag = "black"
        else:
            if 0 < error <= 20 and (period not in harmonics_list) and len(powers_close_max) == 0:
                flag = "green"
            elif 20 < error <= 30 and period not in harmonics_list and len(powers_close_max) == 0:
                flag = "yellow"
            elif (period in harmonics_list or period_err == 0.0 or error > 30) and len(powers_close_max) == 0:
                flag = "orange"
            elif len(powers_close_max) > 0:
                flag = "red"

        return flag
    
    def _get_period_error(self,t,y,power,power_best,freq=None):
        '''Period error estimation of the periodogram based on the GLS implementation by PyAstronomy.'''
        N = len(y)
        if isinstance(freq, np.ndarray) == False:
            th = t - np.min(t)
            tbase = np.max(th)
            ofac = 10; hifac = 1 #default values in PyAstronomy
            
            fstep = 1 / ofac / tbase; fnyq = 1/2 / tbase * N
            fbeg = fstep; fend = fnyq * hifac
            
            freq = np.arange(fbeg, fend, fstep)

        nf = len(freq)

        k = np.where(power == power_best)[0][0]
        
        # Maximum power
        pmax = power[k]
        
        # Get the curvature in the power peak by fitting a parabola y=aa*x^2
        if 1 < k < nf-2:
            # Shift the parabola origin to power peak
            xh = (freq[k-1:k+2] - freq[k])**2
            yh = power[k-1:k+2] - pmax
            # Calculate the curvature (final equation from least square)
            aa = np.dot(yh, xh) / np.dot(xh, xh)
            e_f = np.sqrt(-2./N / aa * (1.-pmax))
            P_err = e_f / freq[k]**2
        else:
            P_err= np.nan
        
        return P_err


    def _get_harmonic_list(self, period, tolerance=0.01, verb=False):
        '''
        Get list of the harmonics in the list of periods.

        Args:
            period (ndarray): period array.
            tolerance (float): tolerance.
        Returns:
            harmonics_list (list): list of pairs of periods that are harmonics.

        '''
        harmonics_list = []

        for i in range(len(period)):
            for j in range(i+1, len(period)):
                ratio = period[j] / period[i]
                #Check if the ratio is close to an integer or simple fraction
                if abs(ratio - round(ratio)) < tolerance:
                    harmonic_bool =  True
                else:
                    harmonic_bool =  False

                if harmonic_bool == True:
                    harmonics_list.append([period[i], period[j]])
                    if verb == True:
                        print(f"Period {period[i]} and {period[j]} are harmonics of each other")

        return harmonics_list