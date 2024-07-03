import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from pipeline_functions.general_funcs import read_fits, plot_line

def get_alphaRV(data, line="CaI"):
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


def get_betaRV(files,instr):
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
        wv, flux, _, _ = read_fits(file,instrument=instr,mode="rv_corrected")

        alpha_RV_arr = np.zeros_like(offset_list)
        for j,offset in enumerate(offset_list):
            alpha_RV, _, _ = get_alphaRV([(wv+offset,flux)], line="CaI")
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

instr = "HARPS"
target_save_name="HD209100"
files = glob.glob(os.path.join(f"/home/telmo/PEEC-24-main/pipeline_products/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))

#Plot the first spectrum of the star given, with the regions that matter highlighted.
wv, flux, flux_err, hdr = read_fits(files[0],instrument=instr,mode="rv_corrected")
alpha_RV, center_flux_line_arr, flux_continuum_arr = get_alphaRV([(wv,flux)], line="CaI")
print(f"For one spectrum:\n Ratio: {alpha_RV[0]}\n Flux center line: {center_flux_line_arr[0]}\n Flux continuum: {flux_continuum_arr[0]}")
print("#"*20)
plt.figure(1)
plot_line(data=[(wv,flux)], line="CaI",lstyle="-",line_color="black")
plt.axhline(y=center_flux_line_arr,xmin=0,xmax=1,ls="--",ms=0.2,c="red")
plt.axhline(y=flux_continuum_arr,xmin=0,xmax=1,ls="--",ms=0.2,c="red")
line_wv = 6572.795; window = 0.7
wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
wv1 = wv[wv_array]
flux1 = flux[wv_array]
flux_normalized = flux1/np.median(flux1)
plt.fill_betweenx(y=[np.min(flux_normalized),np.max(flux_normalized)], x1=line_wv-window, x2=line_wv - window + 0.2, alpha=0.15,color="blue")
plt.fill_betweenx(y=[np.min(flux_normalized),np.max(flux_normalized)], x1=line_wv + window - 0.2, x2=line_wv + window, alpha=0.15,color="blue")
plt.fill_betweenx(y=[np.min(flux_normalized),np.max(flux_normalized)], x1=line_wv-0.03, x2=line_wv+0.03,alpha=0.15,color="orange")
plt.savefig("CaI_line_alpha_RV.pdf",bbox_inches="tight")

#Plot that first spectra three times, one in the original position and the other two shifted in wavelength.
offset_list = np.linspace(-0.1,0.1,3)
plt.figure(2)
for offset in offset_list:
    plot_line(data=[(wv+offset,flux)], line="CaI",lstyle="-")
plt.title("Offsetting spectrum by 0.2 and -0.2 Angs")
plt.savefig("CaI_offset_2.pdf",bbox_inches="tight")

#Plot alpha_RV as a function of the offset in wavelength.
offset_list = np.linspace(-0.1,0.1,201)
alpha_RV_arr = np.zeros_like(offset_list)
for i,offset in enumerate(offset_list):
    alpha_RV, center_flux_line_arr, flux_continuum_arr = get_alphaRV([(wv+offset,flux)], line="CaI")
    alpha_RV_arr[i]=alpha_RV[0]
plt.figure(3)
plt.plot(offset_list,alpha_RV_arr)
plt.axvline(x=0,ymin=0,ymax=1,ls="-",ms=0.1,c="black")
plt.axvline(x=0.03,ymin=0,ymax=1,ls="--",ms=0.1,c="red")
plt.axvline(x=-0.03,ymin=0,ymax=1,ls="--",ms=0.1,c="red")
plt.xlabel(r"Offset ($\lambda$)"); plt.ylabel(r"$\alpha_{RV}$")
plt.savefig("alpha_RV_vs_offset.pdf",bbox_inches="tight")

#Plot the I_CaII activity indice from ACTIN as a function of the offset in wavelength
from actin2 import ACTIN
actin = ACTIN()
indices= ['I_CaII', 'I_Ha06', 'I_NaI']
headers = {}
offset_list = np.linspace(-0.1,0.1,101)
ICaII = np.zeros_like(offset_list); ICaII_err = np.zeros_like(offset_list)
for i,offset in enumerate(offset_list):
    spectrum = dict(wave=wv+offset, flux=flux)
    df_ind = actin.CalcIndices(spectrum, headers, indices).indices
    ICaII[i] = df_ind["I_CaII"]
    ICaII_err[i] = df_ind["I_CaII_err"]
    if offset == 0.03:
        offset_03 = ICaII[i]
        print(f"I_CaII at offset = 0.03: {offset_03}")
    elif offset == -0.03:
        offset__03 = ICaII[i]
        print(f"I_CaII at offset = -0.03: {offset__03}")
    elif offset == 0:
        offset_0 = ICaII[i]
        print(f"I_CaII at offset = 0: {offset_0}")
mean_offset = offset_0 - np.mean(np.array([offset_03,offset__03]))
print("Mean difference for offset 0.03: ",mean_offset)
plt.figure(4)
plt.errorbar(offset_list,ICaII,yerr=ICaII_err)
plt.xlabel(r"Offset ($\lambda$)"); plt.ylabel(r"$I_{CaII}$")
plt.axvline(x=0.03,ymin=0,ymax=1,ls="--",ms=0.2,c="red")
plt.hlines(y=offset__03,xmin=-0.03,xmax=0,ls="-",lw=1.4,color="red")
plt.axvline(x=-0.03,ymin=0,ymax=1,ls="--",ms=0.2,c="red")
plt.hlines(y=offset_03,xmin=0,xmax=0.03,ls="-",lw=1.4,color="red")
plt.axvline(x=-0,ymin=0,ymax=1,ls="--",ms=0.2,c="black")
plt.savefig("I_CaII_vs_offset.pdf",bbox_inches="tight")

#Plot an histogram of gamma_RV for each spectrum. 
plt.figure(5)
beta_RV, gamma_RV_arr = get_betaRV(files,instr)
plt.hist(gamma_RV_arr, bins=[-0.5, 0.5, 1.5], rwidth=0.8, align='mid', color='skyblue', edgecolor='black')
plt.ylabel("# Spectra"); plt.xlabel(r"$\gamma_{RV}$")
plt.xticks([0, 1])
plt.grid(axis='y')
print(r"$\beta_{RV}$: ",beta_RV)
plt.savefig("beta_RV_histogram.pdf",bbox_inches="tight")

plt.show()

'''
Plots the line, CaI, and computes the median of the first and last 20 points. then it takes the median of both values and that is 
considered the continuum flux

The flux of the line center is taken as the median of the points in an interval equal to  tabulated center +- window/30, where window for CaI is defined as 0.6 Angstrom

The ratio of the center and continuum flux is computed.

The flag is obtained by computing the ratio of fluxes for different offsets of wavelength (displacing the line). Naturally, the lowest ratio will be where the offset = 0,
if the spectrum is well corrected for RV. The flag is raised if the lowest ratio is not in a tolerance window of 0.02 Angstrom. Flag 1 = bad and 0 = good.

To apply this indicator to a star with several spectra, one has to compute the flag for each spectrum and then take the Number of 0 / Total spectra = #0 / #total
So if every spectrum is ok, the flag ratio will be = 1. if there is even one single not good spectra, the flag ratio is < 1. this number is then saved
in the header of the final fits file of the pipeline.

'''