import numpy as np, pandas as pd, matplotlib.pylab as plt, os, glob
from util_funcs import read_fits, plot_line

def line_ratio_indice(data, line="CaI"):
    '''
    Computes a ratio between the continuum and line flux of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,"NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 2
    else: window = 0.6

    ratio_arr = np.zeros(len(data)); center_flux_line_arr = np.zeros(len(data)); flux_continuum_arr = np.zeros(len(data)) 

    for i in range(len(data)):
        wv = data[i][0]; flux = data[i][1]
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
        wv = wv[wv_array]
        flux = flux[wv_array]
        #flux_normalized = flux/np.linalg.norm(flux)
        flux_normalized = (flux-np.min(flux))/(np.max(flux)-np.min(flux))

        flux_left = np.median(flux_normalized[:20])
        flux_right = np.median(flux_normalized[:-20])
        flux_continuum = np.median([flux_left,flux_right]) #median
        #print("Flux continuum: ",flux_continuum)
        
        wv_center_line = np.where((line_wv-window/30 < wv) & (wv < line_wv+window/30))
        flux_line = flux_normalized[wv_center_line]
        center_flux_line = np.median(flux_line)
        #print("Flux center line: ",center_flux_line)

        ratio = center_flux_line/flux_continuum 

        ratio_arr[i] = ratio
        center_flux_line_arr[i] = center_flux_line
        flux_continuum_arr[i] = flux_continuum
         
    return ratio_arr, center_flux_line_arr, flux_continuum_arr


def flag_ratio_RV_corr(files,instr):
    '''
    For each spectrum, run an interval of offsets and if the minimum ratio is not in [-0.02,0.02], raises a flag = 1.
    Flag = 0 is for good spectra. Then the flag ratio #0/#total is computed
    '''
    offset_list = np.linspace(-1,1,1001)
    flag_list = np.zeros((len(files)))

    for i,file in enumerate(files):
        wv, flux, hdr = read_fits(file,instrument=instr,mode="rv_corrected")
        #if i == 0: wv += 0.2 #just to fake a bad spectrum
        ratio_list = np.zeros_like(offset_list)
        for j,offset in enumerate(offset_list):
            ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv+offset,flux)], line="CaI")
            ratio_list[j]=ratio_arr

        min_ratio_ind = np.argmin(ratio_list)
        offset_min = offset_list[min_ratio_ind]
        #print(offset_min)
        if offset_min < -0.02 or offset_min > 0.02:
            flag_list[i] = 1
        else: flag_list[i] = 0

    good_spec_ind = np.where((flag_list == 0))
    N_good_spec = len(flag_list[good_spec_ind])
    flag_ratio = N_good_spec / len(flag_list)

    return flag_ratio, flag_list


target_save_name = "HD108147" #a linha do CaI está um caos, mas as do CaII H&K e Halpha estao ok?
instr = "HARPS"
files = glob.glob(os.path.join(f"teste_download_rv_corr/{target_save_name}/{target_save_name}_{instr}/ADP", "ADP*.fits"))

'''
#spectrum with highest ratio
wv, flux, hdr = read_fits(files[21],instrument=instr,mode="rv_corrected") 
plt.figure(7)
ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv,flux)], line="CaI")
print("Ratio: ",ratio_arr[0])
print("Flux center line: ",center_flux_line_arr[0])
plot_line(data=[(wv,flux)], line="CaI",lstyle="-")
plt.axhline(y=center_flux_line_arr,xmin=0,xmax=1,ls="--",ms=0.2)
plt.axhline(y=flux_continuum_arr,xmin=0,xmax=1,ls="--",ms=0.2)
plt.title("Spectrum nr 21, highest ratio")
'''
#first spectrum
wv, flux, hdr = read_fits(files[1],instrument=instr,mode="rv_corrected")
plt.figure(1)
ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv,flux)], line="CaI")
print("Ratio: ",ratio_arr[0])
print("Flux center line: ",center_flux_line_arr[0])
plot_line(data=[(wv,flux)], line="CaI",lstyle="-")
plt.axhline(y=center_flux_line_arr,xmin=0,xmax=1,ls="--",ms=0.2)
plt.axhline(y=flux_continuum_arr,xmin=0,xmax=1,ls="--",ms=0.2)
plt.title("Spectrum nr 0")
'''
#plot line to see it displaced
plt.figure(2)
offset_list = np.linspace(-0.2,0.2,3)
for offset in offset_list:
    plot_line(data=[(wv+offset,flux)], line="CaI",lstyle="-")
plt.title("Offsetting the spectrum nr 0 by 0.2 and -0.2 Angs")
'''
#plot offset vs ratio
#wv += 0.2
offset_list = np.linspace(-1,1,1001)
ratio_list = np.zeros_like(offset_list)
for i,offset in enumerate(offset_list):
    ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv+offset,flux)], line="CaI")
    ratio_list[i]=ratio_arr
plt.figure(3)
plt.plot(offset_list,ratio_list)
plt.axvline(x=0,ymin=0,ymax=1,ls="-",ms=0.1)
plt.axvline(x=0.025,ymin=0,ymax=1,ls="--",ms=0.1)
plt.axvline(x=-0.025,ymin=0,ymax=1,ls="--",ms=0.1)
plt.xlabel(r"Offset ($\lambda$)"); plt.ylabel("Ratio (Center of line / Continuum)")
'''
#plot offset vs I_CaII
#wv -= 0.2
from actin2.actin2 import ACTIN
actin = ACTIN()
indices= ['I_CaII', 'I_Ha06', 'I_NaI']
headers = {}
offset_list = np.linspace(-1,1,101)
ICaII = np.zeros_like(offset_list)
ICaII_err = np.zeros_like(offset_list)
for i,offset in enumerate(offset_list):
    spectrum = dict(wave=wv+offset, flux=flux)
    df_ind = actin.CalcIndices(spectrum, headers, indices).indices
    ICaII[i] = df_ind["I_CaII"]
    ICaII_err[i] = df_ind["I_CaII_err"]
plt.figure(4)
plt.errorbar(offset_list,ICaII,yerr=ICaII_err)
plt.xlabel(r"Offset ($\lambda$)"); plt.ylabel(r"$S_{CaII}$")
'''

#flag maker: for each spectrum, run an interval of offsets and if the minimum ratio is not in [-0.02,0.02], raises a flag
plt.figure(5)
flag_ratio, flag_list = flag_ratio_RV_corr(files,instr)
plt.hist(flag_list, bins=[-0.5, 0.5, 1.5], rwidth=0.8, align='mid', color='skyblue', edgecolor='black')
plt.ylabel("# Spectra"); plt.xlabel("Flag (0 is good)")
plt.xticks([0, 1])
plt.grid(axis='y')
print("Flag ratio: ",flag_ratio)
print(flag_list)

'''
# Write to a text file
with open(f"flag_ratios_{instr}.txt", "w") as file:
    file.write("##########################\n")
    file.write(f"Star: {target_save_name}\n")
    file.write(f"Flag ratio: {flag_ratio}\n")

    if flag_ratio == 1:
        file.write("Bad spectrum: None\n")
    else:
        bad_spec_indices = np.where(flag_list == 1)[0]
        for index in bad_spec_indices:
            file.write(f"Bad spectrum: {files[index]}\n")

'''
'''
plt.figure(6)
#plot spectrum vs ratio (all files)
ratio_list = []
for i,file in enumerate(files):
    wv, flux, hdr = read_fits(file,instrument=instr,mode="rv_corrected")
    #if i == 0: wv += 0.2
    ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv,flux)], line="CaI")
    ratio_list.append(ratio_arr[0])
plt.scatter(np.arange(0,len(files)),ratio_list)
plt.xlabel("Spectra"); plt.ylabel("Ratio (Center of line / Continuum)")
'''
plt.show()
'''
'''
'''
Plots the line, CaI, and computes the median of the first and last 20 points. then it takes the median of both values and that is considered the continuum flux

The flux of the line center is taken as the median of the points in an interval equal to  tabulated center +- window/30, where window for CaI is defined as 0.6 Angstrom

The ratio of the center and continuum flux is computed.

The flag is obtained by computing the ratio of fluxes for different offsets of wavelength (displacing the line). Naturally, the lowest ratio will be where the offset = 0,
if the spectrum is well corrected for RV. The flag is raised if the lowest ratio is not in a tolerance window of 0.02 Angstrom. Flag 1 = bad and 0 = good.

To apply this indicator to a star with several spectra, one has to compute the flag for each spectrum and then take the Number of 0 / Total spectra = #0 / #total
So if every spectrum is ok, the flag ratio will be = 1. if there is even one single not good spectra, the flag ratio is < 1. this number is then saved
in the header of the final fits file of the pipeline.

'''