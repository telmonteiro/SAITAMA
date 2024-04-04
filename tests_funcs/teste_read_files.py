from astropy.io import fits
from astropy.table import Table
from util_funcs import read_fits, read_bintable, plot_line, get_rv_ccf, correct_spec_rv, line_ratio_indice
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from actin2.actin2 import ACTIN
actin = ACTIN()

file_name = "teste_download/HD16141/HD16141_UVES/ADP/ADP.2020-06-19T09:57:09.085.fits"
instrument = "UVES"
star = "HD16141"
#file_name = "teste_download/HD20794/HD20794_UVES/ADP/ADP.2020-06-09T17:00:01.514.fits"
#file_name = "ADP.2021-04-13T13:28:15.797.fits"

#file_name = "ADP.2021-04-13T13:26:21.358.fits"
#star = "HD22049"
#instrument = "ESPRESSO"
wv, flux, hdr = read_fits(file_name,instrument,mode="raw")

plt.figure(6)
plt.plot(wv,flux)

plt.figure(1)
plot_line([(wv,flux)], line="CaI")
plt.figure(2)
plot_line([(wv,flux)], line="Ha")

sun_template_wv, sun_template_flux, sun_header = read_fits(file_name="Sun1000.fits",instrument=None, mode=None) #template spectrum for RV correction
bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star = star, stellar_wv = wv, stellar_flux = flux, stellar_header = hdr,
                                                        template_hdr = sun_header, template_spec = sun_template_flux, 
                                                        drv = .1, units = "m/s", instrument=instrument)
print(radial_velocity)
plt.figure(5)
plt.plot(rv,cc)

wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")

offset_list = np.linspace(-1,1,1001)
flag_list = np.zeros((1))
ratio_list = np.zeros_like(offset_list)
for j,offset in enumerate(offset_list):
    ratio_arr, center_flux_line_arr, flux_continuum_arr = line_ratio_indice([(wv+offset,flux)], line="CaI")
    ratio_list[j]=ratio_arr
min_ratio_ind = np.argmin(ratio_list)
offset_min = offset_list[min_ratio_ind]
if offset_min < -0.03 or offset_min > 0.03:
    flag_list[0] = 1
else: flag_list[0] = 0
print(flag_list)

plt.figure(3)
plot_line([(wv_corr,f)], line="CaI")
plt.figure(4)
plot_line([(wv_corr,f)], line="Ha")
plt.figure(7)
plot_line([(wv_corr,f)], line="CaIIK")
plt.figure(8)
plot_line([(wv_corr,f)], line="CaIIH")
plt.show()

columns = ['I_CaII', 'I_CaII_err', 'I_CaII_Rneg', 'I_Ha06', 'I_Ha06_err', 'I_Ha06_Rneg', 'I_NaI', 'I_NaI_err', 'I_NaI_Rneg', 'file', 'instr'] 
df = pd.DataFrame(columns=columns)
spectrum = dict(wave=wv_corr, flux=flux)
indices= ['I_CaII', 'I_Ha06', 'I_NaI']
headers = {"file":file_name,"instr":instrument}

df_ind = actin.CalcIndices(spectrum, headers, indices).indices
df = df.append(pd.DataFrame([{**df_ind, **headers}]), ignore_index=True,sort=True)
print(df)

#wv, flux, hdr = read_fits(file_name,instrument,mode=None)
#print(hdr)