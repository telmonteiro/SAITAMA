from astropy.io import fits
import matplotlib.pyplot as plt, numpy as np, math
'''
#file_name = "ADP.2021-04-13T13:26:21.358.fits"
#star = "HD22049"
file_name = "teste_download/HD209100/HD209100_ESPRESSO/ADP/ADP.2022-10-11T10:20:06.404.fits"
star="HD209100"
instrument = "ESPRESSO"

hdul = fits.open(file_name)
#print(fits.info(file_name))
print(repr(hdul[1].header))

flux = hdul[1].data["FLUX"][0]
wv = hdul[1].data["WAVE_AIR"][0]
plt.plot(wv,flux)
flux_el = hdul[1].data["FLUX_EL"][0]
print(wv)
print(flux)
valid_indices = ~np.isnan(flux_el)  # Get indices where flux is not NaN
wv = wv[valid_indices]     # Select corresponding wv values
flux_el = flux_el[valid_indices]
print(flux_el)
plt.plot(wv,flux_el)
plt.show()
'''
#####################3

file_name = "teste_download/HD46375/HD46375_ESPRESSO/ADP/ADP.2021-04-15T08:27:41.684.fits"
star = "HD46375"

hdul = fits.open(file_name)
print(fits.info(file_name))
print(repr(hdul[1].header))
print(hdul[1].data[0])

#flux_red = hdul[1].data["FLUX_REDUCED"][0]
#flux_red_err = hdul[1].data["ERR_REDUCED"][0]
wv = hdul[1].data["WAVE"][0]
flux = hdul[1].data["FLUX"][0]
flux_err = hdul[1].data["ERR"][0]
print(wv)
print(flux)
#print(flux_red)
print(flux_err)
#print(flux_red_err)
plt.plot(wv,flux)
#plt.plot(wv,flux_red)
plt.show()
