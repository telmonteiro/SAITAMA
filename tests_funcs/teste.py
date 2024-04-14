from astropy.io import fits
import matplotlib.pyplot as plt
'''
file_name = "ADP.2021-04-13T13:26:21.358.fits"
star = "HD22049"
instrument = "ESPRESSO"

hdul = fits.open(file_name)
#print(fits.info(file_name))
print(repr(hdul[1].header))

flux_cal = hdul[1].data["FLUX_CAL"][0]
wv = hdul[1].data["WAVE_AIR"][0]
flux = hdul[1].data["FLUX_EL"][0]
print(wv)
print(flux)
print(flux_cal)
plt.plot(wv,flux)
plt.plot(wv,flux_cal)
plt.show()
'''
#####################3

file_name = "teste_download/HD16141/HD16141_UVES/ADP/ADP.2020-06-19T09:57:09.085.fits"
star = "HD16141"
instrument = "UVES"

hdul = fits.open(file_name)
print(fits.info(file_name))
print(repr(hdul[1].header))
print(hdul[1].data[0])

flux_red = hdul[1].data["FLUX_REDUCED"][0]
flux_red_err = hdul[1].data["ERR_REDUCED"][0]
wv = hdul[1].data["WAVE"][0]
flux = hdul[1].data["FLUX"][0]
flux_err = hdul[1].data["ERR"][0]
print(wv)
print(flux)
print(flux_red)
print(flux_err)
print(flux_red_err)
plt.plot(wv,flux)
plt.plot(wv,flux_red)
plt.show()