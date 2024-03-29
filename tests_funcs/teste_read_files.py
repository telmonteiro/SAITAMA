from astropy.io import fits
from astropy.table import Table
from util_funcs import read_fits, read_bintable

file_name = "ADP.2020-06-09T17:00:01.578.fits"
instrument = "UVES"
hdul = fits.open(file_name)
print(hdul.info())
print(hdul[0].header)
print(hdul[1].data[0][0])

table = hdul[1].data
astropy_table = Table(table)
print(astropy_table)
#wv, flux, hdr = read_fits(file_name,instrument,mode=None)
#print(hdr)
#MJD-OBS =      57993.277344162