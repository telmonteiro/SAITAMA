from astropy.io import fits
from astropy.table import Table
from util_funcs import read_fits, read_bintable
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from actin2.actin2 import ACTIN
actin = ACTIN()

file_name = "ADP.2020-06-09T17:00:01.578.fits"
instrument = "UVES"
hdul = fits.open(file_name)
print(hdul.info())
print(hdul[0].header["MJD-OBS"]+2400000.5)

wv = hdul[1].data[0][0]
flux = hdul[1].data[0][1]

plt.plot(wv, flux)
plt.show()

columns = ['I_CaII', 'I_CaII_err', 'I_CaII_Rneg', 'I_Ha06', 'I_Ha06_err', 'I_Ha06_Rneg', 'I_NaI', 'I_NaI_err', 'I_NaI_Rneg', 'file', 'instr'] 
df = pd.DataFrame(columns=columns)
spectrum = dict(wave=wv, flux=flux)
indices= ['I_CaII', 'I_Ha06', 'I_NaI']
headers = {"file":file_name,"instr":instrument}

df_ind = actin.CalcIndices(spectrum, headers, indices).indices
df = df.append(pd.DataFrame([{**df_ind, **headers}]), ignore_index=True,sort=True)
print(df)

#wv, flux, hdr = read_fits(file_name,instrument,mode=None)
#print(hdr)
#MJD-OBS =      57993.277344162