import urllib, logging, pandas as pd, numpy as np
from astroquery.eso import Eso
from astroquery.simbad import Simbad
from util_funcs import get_gaiadr3, get_gaiadr2

df = pd.read_csv("sweet_cat_stars.csv")

df = df.sort_values(by=["N_HARPS"],ascending=False)
df = df[["Name", "hd", "Vmag", "Teff", "N_HARPS", "SNR_MIN_HARPS", "SNR_MAX_HARPS", "N_UVES", "SNR_MIN_UVES", "SNR_MAX_UVES"]]

df_cut = df.loc[(df["N_HARPS"]>=20) & (df["N_UVES"]>=12)]
print(df_cut.sort_values(by=["Teff"],ascending=True))

df_cut1 = df.loc[(df["hd"]=="47536")]
print(df_cut1)

star_list = list(df_cut["hd"])
star_list.append(list(df_cut1["hd"])[0])
star_list = ["HD"+x for x in star_list]
print(star_list)
print("Total stars to be studied: ",len(df_cut)+len(df_cut1))