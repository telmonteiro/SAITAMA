import os, astropy
from astropy.io import fits
import numpy as np
from util_funcs import create_average_spectrum
from collections import defaultdict
"""
this script aims to get an average spectrum for each night of observation in the folder of the stars
    1 - the files intended are HARPS.YYYY-MM-DDThour_extension.fits
    2 - the files are inside pepe_2011/star/data/reduced/
"""
stars_list = ["HD20794","HD85512","HD192310"] #pepe et al 2011 stars

#can be in two folders
path = ["main folder", "data/reduced"]

for i, star in enumerate(stars_list):
    print(f"-------------------------------------------------------\n PROCESSING {star}\n -------------------------------------------------------")
    output_directory = "/home/telmo/PEEC/ACTIN2/pepe_2011/{}/avg_spec/".format(star)
    if star == "HD20794": extension_list = ["s1d_A","bis_G2_A","ccf_G2_A","e2ds_A","e2ds_B","s1d_B"] #could use any extension, but the useful one is s1d_A
    else: extension_list = ["s1d_A","bis_K5_A","ccf_K5_A","e2ds_A","e2ds_B","s1d_B"]

    for p in path:

        if p == "main folder": 
            root_directory = "/home/telmo/PEEC/ACTIN2/pepe_2011/{}/".format(star)

            #group files by observation date
            file_groups = defaultdict(list)
            for extension in extension_list:  #assuming the date is extracted correctly for extension files
                files = [f for f in os.listdir(root_directory) if f.startswith(f'HARPS.') and f.endswith(f'{extension}.fits')]
                for file in files:
                    date = file.split('.')[1].split('T')[0]  #extract date from the filename
                    file_groups[date].append(file)

            #process each group
            for i, ext in enumerate(extension_list):
                for date, files in file_groups.items():
                    files = [x for x in files if ext in x]
                    #print(files)

                    output_filename = f"average_{date}_{ext}.fits"
                    output_path = os.path.join(output_directory, output_filename)

                    avg_data, hdr = create_average_spectrum(folder_path=root_directory, files=files, extension=None)
                    if type(avg_data) != int and type(hdr) != int: #crude way of checking if its in the correct type
                        fits.writeto(output_path, avg_data, header=hdr, overwrite=True, output_verify='ignore')
                        print(f"{date} data processed")

        elif p == "data/reduced": 
            root_directory = "/home/telmo/PEEC/ACTIN2/pepe_2011/{}/data/reduced/".format(star)

            #iterate through folders
            for folder_name in os.listdir(root_directory):
                folder_path = os.path.join(root_directory, folder_name)

                #check if it's a directory and not a file
                if os.path.isdir(folder_path):
                    for extension in extension_list:
                        output_filename = f"average_{folder_name}_{extension}.fits"
                        output_path = os.path.join(output_directory, output_filename)

                        avg_data, hdr = create_average_spectrum(folder_path=folder_path, files=None, extension=extension)
                        if type(avg_data) != int and type(hdr) != int:
                            fits.writeto(output_path, avg_data, header=hdr, overwrite=True)
                            print(f"{date} data processed")
