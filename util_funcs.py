from astroquery.simbad import Simbad
import numpy as np, pandas as pd, matplotlib.pylab as plt, os
from astropy.io import fits

def _get_simbad_data(star, alerts=True):
    """Get selected Simbad data for 'star'.
    Simbad url: http://simbad.cds.unistra.fr/simbad/
    """
    customSimbad = Simbad()
    #Simbad.list_votable_fields() # Uncomment to check all fields available
    customSimbad.add_votable_fields('flux(V)')
    customSimbad.add_votable_fields('flux_error(V)')
    customSimbad.add_votable_fields('flux(B)')
    customSimbad.add_votable_fields('plx')
    customSimbad.add_votable_fields('plx_error')
    customSimbad.add_votable_fields('sptype')
    customSimbad.add_votable_fields('otype')
    customSimbad.add_votable_fields('rv_value')
    customSimbad.add_votable_fields('otypes')
    customSimbad.add_votable_fields('pmdec')
    customSimbad.add_votable_fields('pmra')
    customSimbad.get_votable_fields()
    err_msg = None
    try:
        query = customSimbad.query_object(star)
    except:
        err_msg = f"*** ERROR: Could not identify {star}."
        if alerts:
            print(err_msg)
        return None, err_msg

    #print(list(query));sys.exit() # Uncomment to check all query options
    keys = [
        'FLUX_V',
        'FLUX_ERROR_V',
        'B-V',
        'PLX_VALUE', # mas
        'PLX_ERROR',
        'SP_TYPE',
        'OTYPE',
        'RV_VALUE', # km/s
        'OTYPES',
        'PMDEC', # mas/yr
        'PMRA' # mas/yr
            ]

    results = {}
    const = np.ma.core.MaskedConstant

    def no_value_found(key, star):
        err_msg = f"*** ERROR: {star}: No values of {key} in Simbad."
        if alerts:
            print(err_msg)
        return err_msg

    for key in keys:
        if key == 'B-V':
            if not isinstance(query['FLUX_V'][0], const) or not isinstance(query['FLUX_V'][0], const):
                results[key] = query['FLUX_B'][0]-query['FLUX_V'][0]

            else:
                err_msg = no_value_found(key, star)
                results[key] = float('nan')

        elif not query[key][0]:
            err_msg = no_value_found(key, star)
            results[key] = float('nan')

        elif isinstance(query[key][0], const):
            err_msg = no_value_found(key, star)
            results[key] = float('nan')

        elif isinstance(query[key][0], bytes):
            results[key] = query[key][0].decode('UTF-8')

        else:
            results[key] = query[key][0]

    return results

##########################

def plot_RV_indices(star,df,indices,save):
    """
    Plot RV and indices given as a function of time
    """
    plt.figure(figsize=(6, (len(indices)+1)*2))
    plt.suptitle(star, fontsize=14)
    plt.subplot(len(indices)+1, 1, 1)
    plt.errorbar(df.bjd - 2450000, df.rv, df.rv_err, fmt='k.')
    plt.ylabel("RV [m/s]")
    for i, index in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        plt.ylabel(index)
        plt.errorbar(df.bjd - 2450000, df[index], df[index + "_err"], fmt='k.')
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.subplots_adjust(top=0.95)
    if save == True:
        plt.savefig(f"{star}.png", bbox_inches="tight")

#########################
    
def stats_indice(star,indices,df):
    """
    Return pandas data frame with statistical data on the indice(s) given: max, min, mean, median, std and N (number of spectra)
    """
    df_stats = pd.DataFrame(columns=["star","indice","max","min","mean","median","std","N_spectra"])

    if len(indices) == 1:
        row = {"star":star,"indice":indices,
            "max":max(df[indices]),"min":min(df[indices]),
            "mean":np.mean(df[indices]),"median":np.median(df[indices]),
            "std":np.std(df[indices]),"N_spectra":len(df[indices])}
        df_stats.loc[len(df_stats)] = row
    elif len(indices) > 1:
        for i in indices:
            row = {"star":star,"indice":i,
            "max":max(df[i]),"min":min(df[i]),
            "mean":np.mean(df[i]),"median":np.median(df[i]),
            "std":np.std(df[i]),"N_spectra":len(df[i])}
            df_stats.loc[len(df_stats)] = row
    else:
        print("ERROR: No indices given")
        df_stats = None
    
    return df_stats

#########################

def find_s1d_A(directory):
    """
    Find files inside folders that match the requirement (s1d_A)
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("s1d_A.fits"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

########################

def create_average_spectrum(folder_path, extension, files):
    '''
    This function creates an average spectrum per day. It has two kind of paths because it is used two times, with a difference:
        - fits files are stored in the main folder of the star (before data/reduced)
        - fits files are stored in the data/reduced folder
    I don't know why it happens but it seems that the data/reduced files are the most recent ones, 2010-
    '''
    if files == None:
        files = [f for f in os.listdir(folder_path) if f.endswith("{}.fits".format(extension))]

    if not files:
        print(f"No {extension} fits files found in {folder_path}")
        return 0,0

    #header of first observation. using the first obs as the final header just because it is easier
    with fits.open(os.path.join(folder_path, files[0])) as hdul: hdr = hdul[0].header

    #load each data and pad or truncate to a common length
    data_list = [fits.getdata(os.path.join(folder_path, file), ext=0) for file in files]
    #make sure all arrays have same shape
    min_shape = min(data.shape for data in data_list)
    
    if files == None:
        #check if the arrays are 1D or 2D
        if len(min_shape) == 1: 
            data_padded = np.array([data[:min_shape[0]] for data in data_list])
        elif len(min_shape) == 2:
            data_padded = np.array([data[:min_shape[0], :min_shape[1]] for data in data_list])
        else:
            print("Unexpected dimensionality in the data.")
    else:
        data_padded = np.array([data[:min_shape[0]] for data in data_list])

    #compute average data
    avg_data = np.mean(data_padded, axis=0)

    return avg_data, hdr

########################

def sigma_clip(df, cols, sigma):
    '''
    Rough sigma clipping of a data frame.
    '''
    for col in cols:
        mean= df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - sigma * std) & (df[col] <= mean + sigma * std)]
    return df