from __future__ import print_function, division
from astroquery.simbad import Simbad
import numpy as np, pandas as pd, matplotlib.pylab as plt, os, tarfile, glob
from astropy.io import fits
from PyAstronomy import pyasl
from astroquery.eso import Eso
from astropy.table import Table, vstack
from astropy.time import Time
from PyAstronomy.pyTiming import pyPeriod

eso = Eso()

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

def plot_RV_indices(star,df,indices,save, path_save):
    """
    Plot RV and indices given as a function of time
    """
    plt.figure(figsize=(6, (len(indices)+1)*2))
    plt.suptitle(star, fontsize=14)
    plt.subplot(len(indices)+1, 1, 1)
    if "rv_err" not in df.columns: yerr = 0
    else: yerr = df.rv_err
    plt.errorbar(df.bjd - 2450000, df.rv, yerr, fmt='k.')
    plt.ylabel("RV [m/s]")
    #print(indices)
    for i, index in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        plt.ylabel(index)
        plt.errorbar(df.bjd - 2450000, df[index], df[index + "_err"], fmt='k.')
    plt.xlabel("BJD $-$ 2450000 [days]")
    plt.subplots_adjust(top=0.95)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

#########################
    
def stats_indice(star,cols,df):
    """
    Return pandas data frame with statistical data on the indice(s) given: max, min, mean, median, std and N (number of spectra)
    """
    df_stats = pd.DataFrame(columns=["star","indice","max","min","mean","median","std","time_span","N_spectra"])
    if len(cols) == 1:
            row = {"star":star,"column":cols,
                "max":max(df[cols]),"min":min(df[cols]),
                "mean":np.mean(df[cols]),"median":np.median(df[cols]),
                "std":np.std(df[cols]),"time_span":max(df["bjd"])-min(df["bjd"]),
                "N_spectra":len(df[cols])}
            df_stats.loc[len(df_stats)] = row
    elif len(cols) > 1:
        for i in cols:
            if i != "rv":
                indices = df[df[i+"_Rneg"] < 0.001].index
                data = df.loc[indices, i]
            else:
                data = df[i]
            row = {"star": star, "indice": i,
                   "max": max(data), "min": min(data),
                   "mean": np.mean(data), "median": np.median(data),
                   "std": np.std(data), "time_span": max(df["bjd"]) - min(df["bjd"]),
                   "N_spectra": len(data)}
            df_stats.loc[len(df_stats)] = row

    else:
        print("ERROR: No columns given")
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
    with fits.open(os.path.join(folder_path, files[0])) as hdul: 
        hdr = hdul[0].header
        date_bjd = np.around(hdr["HIERARCH ESO DRS BJD"])
        hdr["HIERARCH ESO DRS BJD"] == date_bjd

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

########################

def calc_fits_wv_1d(hdr, key_a='CRVAL1', key_b='CDELT1', key_c='NAXIS1'):
    '''
    Compute wavelength axis from keywords on spectrum header.
    '''
    try:
        a = hdr[key_a]; b = hdr[key_b]
    except KeyError:
        a = hdr["WAVELMIN"]*10; b = hdr["WAVELMAX"]*10
    try: 
        c = hdr[key_c]
    except KeyError:
        c = hdr["NELEM"]

    return a + b * np.arange(c)

########################

def correct_spec_rv(wv, rv, units):
    '''
    Correct wavelength of spectrum by the RV of the star with a Doppler shift.
    '''
    c = 299792.458 #km/s
    if units == "m/s":
        c *= 1000
    #delta_wv = wv * rv / c #shift
    #wv_corr = wv + delta_wv #o problema era o sinal... é wv - delta_wv
    wv_corr = wv / (1+rv/c)
    delta_wv = wv - wv_corr
    return wv_corr, np.mean(delta_wv)

########################

def read_fits(file_name,instrument,mode):
    '''
    Read fits file and get header and data. Varies if instrument is HARPS, ESPRESSO, UVES or FEROS. missing espresso
    '''
    hdul = fits.open(file_name)
    if instrument == "HARPS":
        if mode == "raw":
            if "s1d_A" in file_name:
                flux = hdul[0].data
                header = hdul[0].header
                wv = calc_fits_wv_1d(header)
            elif "ADP" in file_name:
                wv = hdul[1].data[0][0]
                flux = hdul[1].data[0][1]
                bjd = hdul[0].header["HIERARCH ESO DRS BJD"]
                header = hdul[0].header
                header["HIERARCH ESO DRS BJD"] = bjd
        elif mode == "rv_corrected":
            wv = hdul[0].data[0]
            flux = hdul[0].data[1]
            header = hdul[0].header

    elif instrument == "UVES" or instrument == "FEROS":
        wv = hdul[1].data[0][0]
        flux = hdul[1].data[0][1]
        header = hdul[1].header
    
    else:
        flux = hdul[0].data
        header = hdul[0].header
        wv = calc_fits_wv_1d(header)
    hdul.close()

    return wv, flux, header

########################

def get_rv_ccf(star, stellar_wv, stellar_flux, stellar_header, template_hdr, template_spec, drv, units, instrument):
    '''
    Uses crosscorrRV function from PyAstronomy to get the CCF of the star comparing with a spectrum of the Sun.
    Returns the BJD in days, the RV, the max value of the CCF, the list of possible RV and CC, as well as the flux and raw wavelength.
    To maximize the search for RV and avoid errors, the script searches the RV in SIMBAD and makes a reasonable interval. 
    '''
    
    if instrument == "HARPS":
        bjd = stellar_header["HIERARCH ESO DRS BJD"] #may change with instrument
    else: bjd = None

    #get wavelength and flux of both sun (template) and star
    w = stellar_wv; f = stellar_flux
    tw = calc_fits_wv_1d(template_hdr); tf = template_spec

    try:
        rv_simbad = _get_simbad_data(star=star, alerts=False)["RV_VALUE"] #just to minimize the computational cost
        rvmin = rv_simbad - 1; rvmax = rv_simbad + 1
    except:
        rvmin = -200; rvmax = 200

    #get the cross-correlation
    skipedge_values = [500, 1000, 5000, 20000, 50000, 80000]

    for skipedge in skipedge_values:
        try:
            rv, cc = pyasl.crosscorrRV(w=w, f=f, tw=tw,
                                    tf=tf, rvmin=rvmin, rvmax=rvmax, drv=drv, skipedge=skipedge)
            break  # Break out of the loop if successful
        except Exception as e:
            print(f"Error with skipedge={skipedge}: {e}")
            # Continue to the next iteration

    #index of the maximum cross-correlation function
    maxind = np.argmax(cc)
    radial_velocity = rv[maxind]

    if units == "m/s":
        radial_velocity *= 1000
        rv *= 1000
    
    return bjd, radial_velocity, cc[maxind], np.around(rv,0), cc, w, f

########################
'''
These two functions get the Gaia DR2 ID for the star and cleans it to be in the correct format.
'''
def get_gaia_dr2_id(results_ids):
  for name in results_ids[::-1]:
    if "Gaia DR2 " in name[0]:
      return name[0].split(" ")[-1]
  return -1

def get_gaiadr2(name):
  customSimbad=Simbad()
  
  if name[-2:] == " A":
    name =  name[:-2]
  if "(AB)" in name:
    name = name.replace("(AB)", "")
  if "Qatar" in name:
    name = name.replace("-","")

  result_ids = customSimbad.query_objectids(name)
  if result_ids is None:
    gaiadr2 = -1
  else:

    gaiadr2 = get_gaia_dr2_id(result_ids)
  return gaiadr2

########################
'''
These two functions get the Gaia DR3 ID for the star and cleans it to be in the correct format.
'''
def get_gaia_dr3_id(results_ids):
  for name in results_ids[::-1]:
    if "Gaia DR3 " in name[0]:
      return name[0].split(" ")[-1]
  return -1

def get_gaiadr3(name):
  customSimbad=Simbad()
  
  if name[-2:] == " A":
    name =  name[:-2]
  if "(AB)" in name:
    name = name.replace("(AB)", "")
  if "Qatar" in name:
    name = name.replace("-","")

  result_ids = customSimbad.query_objectids(name)
  if result_ids is None:
    gaiadr3 = -1
  else:

    gaiadr3 = get_gaia_dr3_id(result_ids)
  return gaiadr3

########################

def choose_snr(snr_arr, min_snr = 30, max_snr = 550):
  """Function to select the individual spectra given their respective minimum SNR"""
  #print("Min snr:", min_snr)
  #print("Max snr:", max_snr)
  index_cut = np.where((snr_arr > min_snr) & (snr_arr < max_snr))
  if len(index_cut[0]) == 0:
    print("Not enough SNR")
    return ([],)
  return index_cut

########################

def untar_ancillary_harps(path_download):
  """Function to uncompress and copy the s1d and ccd files from the harps ancillary data"""
  files_tar = glob.glob(path_download+"*.tar")

  for f in files_tar:
    file = tarfile.open(f)
    file.extractall(path=path_download)
  if not os.path.isdir(path_download+"S1D_CCF"):
    os.mkdir(path_download+"S1D_CCF")

  os.system("mv "+path_download+"data/*/*/*s1d_A*.fits "+path_download+"S1D_CCF/")
  os.system("mv "+path_download+"data/*/*/*ccf*_A*.fits "+path_download+"S1D_CCF/")

########################

def check_downloaded_data(path_download):
  """Report on the downloaded data: showing the potential snr for the combined spectrum"""
  files_fits = glob.glob(path_download+"*.fits")
  if len(files_fits) > 0:
    snr_arr = np.array([fits.getheader(filef)["SNR"] for filef in files_fits])
    max_down_snr = np.max(snr_arr)
    min_down_snr = np.min(snr_arr)
    print ("Download: Min SNR %7.1f - Max SNR %7.1f; nspec: %d" % (min_down_snr, max_down_snr, len(snr_arr)))
  else:
    print("No downloaded files? All private?")
    snr_arr = []
  return snr_arr

########################

def plot_line(data, line, lstyle = "-"):
    '''
    Plots the spectra used in the position of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,
             "NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 2
    else: window = 0.6

    for array in data:
        wv = array[0]; flux = array[1]
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
        wv = wv[wv_array]
        flux = flux[wv_array]
        #flux_normalized = flux/np.linalg.norm(flux)
        flux_normalized = (flux-np.min(flux))/(np.max(flux)-np.min(flux))
        plt.plot(wv, flux_normalized, lstyle)
        plt.axvline(x=wv[19],ymin=0,ymax=1,ls="--",ms=0.2)
        plt.axvline(x=wv[-19],ymin=0,ymax=1,ls="--",ms=0.2)
        plt.axvline(x=line_wv-window/30,ymin=0,ymax=1,ls="--",ms=0.2)
        plt.axvline(x=line_wv+window/30,ymin=0,ymax=1,ls="--",ms=0.2)

    plt.axvline(x=line_wv,ymin=0,ymax=1,ls="-",ms=0.2)
    

    plt.xlabel(r"Wavelength ($\AA$)"); plt.ylabel("Normalized Flux")
    plt.title(f"{line} line")

#######################

def select_best_spectra(spectra_table, max_spectra=200):
    '''
    Selects the best spectra from the ESO data base while maintaining a good time span (normally the maximum).
    Groups my month and year, orders each group by SNR. 
    Then iterates for each group, adding to the new table the best SNR spectra until the maximum number of spectra is achieved.
    '''
    #Astropy Time to handle date parsing, allowing for invalid dates
    date_obs_np = Time(spectra_table['Date Obs'], format='isot', scale='utc')

    #filter out invalid dates
    valid_mask = ~date_obs_np.mask
    spectra_table = spectra_table[valid_mask]

    #extract year and month
    year_month = np.array([(d.datetime64.astype('datetime64[Y]').item(), d.datetime64.astype('datetime64[M]').item()) for d in date_obs_np])

    #add 'year' and 'month' columns to the table
    spectra_table['year'] = np.array([str(x)[:4] for x in year_month[:, 0]])
    spectra_table['month'] = np.array([str(x)[5:-3] for x in year_month[:, 1]])

    #group by 'year' and 'month'
    grouped = spectra_table.group_by(['year', 'month'])

    selected_spectra = Table()
    excess_spectra = max_spectra

    max_group_length = max(len(group) for group in grouped.groups)

    for i in range(max_group_length):
        for group in grouped.groups: #for each month of each year
            if i < len(group):  #check if the current index is within the length of the group
                sorted_group = group[np.argsort(group['SNR'])[::-1]] #sort by descending order of SNR

                if excess_spectra > 0:
                    selected_spectra = vstack([selected_spectra, sorted_group[i:i + 1]])
                    excess_spectra -= 1
                else:
                    break
        
        #print(f"Iteration {i + 1}: selected_spectra length = {len(selected_spectra)}, excess_spectra = {excess_spectra}")

        if excess_spectra <= 0:
            break

    #calculate and print time span
    min_date = min(date_obs_np[valid_mask]).datetime64.astype('datetime64[D]').item()
    max_date = max(date_obs_np[valid_mask]).datetime64.astype('datetime64[D]').item()
    days_span = (max_date - min_date).total_seconds() / (24 * 3600)
    print(f"Start Date: {min_date}")
    print(f"End Date: {max_date}")
    print(f"Days Span: {days_span} days")

    return selected_spectra

################################

def line_ratio_indice(data, line="CaI"):
    '''
    Computes a ratio  between the continuum and line flux of a reference line to check if everything is alright.
    '''
    lines_list = {"CaIIK":3933.664,"CaIIH":3968.47,"Ha":6562.808,"NaID1":5895.92,
             "NaID2":5889.95,"HeI":5875.62,"CaI":6572.795,"FeII":6149.240}
    line_wv = lines_list[line]
    
    if line in ["CaIIK","CaIIH"]: window = 12
    elif line in ["Ha","NaID1","NaID2"]: window = 2
    else: window = 1 

    flag_ind = []
    for i,array in enumerate(data):
        wv = array[0]; flux = array[1]
        wv_array = np.where((line_wv-window < wv) & (wv < line_wv+window))
        wv = wv[wv_array]
        flux = flux[wv_array]
        flux_normalized = flux/np.linalg.norm(flux)

        flux_left = flux_normalized[0]
        flux_right = flux_normalized[-1]
        flux_continuum = (flux_left+flux_right)/2 #mean
        flux_line = min(flux_normalized)
        ratio = flux_line/flux_continuum
        if ratio > 0.5:
            flag_ind.append(i)
        
    return flag_ind

#################################

def negative_flux_treatment(f, method):
    '''
    This function is no longer used, since the negative flux is only a problem if it is in the lines
    where the activity indices are computed. 
    Nonetheless, one can choose wether so skip the spectrum entirely or to replace the negative values by zero.
    Also not sure if this works correctly because it is no longer needed.
    '''
    if method == "skip":
        # skips the spectra that have negative fluxes           
        neg_flux_position = np.where(f < 0)[0]
        if neg_flux_position.shape[0] != 0:
            return neg_flux_position
        
    elif method == "zero_pad":
        #replace negative flux by zero flux
        f = np.where(f < 0, 1e-5, f)
        return f
    
#################################
    
def general_fits_file(data_array, stats_df, df, folder_path, min_snr, max_snr, instr, period, period_err):
    '''
    Takes the data array that consists in a 3D cube containing the wavelength and flux for each spectrum used and
    the data frame with the statistics.
    '''
    hdr = fits.Header() 

    star_id = stats_df["star"][0]; time_span = stats_df["time_span"][0]; N_spectra = stats_df["N_spectra"][0]
    dict_hdr = {"STAR_ID":[star_id,'Star ID in HD catalogue'],
                "INSTR":[instr,"Instrument used"],
                "TIME_SPAN":[time_span, 'Time span in days between first and last observations used'],
                "N_SPECTRA":[N_spectra,"Number of spectra used"],
                "SNR_MIN":[min_snr,"Minimum SNR"],
                "SNR_MAX":[max_snr,"Maximum SNR"],
                "PERIOD_I_CaII":[period,"Period of CaII activity index"],
                "PERIOD_I_CaII_ERR":[period_err,"Error of period of CaII activity index"],
                "COMMENT":["Spectra based on SNR - time span trade-off","Comment"],
                "COMMENT1":["RV obtained from CCF measure","Comment"],
                "COMMENT2":["3D data of wv (Angs) and flux of each spectrum","Comment"]}

    indices = ['I_CaII', 'I_Ha06', 'I_NaI', 'rv']
    stats = ["max","min","mean","median","std"]

    for i,ind in enumerate(indices):
        for col in stats:
            stat = stats_df[col][i]
            if col == "rv": comment = f"{col} of {ind.upper()} (m/s)"
            else: comment = f"{col} of {ind}"
            dict_hdr[ind.upper()+"_"+col.upper()] = [stat,comment]
    
    for keyword in dict_hdr.keys():
        hdr.append(("HIERARCH "+keyword, dict_hdr[keyword][0], dict_hdr[keyword][1]), end=True)

    #hdu = fits.PrimaryHDU(data_array, header=hdr) #CAN'T SAVE THE DATA ARRAY BECAUSE IT'S A LIST WITH ARRAYS OF DIFFERENT SIZES INSIDE REEEEE

    # Assuming 'df' is your DataFrame
    #numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns

    # Convert the DataFrame to an Astropy Table
    #table = Table.from_pandas(numeric_df)

    hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([hdu])

    hdul.writeto(f"{folder_path}", overwrite=True)
    hdul.close()

#################################
    
def gls_periodogram(star, I_CaII, I_CaII_err, bjd, print_info, save, path_save):
    # Compute the GLS periodogram with default options. Choose Zechmeister-Kuerster normalization explicitly
    clp = pyPeriod.Gls((bjd - 2450000, I_CaII, I_CaII_err), norm="ZK", verbose=print_info,
                       Pbeg=3000, Pend=4000,ofac=30)
    dic_clp = clp.info(noprint=True)
    period = dic_clp["best_sine_period"]
    period_err = dic_clp["best_sine_period_err"]
    # Define FAP levels of 10%, 5%, and 1%
    fapLevels = np.array([0.1, 0.05, 0.01])
    # Obtain the associated power thresholds
    plevels = clp.powerLevel(fapLevels)

    plt.figure(1, figsize=(13, 4))
    plt.suptitle(f"GLS periodogram for {star} I_CaII", fontsize=12)

    # and plot power vs. frequency.
    plt.subplot(1, 2, 1)
    plt.xlabel("Period"); plt.ylabel("Power")
    plt.title("Power vs Frequency for GLS Periodogram")
    plt.plot(1/clp.freq, clp.power, 'b-')
    # Add the FAP levels to the plot
    for i in range(len(fapLevels)):
        plt.plot([min(clp.freq), max(clp.freq)], [plevels[i]]*2, '--',
                label="FAP = %4.1f%%" % (fapLevels[i]*100))
    plt.xlim([100, 5000])
    plt.legend()

    plt.subplot(1, 2, 2)
    timefit = np.linspace(min(bjd - 2450000),max(bjd - 2450000),1000)
    plt.plot(timefit,clp.sinmod(timefit),label="fit")
    plt.errorbar(bjd - 2450000,I_CaII,yerr=I_CaII_err,fmt='.', color='k',label='data')
    plt.xlabel('BJD $-$ 2450000 [days]'); plt.ylabel(r'$S_\mathrm{CaII}$')
    plt.title("Fitting the data with GLS")
    plt.legend()

    plt.subplots_adjust(top=0.85)
    if save == True:
        plt.savefig(path_save, bbox_inches="tight")

    return round(period,3), round(period_err,3)

#################################

def read_bintable(file,print_info=False):
    '''
    Simple function to read the header and BinTable from the fits file that contains the statistics in the header and the data frame as BinTable
    '''
    hdul = fits.open(file)
    if print_info == True: hdul.info()
    hdr = hdul[1].header
    table = hdul[1].data
    astropy_table = Table(table)
    data_dict = astropy_table.to_pandas().to_dict()
    df = pd.DataFrame(data_dict)
    return df, hdr