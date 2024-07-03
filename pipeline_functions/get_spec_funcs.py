'''
This file contains functions to be used by the get_adp_spec function in the SAITAMA pipeline
'''
from astroquery.simbad import Simbad # type: ignore
import numpy as np
import glob
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time

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

def get_gaia_dr3_id(results_ids):
  for name in results_ids[::-1]:
    if "Gaia DR3 " in name[0]:
      return name[0].split(" ")[-1]
  return -1

def get_gaiadr3(name):
  '''Get the Gaia DR3 ID for the star and cleans it to be in the correct format.'''
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

def choose_snr(snr_arr, min_snr = 15, max_snr = 550):
  """Function to select the individual spectra given their respective minimum SNR"""
  #print("Min snr:", min_snr)
  #print("Max snr:", max_snr)
  index_cut = np.where((snr_arr > min_snr) & (snr_arr < max_snr))
  if len(index_cut[0]) == 0:
    print("Not enough SNR")
    return ([],)
  return index_cut

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

