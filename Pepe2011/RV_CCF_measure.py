import glob, os, tqdm, time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from util_funcs import correct_spec_rv, read_fits, get_rv_ccf

stars = ["HD20794", "HD85512", "HD192310"]

for star in stars:
    files = glob.glob(os.path.join(f"pepe_2011/{star}/avg_spec/", "*s1d_A.fits"))
    stellar_spectrum_filenames = tqdm.tqdm(files)

    sun_template_spectrum, sun_header = read_fits(file_name="Sun1000.fits")

    radial_velocities = []; bjd_list = []
    '''
    fig, axs = plt.subplots(1,1)
    fig1, axs1 = plt.subplots(1,1)
    '''
    bjd_list = []; rv_list = []; cc_list = []; shift_wv_list = []

    for i, spectrum_filename in enumerate(stellar_spectrum_filenames):
        time.sleep(0.001)

        stellar_spectrum, stellar_header = read_fits(file_name=spectrum_filename)

        drv = 0.01 #step to search rv
        bjd, radial_velocity, cc_max, rv, cc, w, f = get_rv_ccf(star, spectrum_filename, 
                                                template_hdr = sun_header, template_spec = sun_template_spectrum, drv = drv, units = "m/s")

        rv_list.append(radial_velocity); bjd_list.append(bjd); cc_list.append(cc_max)
        '''
        # Plot the cross-correlation function for each spectrum
        if i == 0:
            axs.plot(rv, cc)
            axs.plot(radial_velocity, cc_max, 'ro', label=f"{radial_velocity} m/s")
        '''
        wv_corr, mean_delta_wv = correct_spec_rv(w, radial_velocity, units = "m/s")
        shift_wv_list.append(mean_delta_wv)

        data = np.vstack((wv_corr, f))
        file_path = spectrum_filename.replace('avg_spec', 'avg_spec_rv_corr')

        header = fits.Header()
        header['BJD'] = bjd
        header["rv"] = radial_velocity

        hdu = fits.PrimaryHDU(data, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(f"{file_path}", overwrite=True)

    df = pd.DataFrame({"BJD":bjd_list,"rv":rv_list,"cc":cc_list, "mean_wv_shift":shift_wv_list})
    df.to_csv(f"pepe_2011/results/{star}_rv.csv")

    '''
    axs.set_xlabel('Radial Velocity (m/s)')
    axs.set_ylabel('Cross-correlation')
    axs.set_title('Cross-correlation with the Sun Template')
    axs.legend()

    axs1.scatter(bjd_list, rv_list)
    axs1.set_xlabel("BJD [days]")
    axs1.set_ylabel("Radial Velocity (m/s)")
    plt.show()
    '''