import numpy as np

spec_hdrs = dict(
    bjd = "BJD",
    rv = "rv"
)

class test_instrument:
    def __init__(self, hdu, **spec_kw):

        instr = 'test'

        # Create dictionary to hold the spectrum data
        spec = dict()

        # Obtain the flux and headers from fits HDU
        spec['flux'] = hdu[0].data[1]
        spec['wave'] = hdu[0].data[0]
        hdr = hdu[0].header
        hdu.close()

        # Get spectrum selected header values:
        headers = {}
        for key, hdr_id in zip(spec_hdrs.keys(), spec_hdrs.values()):
            try:
                headers[key] = hdr[hdr_id]
            except KeyError:
                headers[key] = None

        headers['instr'] = instr

        # Flux photon noise
        spec['flux_err'] = np.sqrt(abs(spec['flux']))

        # output:
        self.spectrum = spec      # spectrum dict (must have 'wave' and 'flux')
        self.headers = headers    # all selected headers dict