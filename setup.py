from setuptools import setup, find_packages

setup(name = 'ACTINometer',
    version = "0.1.0",
    description = 'ACTINometer - a pipeline for the derivation of spectral activity indices for stars with exoplanets',
    url = 'https://github.com/telmonteiro/PEEC-24/',
    author = 'Telmo Monteiro',
    author_email = 'up202308183@up.pt',
    keywords = ['astronomy', 'activity', 'fits', 'harps', 'espresso', 'uves', 'radial velocity', 'exoplanets'],
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv'],},
    install_requires = ['numpy', 'pandas', 'astropy', 'matplotlib', 'scipy', 'astroquery', 'math', 'PyAstronomy',]
)