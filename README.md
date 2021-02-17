![image](https://github.com/patternizer/pha-backend/blob/master/PLOTS/adjustments-absolute/trausti-raw-normals/yearly/raw-v-pha-means.png)

# pha-backend

Python back-end for data handling, analysis and plotting of output from NOAA's PHA v52i algorithm for pairwise homogenisation of land surface air temperature monitoring station subsets from CRUTEM5 in the [GloSAT](https://www.glosat.org) project

* python output file data handling code

## Contents

* `pha-backend.py` - python code to generate station homogenisation dataframes

Plus plots of invidual raw versus PHA adjusted absolute temperatures and anomalies (from 1961-1990), subset means, differences and a histogram of adjustments.

The first step is to clone the latest pha-backend code and step into the check out directory: 

    $ git clone https://github.com/patternizer/pha-backend.git
    $ cd pha-backend

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested in a conda virtual environment running a 64-bit version of Python 3.8+.

pha-backend scripts can be run from sources directly, once the PHA v52i output files are generated and paths set.

Run with:

    $ python pha-backend.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

