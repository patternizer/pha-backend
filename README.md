![image](https://github.com/patternizer/pha-backend/blob/master/australia-adjusted-raw-v-pha-temporal-coverage.png)
![image](https://github.com/patternizer/pha-backend/blob/master/australia-adjusted-adjustment-histogram.png)

# pha-backend

Python back-end for data handling, analysis and plotting of output from NOAA's PHA v52i algorithm for pairwise homogenisation algorithm (PHA) adjustment of land surface air temperature monitoring station observations. Ongoing work for the [GloSAT](https://www.glosat.org) project

* python output file data handling code

## Contents

* `pha-backend.py` - python code to generate station homogenisation timeseries dataframes for raw and adjusted station observations supplied in GHCNm-v4 format.

In addition, the code plots the archive temporal coverage ranked by station, mean raw versus PHA-adjusted absolute temperature timeseries and their differences as well as a histogram of adjustments.

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


