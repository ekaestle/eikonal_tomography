# Python script - Eikonal tomography

v. 2021.03.20
for the latest version, please visit https://github.com/ekaestle

Author: Emanuel D. Kästle
emanuel.kaestle@fu-berlin.de

###########################
# REFERENCES #
###########################

The following article should be cited by anyone who uses this code:

Kästle, E. D., Molinari, I., Boschi, L., Kissling, E., & AlpArray Working Group. (2022).
Azimuthal anisotropy from Eikonal Tomography: example from ambient-noise measurements 
in the AlpArray network. Geophysical Journal International, 229(1), 151-170.


###########################
# INSTRUCTIONS #
###########################

The Python scripts can be used to apply isotropic and azimuthally anisotropic Eikonal
tomography after Lin et al. (2009). Detailed instructions on the principles of the
Eikonal method and how this code works are given in the article referenced above.

# Prerequisites
Python v.3.6 or above with the following packages:
numpy, mpi4py, scipy, matplotlib, basemap, pyproj, cartopy
optional: scikit-fmm, pillow (to use the fast-marching method and create synthetic data)
recommended: spyder or another python editor

It is recommended to install Python and all required packages via Anaconda (executable
installers and instructions for Linux/Windows/Mac can be found on the Anaconda webpage).

Once Anaconda is installed, create a new environment in a console window, for example:
$ conda create -n py39 python=3.9   # or other supported Python version
$ conda activate py39 # activate the newly created environment

Then, install all the required packages
$ conda install numpy scipy matplotlib basemap pyproj cartopy mpi4py

The optional packages can be installed the same way
$ conda install scikit-fmm pillow spyder=4

You should now be able to run the first example
$ cd {path_to_this_folder}/synthetic_example_simple/
$ python eikonal_tomography.py

# Running the synthetic examples

The input should be given as travel times between station pairs. An example input
file is given in the synthetic_example_simple folder. The example dataset can be
re-created with the provided script

$ python create_synthetic_data_simple.py

The creation of synthetic data requires that the FMM.py script is in the same folder.

The second synthetic example provided in the synthetic_example_publication/ folder re-
creates the data presented in the above referenced article. Execution of this script will
take very long. Finally, example_dataset_rayleigh.txt is created. The Eikonal tomography
is applied by running

$ python eikonal_tomography.py

Inside the eikonal_tomography.py script, there is a header section with user-defined
parameters. The function of these parameters is explained in the script itself and in
the above-referenced article.

# Running the Eikonal tomography script with an own dataset

Prepare one or several input files according to the scheme explained at the top of the
eikonal_tomography.py script (compare example datasets in the synthetic_example_.../ folders).

Copy the eikonal_tomography.py script to a new folder. Adapt all the parameters in the 
USER-DEFINED PARAMETERS section of the script to your needs.

Run
$ python eikonal_tomography.py

The results will appear in a newly created subfolder.


In case of questions/problems, feel free to contact emanuel.kaestle@fu-berlin.de


# References
Lin, F.-C., Ritzwoller, M. H., and Snieder, R. (2009). Eikonal
tomography: surface wave tomography by phase front tracking
across a regional broad-band seismic array. Geophysical Journal
International, 177(3):1091–1110.

