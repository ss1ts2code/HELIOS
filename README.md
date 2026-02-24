=====================================================================================
# HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
=====================================================================================


# DESCRIPTION ---------------------
HELIOS is a Parallelized and optimized Python package for classifying large-scale galaxy environments using spectroscopic survey data. The package provides tools for preparing a volume limited sample, follwed by construction of smooth density fields and finding the Hessian of the gravitational potential. Using the eigen valuies of the detormation tensor it provides a framework for environment classifiaction.


# FEATURES -----------------------
Galaxy-level environment classification

Does not require a cube or parallelopiped

Designed for spectroscopic redshift surveys

Modular architecture for pipeline integration

Scalable to large galaxy catalogs

Suitable for research-grade cosmological studies



# SCIENTIFIC GOALS ---------------

Cosmic web classification 
Large-scale structure formation 
Environmental dependence of galaxy properties 
Modelling Redshift distortions

HELIOS is intended to support research in cosmology and extragalactic astrophysics by providing reliable and extensible tools for environment characterization.




# GENERAL REQUIRMENTS ------------
• Python 3.9 or newer
• pip (Python package installer)
• git (if installing from repository)


Creating a virtual environment would be preferable: 
python3 -m venv henv

Activation of the environment: 
          
          Linux/ macOS >>  source henv/bin/activate
          
          Windows >> henv\Scripts\activate ( command propt )      |                 henv\Scripts\Activate.ps1 (powershell)
                     


# INSTALLATION -----------------

Clone the repository:

git clone https://github.com/ss1ts2code/HELIOS.git

cd HELIOS/ 
pip install -e .


# RUN --------------------------
                     
Now run the executable file 

galenv-find 
        
If you get a wanrning like this in windows systems :

**  WARNING: The script galenv-find.exe is installed in 'C:\Users\your-username\AppData\Roaming\Python\Python312\Scripts' 
    which is not on PATH. Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.**
    
then run :
 C:\Users\your-username\AppData\Roaming\Python\Python312\Scripts\galenv-find.exe
 
in the powershell.


# INPUT DATA -----------------

Input data should be kept in 'data/raw/galaxy_catalog.csv' before running the code.

The package expects a galaxy catalog containing at minimum the columns:
---------------------------------------------------------------
1. specObjId ( Unique galaxy ID )
2. 'z' ( Redshift )
3. 'ra' ( Right Ascension in degrees )
4. 'dec' ( Declination in degrees )
5. 'petro_mag' ( Petrosian apparent magnitude )
6. 'e_corr' ( extinction/ reddning correction )
7. 'k_corr' ( K-correction )
---------------------------------------------------------------

Additional properties such as stellar masses, SFR, concentration Index,  may also be included but are not strictly required.

All parameters required to run the analysis are assigned in the config files.
           'config/prep_config.toml'   : Stores reqiured parameters for volume-limited sample preparation
           'config/classification_config.toml'   : Store reqiured parameters for environment classification



sample SQL ---------------------
Data can be obtained from the SDSS SciServer using SQL queries through the CasJobs interface. Given below is 
a minimal query to get required information of the galaxies/ Select the context (Data Release) before running the SQL.


SELECT 
 s.specObjId,
 s.z,
 s.ra,
 s.dec,
 p.petroMag_r as petro_mag,
 p.extinction_r as e_corr, 
 p2.kcorrR as K_corr
INTO mydb.galaxy_catalog
FROM specObj as s,photoObj as p, photoZ as p2
where 
 s.specObjId= p.specObjId and
 p.ObjId= p2.ObjId and
 s.class='GALAXY' and
 s.zWarning =0 and
 s.sciencePrimary =1 and
 s.z < 0.3


# OUTPUT FILES -------------
1. 'data/volume_limited/volume_limited.csv' : contains the selected galaxies in the volume limited sample ( same columns as the input data ).
2. 'data/volume_limited/gal_inf.csv' : contains the information regarding the geometry of the selected region, galaxy count and number density 
                                       in the the volume limited sample.
3. 'data/volume_limited/gal_info.txt' : Holds the information regarding the specifications of the volume-limited sample.

4. 'data/output/galaxies/galaxy_catalog_wenv.csv' : Holds the information regarding the cic-density and environmental classification for each galaxies in the volume limited sample.

5. 'data/output/grids/env_grids.hdf5' : grid wise environmental classification.


# FUTURE DEVELOPMENTS TO BE DONE ------------------------

Modelling Redshift space distortion

Machine learning–based environment tagging

Improved visualization tools

Toogle between Simulation and Observation



# AUTHOR -------------
Suman Sarkar : https://sites.google.com/view/suman-sarkar/home