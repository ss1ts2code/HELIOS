# src/galenv_classifier_SDSS/prep/volume_limited.py
"""
This code prepares a volume limited sample using the raw survey data
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""
import os
import tomllib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

clight = 2.997e5  # km/s
d2r = np.pi / 180.0

def get_cdv(dist):
    return float(str(dist).split(' ')[0])

def cdist(z, cosmo):
    if isinstance(z, float):
        return get_cdv(cosmo.comoving_distance(z))
    z = np.array(z)
    if len(z) < 100:
        return np.array([get_cdv(cosmo.comoving_distance(zi)) for zi in z])
    nint = 1000
    zmn, zmx = np.min(z), np.max(z)
    eps = (zmx - zmn) / nint
    z4int = np.linspace(zmn - eps, zmx + eps, nint)
    x4int = np.array([get_cdv(cosmo.comoving_distance(zi)) for zi in z4int])
    return np.interp(z, z4int, x4int)

def run_volume_limited(config_path="config/prep_config.toml"):
    """Create a volume-limited catalog and survey info."""

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    raw_path = config["paths"]["gal_file"]
    output_file = config["paths"]["volume_output"]
    survey_info_file = config["paths"]["survey_info_output"]
    vls_info_file = config["paths"]["vls_info_output"]

    # Cosmology
    H0 = config["cosmology"]["H0"]
    Om0 = config["cosmology"]["Om0"]
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    # Volume-limited parameters
    mc = config["volume_limited"]["m_c"]
    Mc = config["volume_limited"]["M_c"]
    ran_frac = config["volume_limited"]["ran_frac"]
    seed_vl = config["volume_limited"]["seed"]

    # Survey geometry
    ra1 = config["survey_geometry"]["ra_min"]
    ra2 = config["survey_geometry"]["ra_max"]
    dec1 = config["survey_geometry"]["dec_min"]
    dec2 = config["survey_geometry"]["dec_max"]

    data = pd.read_csv(raw_path)

    # retaining galaxies within the applied geometry and magnitude cut
    mask=((data['z'] > 0.) 
        & (data['ra'] >= ra1) 
        & (data['ra'] <= ra2) 
        & (data['dec'] >= dec1) 
        & (data['dec'] <= dec2) 
        & (data['petro_mag'] <= mc))

    dfs=data[mask].reset_index(drop=True)

    # for aligning the distribution symmetrically about +ve x axis 
    # (and converting degree to radians)
    dfs['ra'] = (dfs['ra'].values - (ra1 + ra2) * 0.5) * d2r
    dfs['dec'] = dfs['dec'].values*d2r
    x = cdist(dfs['z'].to_numpy(), cosmo)
    dl = x * (1 + dfs['z'].values)
    aM = np.array(dfs['petro_mag'].values-dfs['e_corr'].values-dfs['k_corr'].values) - 5*(np.log10(dl) + 5)
    dm=np.array(dfs['petro_mag']-dfs['e_corr'])-aM

    dfs['x']=x
    dfs['aM']=aM
    dfs['dm']=dm

    #random sampling of galaxies for interpolation to determine redshift vs distance-modulas relation
    gc = dfs.shape[0]
    np.random.seed(seed_vl)
    nrand = int(gc * ran_frac) 
    random_indices = np.random.choice(range(gc), nrand, replace=False)
    random_data = dfs.iloc[random_indices]
    rdm = random_data["dm"].values
    rz = random_data["z"].values

    # removing degenerecies in arrays of redshift and distace modulas
    ui = np.unique(rz, return_index=True)[1]
    rz = rz[ui]
    rdm = rdm[ui]
    ui2 = np.unique(rdm, return_index=True)[1]
    rz = rz[ui2]
    rdm = rdm[ui2]

    # sorting the arrays for interpolation (in ascending order of distance modulas )
    sorted_indices = np.argsort(rdm)
    rdm = rdm[sorted_indices]
    rz = rz[sorted_indices]
    dmc=mc-Mc

    # defining function for interpolation
    int_func = interp1d(rdm, rz, kind='linear')
    zc = float(int_func(dmc))

    # sorting galaxies in the volume limited sample
    mask2=((dfs['z'] <= zc) & (dfs['aM'] <= Mc))
    dfs2=dfs[mask2].reset_index(drop=True)

    # finding comoving cartezian coordinates of the galaxies in volume limited sample
    dfs2['e1'] = dfs2['x'].values * np.cos(dfs2['dec'].values) * np.cos(dfs2['ra'].values)
    dfs2['e2'] = dfs2['x'].values * np.cos(dfs2['dec'].values) * np.sin(dfs2['ra'].values)
    dfs2['e3'] = dfs2['x'].values * np.sin(dfs2['dec'].values)

    # setting new column order for output data
    new_col_order=['specObjId','e1', 'e2', 'e3', 'z', 'petro_mag', 'aM']
    df_final= dfs2[new_col_order]           # reorder the columns in the dataframes 

    # Writting data into output datafile
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)

    # Calculation of linear extent, volume, number density, etc.
    ra11 = (ra1 - ra2) * 0.5 *d2r
    ra22 = (ra2 - ra1) * 0.5 *d2r
    dec11 = dec1 * d2r
    dec22 = dec2 * d2r

    xc = cdist(zc, cosmo)
    tvol = ((ra22-ra11)*(np.sin(dec22)-np.sin(dec11))*xc**3)/3.0

    neff = dfs2.shape[0]
    nden = neff / tvol
    igs = nden ** (-1.0 / 3.0)

    # writting on-screen Outputs
    print('')
    print("Details of the prepared volume-limited-sample:")
    print("\tAbsolute magnitude limit used : M <= {:.2f}".format(Mc))
    print("\tRedshift limit obtained : z <= {:.4f}".format(zc))
    print("\tGalaxies within the magnitude limit: {:d}".format(neff))
    print("\tLinear comoving extent: {:.2f} MPc".format(xc))
    print("\tVolume of the region: {:e} Cubic MPc".format(tvol))
    print("\tMean number density = {:e} per CMPc".format(nden))
    print("\tMean intergalactic separation = {:.2f} MPc".format(igs))
    print('')

    os.makedirs(os.path.dirname(survey_info_file), exist_ok=True)
    with open(survey_info_file, "w") as f:
        f.write("{},{},{},{},{},{},{},{},{}".format(zc, xc, Mc, ra11, ra22, dec11, dec22, neff, nden))

    os.makedirs(os.path.dirname(vls_info_file), exist_ok=True)
    with open(vls_info_file, "w") as f2:
        f2.write("Absolute magnitude limit used : M <= {:.2f}\n".format(Mc))
        f2.write("Redshift limit obtained : z <= {:.4f}\n".format(zc))
        f2.write("RA span : {:.2f} <= RA <= {:.2f}\n".format(ra1, ra2))
        f2.write("Dec span : {:.2f} <= Dec <= {:.2f}\n".format(dec1, dec2))
        f2.write("Galaxies within the magnitude limit: {:d}\n".format(neff))
        f2.write("Linear comoving extent: {:.2f} MPc\n".format(xc))
        f2.write("Volume of the region: {:e} Cubic MPc\n".format(tvol))
        f2.write("Mean number density = {:e} per CMPc\n".format(nden))
        f2.write("Mean intergalactic separation = {:.2f} MPc\n".format(igs))

    print(f"Volume-limited catalog saved: {output_file}")
    return output_file, survey_info_file, vls_info_file