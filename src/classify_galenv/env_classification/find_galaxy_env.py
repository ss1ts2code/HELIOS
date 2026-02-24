"""
Environment Classification code

Computes:
- Density (CIC)
- Smoothed density
- Gravitational potential
- Deformation tensor
- Environment on grids
- Environment on galaxies via interpolation

Ignores random outer cube points (specObjId = -1) 
to find environment of galaxies in the provided distribution.
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import os
import numpy as np
import pandas as pd
import h5py
from ..density.cic import cic_density
from ..density.smoothing import smooth_density
from ..potential.find_potential import compute_potential
from ..deformation.deformation_tensor import dtensor_fourier
from .env_utils import EnvOnGrids, EnvOnParts

# ================================================================
# Main function
# ================================================================

def run_env_classification(
    config,
    vls_file,
    ocube_file,
    ocube_geom_file,
    output_dir="data/output"
):

    # ------------------------------------------------
    # 1) Load galaxy + outer cube data
    # ------------------------------------------------
    df_vl = pd.read_csv(vls_file, dtype={'specObjId': np.int64})
    df_oc = pd.read_csv(ocube_file, dtype={'specObjId': np.int64})

    coords = df_oc[['e1', 'e2', 'e3']].values.astype(float)
    specIds = df_oc['specObjId'].values.astype(np.int64)

    # ------------------------------------------------
    # 2) Read cube geometry
    # ------------------------------------------------
    with open(ocube_geom_file, 'r') as f:
        xmin, xmax, ymin, ymax, zmin, zmax = map(
            float, f.readline().split(',')
        )

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    Lb = np.array([Lx, Ly, Lz])

    # ------------------------------------------------
    # 3) Grid parameters
    # ------------------------------------------------
    avg_grids = config['classification']['avg_grids']
    lm_th = config['classification']['lambda_th']
    R_smooth = config['smoothing']['R_smooth']
    proc_mode = config['process']['proc_mode']

    gs0 = ((Lx*Ly*Lz) / (avg_grids**3)) ** (1./3.)

    Ng = np.array([
        int(Lx / gs0),
        int(Ly / gs0),
        int(Lz / gs0)
    ])

    ll = Lb / Ng  # cell size

    coords_shifted = coords - np.array([xmin, ymin, zmin])

    # ------------------------------------------------
    # 4) Density (CIC)
    # ------------------------------------------------
    print("Computing CIC density...")
    rho = cic_density(coords_shifted, Ng, ll, proc_mode)

    # ------------------------------------------------
    # 5) Smooth density
    # ------------------------------------------------
    print(f"Smoothing density with R_smooth = {R_smooth} Mpc.")
    rho_sm = smooth_density(rho, Ng, ll, R_smooth)

    # ------------------------------------------------
    # 6) Potential
    # ------------------------------------------------
    phi = compute_potential(rho_sm, Ng, ll)

    # ------------------------------------------------
    # 7) Deformation tensor
    # ------------------------------------------------
    print("Computing deformation tensor...")
    Tij_grid = dtensor_fourier(phi, Ng, ll)

    # ------------------------------------------------
    # 8) Environment on grids
    # ------------------------------------------------
    print("Classifying environment on grids...")
    env_grid = EnvOnGrids(Tij_grid, lm_th)

    # ------------------------------------------------
    # 9) Save grid environment to HDF5
    # ------------------------------------------------
    grid_output_dir = os.path.join(output_dir, "grids")
    os.makedirs(grid_output_dir, exist_ok=True)

    h5_file = os.path.join(grid_output_dir, "env_grid.h5")

    print(f"Saving grid environment to {h5_file}")

    box_size = [Lx,Ly,Lz]
    with h5py.File(h5_file, "w") as hf:
        hf.create_dataset("environment", data=env_grid, compression="gzip")
        hf.create_dataset("Ng", data=Ng)
        hf.create_dataset("cell_size", data=ll)
        hf.create_dataset("box_size", data=box_size)
        hf.attrs["lambda_threshold"] = lm_th

    # ------------------------------------------------
    # 10) Environment on galaxies
    # ------------------------------------------------
    print("Interpolating environment to galaxy positions...")
    env_gals,lamP = EnvOnParts(coords_shifted, Tij_grid, Ng, ll, lm_th)

    # Keep only real galaxies (exclude padding points with specObjId = -1)
    real_mask = specIds != -1

    env_vls = env_gals[real_mask].astype('int')
    dc_vls = np.sum(lamP[real_mask],axis=1)
    specIds_vls = specIds[real_mask]
    del(lamP)
    MM = df_vl.shape[0]

    # ********************** Check population fraction in each environment ********************#

    N_c = len(env_vls[env_vls == 3])
    N_f = len(env_vls[env_vls == 2])
    N_s = len(env_vls[env_vls == 1])
    N_v = len(env_vls[env_vls == 0])
    cf = (N_c / MM)*100
    ff = (N_f / MM)*100
    sf = (N_s / MM)*100
    vf = (N_v / MM)*100

    print('')
    print('Number of galaxies in different environments:')
    print('\t\tCluster:\t %d ( %.2f percent )'%(N_c,cf))
    print('\t\tFilament:\t %d ( %.2f percent )'%(N_f,ff))
    print('\t\tSheet:\t %d ( %.2f percent )'%(N_s,sf))
    print('\t\tVoid:\t %d ( %.2f percent )'%(N_v,vf))
    print('------------------------------------------------')
    print('\t\tTotal:\t\t %d'%(MM)) 
    print('')

    df_oc_env = pd.DataFrame({
        "specObjId": specIds_vls,
        "dc": dc_vls,
        "env": env_vls
    })

    # Merge with VLS catalogue using specObjId
    df_vl_out = df_vl.merge(df_oc_env, on="specObjId", how="left")

    # ------------------------------------------------
    # 11) Save galaxy environments
    # ------------------------------------------------
    gal_output_dir = os.path.join(output_dir,"galaxies")
    os.makedirs(gal_output_dir, exist_ok=True)

    gal_file = os.path.join(gal_output_dir,"galaxy_catalog_wenv.csv")

    df_vl_out.to_csv(gal_file, index=False)

    print(f"Galaxy environment saved: {gal_file}")

    return {
        "grid_file": h5_file,
        "galaxy_file": gal_file
    }