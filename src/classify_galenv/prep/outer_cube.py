# src/galenv_classifier_SDSS/prep/outer_cube.py
"""
This code prepares a bounding box around the galaxy_catalog 
filled with random points of same number density
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import os
import tomllib
import numpy as np
import pandas as pd

def run_outer_cube(config_path, vl_file, survey_info_file):
    """Generate outer cube galaxies outside survey volume"""

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    outer_cube_output = config["paths"]["cube_catalog"]
    outer_cube_geom_output = config["paths"]["geometry_file"]

    # Read survey info
    with open(survey_info_file, "r") as f:
        _, xc, _, ra11, ra22, dec11, dec22, _, nden = map(float, f.readline().strip().split(","))
    df = pd.read_csv(vl_file)
    xmin, xmax = df['e1'].min(), df['e1'].max()
    ymin, ymax = df['e2'].min(), df['e2'].max()
    zmin, zmax = df['e3'].min(), df['e3'].max()
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    Tot_vol = Lx * Ly * Lz
    N_ran = int((Tot_vol - ((ra22 - ra11)*(np.sin(dec22) - np.sin(dec11))*xc**3)/3.0) * nden)

    print(f"Generating {N_ran} mock galaxies to fill the outer cube ...")

    outer_gals = []
    count = 0
    while count < N_ran:
        x = xmin + np.random.rand() * Lx
        y = ymin + np.random.rand() * Ly
        z = zmin + np.random.rand() * Lz
        rr = np.sqrt(x**2 + y**2 + z**2)
        ra = np.arctan2(y, x)
        dec = np.arctan2(z, np.sqrt(x**2 + y**2))
        if rr < xc and ra11 < ra < ra22 and dec11 < dec < dec22:
            continue
        outer_gals.append(["-1", x, y, z])
        count += 1

    outer_gals = np.array(outer_gals, dtype=object)
    all_gals = np.vstack([df[['specObjId', 'e1', 'e2', 'e3']].values, outer_gals])
    df2 = pd.DataFrame(all_gals, columns=['specObjId', 'e1', 'e2', 'e3'])
    df2['e1'] -= xmin
    df2['e2'] -= ymin
    df2['e3'] -= zmin

    os.makedirs(os.path.dirname(outer_cube_output), exist_ok=True)
    df2.to_csv(outer_cube_output, index=False)
    os.makedirs(os.path.dirname(outer_cube_geom_output), exist_ok=True)
    with open(outer_cube_geom_output, "w") as f:
        f.write(f"{xmin},{xmax},{ymin},{ymax},{zmin},{zmax}")

    print(f"Outer cube saved: {outer_cube_output}")
    return outer_cube_output, outer_cube_geom_output