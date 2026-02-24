# src/galenv_classifier_SDSS/pipeline.py
"""
This code performs the tasks in the entire pipeline
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

from pathlib import Path
import tomllib
import time
# Relative imports
from .prep.volume_limited import run_volume_limited
from .prep.outer_cube import run_outer_cube
from .env_classification.find_galaxy_env import run_env_classification

def seconds_to_hms(seconds):                        # Function to convert seconds to H:M:S format for showing runtime
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    ms=int((seconds-int(seconds))*100)
    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}[{ms:02d}]"

def run_pipeline(prep_config_file: str, classification_config_file: str):
    start_time = time.time()
    print("\nLoading configuration...")
    with open(classification_config_file, "rb") as f:
        config = tomllib.load(f)

    print("\nCreating volume-limited catalog...")
    vl_file, survey_info, vls_info = run_volume_limited(prep_config_file)

    print("\nGenerating outer cube...")
    ocube_file, ocube_geom_file = run_outer_cube(prep_config_file, vl_file, survey_info)

    print("Start environmnet classification\n")
    env_gal_file = run_env_classification(config, vl_file, ocube_file, ocube_geom_file)

    print(f"\nClassification finished successfully.")
    print(f"Volume-limited sample [ data ]: {vl_file}")
    print(f"Volume-limited sample [ info ]: {vls_info}")
    print(f"Outer cube file: {ocube_file}")
    print('\nGalaxy with environment classification: '+env_gal_file["galaxy_file"])
    print('Grids with environment classification: '+env_gal_file["grid_file"])
    print("")
    print("====================================================================================")
    print("|                                                                                   |")
    print("|                       ENVIRONMENT  CLASSIFICATION  DONE                           |")
    print("|                                                                                   |")
    print("====================================================================================|")
    end_time = time.time()
    run_time = end_time - start_time
    show_time = seconds_to_hms(run_time)
    print("Runtime : ", show_time)
    print("")

def main():
    root_dir = Path(__file__).parent.parent.parent
    prep_config_file = root_dir / "config/prep_config.toml"
    classification_config_file = root_dir / "config/classification_config.toml"
    run_pipeline(str(prep_config_file), str(classification_config_file))

if __name__ == "__main__":
    main()