"""
Finds potential on grids for a 3D density field.
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import numpy as np


def compute_potential(smoothed_density, grid_shape, cell_size):
    """
    Solve Poisson equation on a periodic grid.

    Parameters
    ----------
    smoothed_density : Smoothed density contrast field. | (Nx, Ny, Nz)
    grid_shape : tuple containing number of grid cells in each direction.
    cell_size : tuple containing grid spacing in each direction.

    Returns
    -------
    phi : Gravitational potential field.   |  (Nx,Ny,Nz)
    """

    Nx, Ny, Nz = grid_shape
    dx, dy, dz = cell_size

    if smoothed_density.shape != (Nx, Ny, Nz):
        raise ValueError("Input field shape does not match grid_shape.")

    # -------------------------------------------------
    # Fourier transform of density field
    # -------------------------------------------------
    rho_k = np.fft.fftn(smoothed_density)

    # -------------------------------------------------
    # k-vectors
    # -------------------------------------------------
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")

    # -------------------------------------------------
    # Discrete Laplacian operator
    # -------------------------------------------------
    laplacian = -4.0 * (
        (np.sin(0.5 * kx * dx) / dx) ** 2
        + (np.sin(0.5 * ky * dy) / dy) ** 2
        + (np.sin(0.5 * kz * dz) / dz) ** 2
    )

    # -------------------------------------------------
    # Invert Laplacian (handle k=0 mode)
    # -------------------------------------------------
    inv_laplacian = np.zeros_like(laplacian)

    mask = laplacian != 0.0
    inv_laplacian[mask] = 1.0 / laplacian[mask]

    # Set zero mode to zero (potential defined up to constant)
    inv_laplacian[~mask] = 0.0

    # -------------------------------------------------
    # Solve in Fourier space
    # -------------------------------------------------
    phi_k = rho_k * inv_laplacian

    # -------------------------------------------------
    # Back to real space
    # -------------------------------------------------
    phi = np.fft.ifftn(phi_k).real

    return phi