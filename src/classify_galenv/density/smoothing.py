"""
Gaussian smoothing of 3D density field.
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import numpy as np


def smooth_density(rho, grid_shape, cell_size, smoothing_scale):
    """
    Smooth a 3D density field using a Gaussian filter in Fourier space.

    Parameters
    ----------
    rho : ndarray (Nx, Ny, Nz)
        Input density field (can be rho or delta).
    grid_shape : tuple
        (Nx, Ny, Nz) number of grid cells.
    cell_size : tuple
        (dx, dy, dz) grid spacing in physical units (e.g. Mpc/h).
    smoothing_scale : float
        Gaussian smoothing scale R_s (same units as cell_size).

    Returns
    -------
    rho_smoothed : ndarray
        Smoothed density field.
    """

    Nx, Ny, Nz = grid_shape
    dx, dy, dz = cell_size

    if rho.shape != (Nx, Ny, Nz):
        raise ValueError("Input field shape does not match grid_shape.")

    # ---------------------------------------
    # FFT of density field
    # ---------------------------------------
    rho_k = np.fft.fftn(rho)

    # ---------------------------------------
    # k-vectors
    # ---------------------------------------
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")

    k_squared = kx**2 + ky**2 + kz**2

    # ---------------------------------------
    # Gaussian kernel
    # ---------------------------------------
    kernel = np.exp(-0.5 * k_squared * smoothing_scale**2)

    # ---------------------------------------
    # Apply smoothing
    # ---------------------------------------
    rho_k_smoothed = rho_k * kernel

    # ---------------------------------------
    # Back to real space
    # ---------------------------------------
    rho_smoothed = np.fft.ifftn(rho_k_smoothed).real

    return rho_smoothed