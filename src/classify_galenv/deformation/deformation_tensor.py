"""
Deformation tensor computed in Fourier space.
Using finite difference operator
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import numpy as np

def dtensor_fourier(phi, grid_shape, cell_size):
    """
    Compute deformation tensor using Fourier-space differentiation.

    Parameters
    ----------
    phi : ndarray (Nx, Ny, Nz)
        Gravitational potential field.
    grid_shape : tuple
        (Nx, Ny, Nz)
    cell_size : tuple
        (dx, dy, dz)

    Returns
    -------
    Tij : Deformation tensor field. |  (Nx, Ny, Nz, 3, 3)
    """

    Nx, Ny, Nz = grid_shape
    dx, dy, dz = cell_size

    if phi.shape != (Nx, Ny, Nz):
        raise ValueError("phi shape does not match grid_shape.")

    # Fourier transform
    phi_k = np.fft.fftn(phi)

    # k vectors
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)

    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")

    # Discrete derivative operators
    deriv = np.zeros((3, Nx, Ny, Nz), dtype=float)

    deriv[0] = np.sin(kx * dx) / dx
    deriv[1] = np.sin(ky * dy) / dy
    deriv[2] = np.sin(kz * dz) / dz

    Tij = np.zeros((3, 3, Nx, Ny, Nz), dtype=float)

    for i in range(3):
        for j in range(3):
            T_k = -phi_k * deriv[i] * deriv[j]
            Tij[i, j] = np.fft.ifftn(T_k).real

    return Tij.transpose(2, 3, 4, 0, 1)