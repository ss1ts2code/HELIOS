"""
Utility functions for Environment classification

Contains:
- Trilinear interpolation
- Grid-based environment classification
- Particle-based interpolation + classification
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""

import numpy as np

# ================================================================
# Trilinear Interpolation
# ================================================================

def TL_interp(fcoor, cv):
    """
    Trilinear interpolation.

    Parameters
    ----------
    fcoor : Fractional coordinates inside a unit cube.
    cv : Values at the cube vertices.

    Returns
    -------
    float
        Interpolated value.
    """
    xd, yd, zd = fcoor
    c0 = ((cv[0][0][0] * (1 - xd) + cv[1][0][0] * xd) * (1 - yd) + (cv[0][1][0] * (1 - xd) + cv[1][1][0] * xd) * yd)
    c1 = ((cv[0][0][1] * (1 - xd) + cv[1][0][1] * xd) * (1 - yd) + (cv[0][1][1] * (1 - xd) + cv[1][1][1] * xd) * yd)
    c = c0 * (1 - zd) + c1 * zd
    return c


# ================================================================
# Environment classification from eigenvalues
# ================================================================

def get_envo(lamP, lm_th):
    """
    Classify environment from eigenvalues.

    Returns
    -------
    int
        0: void
        1: sheet
        2: filament
        3: knot
    """
    lm1, lm2, lm3 = lamP
    if lm1 > lm_th and lm2 > lm_th and lm3 > lm_th:
        return 3
    elif lm1 > lm_th and lm2 > lm_th and lm3 < lm_th:
        return 2
    elif lm1 > lm_th and lm2 < lm_th and lm3 < lm_th:
        return 1
    elif lm1 < lm_th and lm2 < lm_th and lm3 < lm_th:
        return 0
    else:
        return -1


# ================================================================
# Environment on Grids
# ================================================================

def EnvOnGrids(Tij, lm_th):
    """
    Classify environment on grid points from deformation tensor.

    Parameters
    ----------
    Tij : Deformation tensor on grids.  (Nx, Ny, Nz, 3, 3)
    lm_th : Threshold eigenvalue.

    Returns
    -------
    env_g : Environment classification on grids. (Nx, Ny, Nz)
    """
    lamG = np.linalg.eigvals(Tij).real
    lamG = np.sort(lamG,axis=-1)[..., ::-1]
    env_g=np.zeros((lamG.shape[:3]),dtype='int')
    lm1,lm2,lm3=lamG.transpose(3,0,1,2)
    mask1=((lm1 > lm_th ) & ( lm2 > lm_th ) & ( lm3 > lm_th))
    mask2=((lm1 > lm_th ) & ( lm2 > lm_th ) & ( lm3 < lm_th))
    mask3=((lm1 > lm_th ) & ( lm2 < lm_th ) & ( lm3 < lm_th))
    mask4=((lm1 < lm_th ) & ( lm2 < lm_th ) & ( lm3 < lm_th))
    env_g[mask1]=3
    env_g[mask2]=2
    env_g[mask3]=1
    env_g[mask4]=0
    return(env_g)

# ================================================================
# Environment on Particles
# ================================================================

def EnvOnParts(coors, Tij_g, Ng, ll, lm_th):
    """
    Interpolates deformation tensor to particle positions
    and classifies their environment.

    Parameters
    ----------
    coors : Particle positions (already shifted to cube origin).
    Tij_g : Deformation tensor on grids. (Nx, Ny, Nz, 3, 3)

    Returns
    -------
    env : Environment classification of particles.
    lamP : Eigenvalues at particle positions.
    """
    Npart=len(coors)                                
    pos_part=coors.copy()
    x,y,z=pos_part.T
    Nx,Ny,Nz=Ng
    lx,ly,lz=ll 
    x /= lx;    y /= ly;    z /= lz
    env = np.zeros((Npart),dtype='int')
    Tij_verts=np.zeros((2,2,2,3,3),dtype='float')
    Tij_p=np.zeros((Npart,3,3),dtype='float')
    lamP=np.zeros((Npart,3),dtype='float')
    for p in range(Npart):
        ix = int(x[p]);   iy = int(y[p]);    iz = int(z[p])
        rel_pos=np.array([x[p]-ix,y[p]-iy,z[p]-iz])
        for i,j,k in ((i,j,k) for i in range(2) for j in range(2) for k in range(2)):
            ix2=(ix+i)%Nx
            iy2=(iy+j)%Ny    
            iz2=(iz+k)%Nz
            Tij_verts[i][j][k]=Tij_g[ix2][iy2][iz2]
        Tij_vt=Tij_verts.transpose(3,4,0,1,2)
        for ii,jj in ((ii,jj) for ii in range(3) for jj in range(3)):
            Tij_p[p][ii][jj] = TL_interp(rel_pos,Tij_vt[ii][jj])
        lamP[p] = np.linalg.eigvals(Tij_p[p]).real
        lamP[p]=np.sort(lamP[p])[::-1]
        env[p]=get_envo(lamP[p],lm_th)
    return(env,lamP)