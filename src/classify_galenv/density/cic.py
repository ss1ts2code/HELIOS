"""
Finds density in grids using cloud in cell method
-----------------------------------------------------------------------------------------
HELIOS : Hessian-based Environment Identifier for Large-scale Observational Survey-data
-----------------------------------------------------------------------------------------
Author: Suman Sarkar
"""
import numpy as np
import multiprocessing as mp

# ============================================================
# 1) SINGLE PROCESS ONE-CHUNK (SIMPLE LOOP VERSION)
# ============================================================
def cic_serial_single_simple(coors, Ng, ll):
    Nx,Ny,Nz = Ng
    lx,ly,lz = ll
    Npart = len(coors)
    DC = np.zeros((Nx, Ny, Nz), dtype='float32')
    pos_part = coors.copy()
    x, y, z = pos_part.T
    x /= lx; y /= ly; z /= lz
    rho_b_inv = (Nx * Ny * Nz) / Npart
    for p in range(Npart):
        ix, iy, iz = int(x[p]), int(y[p]), int(z[p])
        wx = 1. - (x[p] - ix)
        wy = 1. - (y[p] - iy)
        wz = 1. - (z[p] - iz)
        for i, j, k in ((i,j,k) for i in range(2) for j in range(2) for k in range(2)):
            W = abs(i - wx) * abs(j - wy) * abs(k - wz)
            ix2 = (ix + i) % Nx
            iy2 = (iy + j) % Ny
            iz2 = (iz + k) % Nz
            DC[ix2][iy2][iz2] += (W * rho_b_inv)
    return DC

# ============================================================
# 2) SINGLE PROCESS  ONE-CHUNK (VECTORIZED VERSION)
# ============================================================
def cic_serial_single_vectorized(coors, Ng, ll):
    Nx, Ny, Nz = Ng
    Npart = len(coors)
    pos_part = coors.copy()
    part_g = pos_part.T / ll[:, None]
    ix = np.floor(part_g).astype('int')
    dd = part_g - ix
    ww = 1 - dd
    rho_b_inv = (Nx * Ny * Nz) / Npart
    DC = np.zeros(Ng, dtype='float32')
    for i, j, k in ((i, j, k) for i in range(2) for j in range(2) for k in range(2)):
        ii = np.array([i, j, k])
        W = np.prod(np.abs(ww - ii[:, None]), axis=0)
        ix2 = (ix + ii[:, None]) % Ng[:, None]
        np.add.at(DC, (ix2[0], ix2[1], ix2[2]), W * rho_b_inv)
    del(ix,dd,ww,Npart,pos_part,part_g)
    return DC

# ============================================================
# 3) SINGLE PROCESS MULTI-CHUNK (VECTORIZED VERSION)
# ============================================================
def cic_chunk(coors_chunk, Ng, ll, rho_b_inv):
    Npart_chunk = len(coors_chunk)
    pos_part = coors_chunk.copy()
    part_g = pos_part.T / ll[:, None]
    ix = np.floor(part_g).astype('int')
    dd = part_g - ix
    ww = 1 - dd
    DC_c = np.zeros(Ng, dtype='float32')
    for i,j,k in ((i,j,k) for i in range(2) for j in range(2) for k in range(2)):
        ii = np.array([i,j,k])
        W = np.prod(np.abs(ww - ii[:, None]), axis=0)
        ix2 = (ix + ii[:, None]) % Ng[:, None]
        np.add.at(DC_c, (ix2[0], ix2[1], ix2[2]), W * rho_b_inv)
    del(ix,dd,ww,Npart_chunk,pos_part,part_g)
    return DC_c

def cic_serial_multi_simple(coors, Ng, ll):
    chunk_size = 100_000
    Npart = len(coors)
    num_chunks = (Npart + chunk_size - 1) // chunk_size
    rho_b_inv = (Ng[0]*Ng[1]*Ng[2])/Npart
    DC_all = np.zeros(Ng, dtype='float32')
    for i in range(num_chunks):
        start_idx = i*chunk_size
        end_idx = min(start_idx+chunk_size, Npart)
        coors_chunk = coors[start_idx:end_idx]
        DC_all += cic_chunk(coors_chunk, Ng, ll, rho_b_inv)
    return DC_all

# ============================================================
# 4) PARALLEL PROCESS MULTI-CHUNK (VECTORIZED VERSION)
# ============================================================
def cic_chunk_worker(args):
    coors_chunk, Ng, ll, rho_b_inv = args
    Nx, Ny, Nz = Ng
    part_g = coors_chunk.T / ll[:, None]
    ix = np.floor(part_g).astype(np.int32)
    dd = part_g - ix
    ww = 1.0 - dd
    DC_chunk = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                ii = np.array([i, j, k], dtype=np.int32)
                W = np.prod(np.abs(ww - ii[:, None]), axis=0)
                ix2 = (ix + ii[:, None]) % np.array(Ng)[:, None]
                np.add.at(DC_chunk,(ix2[0], ix2[1], ix2[2]), W * rho_b_inv)
    return DC_chunk

def cic_parallel_multi_vectorized(coors, Ng, ll, num_proc):
    Nx, Ny, Nz = Ng
    Npart = len(coors)
    rho_b_inv = (Nx * Ny * Nz) / Npart
    chunk_size = (Npart + num_proc - 1) // num_proc
    chunks = [
        (coors[i:i + chunk_size], Ng, ll, rho_b_inv)
        for i in range(0, Npart, chunk_size)
    ]
    with mp.Pool(processes=num_proc) as pool:
        results = pool.map(cic_chunk_worker, chunks)
    DC_total = np.sum(results, axis=0)
    return DC_total

# ============================================================
# 5) NON-INTERACTIVE WRAPPER
# ============================================================

def cic_density(coors, Ng, ll, proc_mode):
    """
    Automatic CIC density computation.
    """

    mode_names={"mode_1" : "Serial_one_chunk_simple", 
     "mode_2" : "Serial_one_chunk_vectorized",
     "mode_3" : "Serial_multi_chunk_vectorized",
     "mode_4" : "Parallel_multi_chunk_vectorized"}

    print("Processing mode :  "+mode_names[proc_mode])
    if proc_mode=="mode_1":
        return cic_serial_single_simple(coors, Ng, ll)
    elif proc_mode =="mode_2":
        return cic_serial_single_vectorized(coors, Ng, ll)
    elif proc_mode =="mode_3":
        return cic_serial_multi_simple(coors, Ng, ll)
    elif proc_mode =="mode_4":
        num_proc = max(mp.cpu_count() - 1, 1)
        return cic_parallel_multi_vectorized(coors, Ng, ll, num_proc)
    else:
        raise ValueError("Invalid mode. Choose from { 'mode_1', 'mode_2', 'mode_3', 'mode_4'}")