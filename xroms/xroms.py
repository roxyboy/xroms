import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import scipy.interpolate as naiso
import dask.array as dsar

__all__ = ["sig2z"]

def sig2z(da, zr, zi, nvar='u'):
    """
    Interpolate variables on \sigma coordinates onto z coordinates.

    Parameters
    ----------
    da : `xarray.DataArray`
        The data on sigma coordinates to be interpolated
    zr : `numpy.array`
        The depths corresponding to sigma layers
    zi : `numpy.array`
        The depths which to interpolate the data on
    nvar : str (optional)
        Name of the variable

    Returns
    -------
    dai : `xarray.DataArray`
        The data interpolated onto a spatial uniform z coordinate
    """
    nzi = len(zi)
    N = da.shape
    if len(N) == 4:
        dai = np.empty((N[0],nzi,N[2],N[3]))
        dim = [da.dims[0],'z',da.dims[2],da.dims[3]]
        coord = {da.dims[0]:da.coords[da.dims[0]],
                'z':zi, da.dims[2]:da.coords[da.dims[2]],
                da.dims[3]:da.coords[da.dims[3]]
                }
    elif len(N) == 3:
        dai = np.empty((nzi,N[1],N[2]))
        dim = ['z',da.dims[2],da.dims[3]]
        coord = {'z':zi,da.dims[2]:da.coords[da.dims[2]],
                da.dims[3]:da.coords[da.dims[3]]
                }
    else:
        raise ValueError("The data should at least have three dimensions")
    dai[:] = np.nan

    for i in range(N[-1]):
        for j in range(N[-2]):
            if nvar=='u':  # u variables
                zl = np.squeeze(.5*(zr[:,j+1,i+1]+zr[:,j+1,i]))
            elif nvar=='v': # v variables
                zl = np.squeeze(.5*(zr[:,j,i+1]+zr[j+1,i+1,:]))
            else:
                zl = np.squeeze(zr[:,j,i])

            if zl.min() < -5e2: # only bother for sufficiently deep regions
                ind = np.argwhere(zi >= zl.min()) # only interp on z above topo
                for s in range(nt):
                    dal = np.squeeze(da[s,j,i,:])
                    f = naiso.interp1d(zl, dal, fill_values='extrapolate')
                    dai[s,0:length(ind),j,i] = f(zi[int(ind[0]):])

    return xr.DataArray(dai, dims=dim, coords=coord)
