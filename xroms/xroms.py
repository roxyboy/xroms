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
    dai = np.empty((nt,nzi,ny,nx))
    dai[:] = np.nan

    for i in range(nx):
        for j in range(ny):
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

    return xr.DataArray(dai, dims=[da.dims[0],'z',da.dims[2],da.dims[3]],
                        coords={da.dims[0]:da.coords[da.dims[0]],
                                'z':zi,da.dims[2]:da.coords[da.dims[2]],
                                da.dims[3]:da.coords[da.dims[3]]})