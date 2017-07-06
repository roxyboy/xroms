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
    zr : `xarray.DataArray`
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
    if np.diff(zi)[0] < 0. or zi.max() < 0.:
        raise ValueError("The values in `zi` should be postive and increasing.")
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
        dim = ['z',da.dims[1],da.dims[2]]
        coord = {'z':zi, da.dims[1]:da.coords[da.dims[1]],
                da.dims[2]:da.coords[da.dims[2]]
                }
    else:
        raise ValueError("The data should at least have three dimensions")
    dai[:] = np.nan
    zi = -zi[::-1] # ROMS has deepest level at index=1

    for i in range(N[-1]):
        for j in range(N[-2]):
            if nvar=='u':  # u variables
                zl = np.squeeze(.5*(zr.roll(eta_rho=-1,xi_rho=-1)[:,j,i].values
                                    +zr.roll(eta_rho=-1)[:,j,i].values)
                                )
            elif nvar=='v': # v variables
                zl = np.squeeze(.5*(zr.roll(xi_rho=-1)[:,j,i].values
                                    +zr.roll(eta_rho=-1,
                                            xi_rho=-1)[:,j,i].values)
                                )
            else:
                zl = np.squeeze(zr[:,j,i].values)

            if zl.min() < -5e2: # only bother for sufficiently deep regions
                ind = np.argwhere(zi >= zl.min()) # only interp on z above topo
                if len(N) == 4:
                    for s in range(N[0]):
                        dal = np.squeeze(da[s,:,j,i])
                        f = naiso.interp1d(zl, dal, fill_values='extrapolate')
                        dai[s,:length(ind),j,i] = f(zi[int(ind[0]):])[::-1]
                else:
                    dal = np.squeeze(da[:,j,i])
                    f = naiso.interp1d(zl, dal, fill_values='extrapolate')
                    dai[:length(ind),j,i] = f(zi[int(ind[0]):])[::-1]

    return xr.DataArray(dai, dims=dim, coords=coord)
