import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import scipy.interpolate as naiso
import dask.array as dsar

__all__ = ["sig2z"]

def _interpolate(x,y,xnew):
    f = naiso.interp1d(x,y,
                    fill_value='extrapolate')
    return f(xnew)[::-1]

def sig2z(da, zr, zi, nvar=None):
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
        Name of the variable. Only necessary when the variable is
        horizontal velocity.

    Returns
    -------
    dai : `xarray.DataArray`
        The data interpolated onto a spatial uniform z coordinate
    """

    if np.diff(zi)[0] < 0. or zi.max() <= 0.:
        raise ValueError("The values in `zi` should be postive and increasing.")
    if zr.ndim > da.ndim:
        raise ValueError("`da` should have the same or more dimensions than `zr`")

    dimd = da.dims
    N = da.shape
    nzi = len(zi)
    if len(N) == 4:
        dai = np.empty((N[0],nzi,N[-2],N[-1]))
        dim = [dimd[0],'z',dimd[-2],dimd[-1]]
        coord = {dimd[0]:da.coords[dimd[0]],
                'z':zi, dimd[-2]:da.coords[dimd[-2]],
                dimd[-1]:da.coords[dimd[-1]]
                }
    elif len(N) == 3:
        dai = np.empty((nzi,N[-2],N[-1]))
        dim = ['z',dimd[-2],dimd[-1]]
        coord = {'z':zi, dimd[-2]:da.coords[dimd[-2]],
                dimd[-1]:da.coords[dimd[-1]]
                }
    else:
        raise ValueError("The data should at least have three dimensions")
    dai[:] = np.nan

    zi = -zi[::-1] # ROMS has deepest level at index=0

    if nvar=='u':  # u variables
        zl = .5*(zr.shift(eta_rho=-1,xi_rho=-1)
                 +zr.shift(eta_rho=-1)
                )
    elif nvar=='v': # v variables
        zl = .5*(zr.shift(xi_rho=-1)
                 +zr.shift(eta_rho=-1,xi_rho=-1)
                )
    else:
        zl = zr

    for i in range(N[-1]):
        for j in range(N[-2]):
            # only bother for sufficiently deep regions
            if zl[:,j,i].values.min() < -1e2:
                # only interp on z above topo
                ind = np.argwhere(zi >= zl[:,j,i].values.min())
                if len(N) == 4:
                    for s in range(N[0]):
                        dal = da[s,:,j,i].values
                        dai[s,:len(ind),j,i] = _interpolate(zl[:,j,i].values,
                                                            dal,
                                                            zi[int(ind[0]):])
                else:
                    dal = da[:,j,i].values
                    dai[:len(ind),j,i] = _interpolate(zl[:,j,i].values,
                                                     dal,
                                                     zi[int(ind[0]):])

    return xr.DataArray(dai, dims=dim, coords=coord)
