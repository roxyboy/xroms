import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt
import pytest
import xroms as xm

def test_sig2z():
    N = 16
    x = np.arange(N+1)
    y = np.arange(N-1)
    t = np.linspace(-int(N/2), int(N/2), N-6)
    z = np.arange(int(N/2))
    d4d = (t[:,np.newaxis,np.newaxis,np.newaxis]
            + z[np.newaxis,:,np.newaxis,np.newaxis]
            + y[np.newaxis,np.newaxis,:,np.newaxis]
            + x[np.newaxis,np.newaxis,np.newaxis,:]
          )
    da4d = xr.DataArray(d4d, dims=['time','z','y','x'],
                     coords={'time':range(len(t)),'z':range(len(z)),'y':range(len(y)),
                             'x':range(len(x))}
                     )
    znew = np.linspace(z.min(), z.max(), N)
    znew[1] = -1
    with pytest.raises(ValueError):
        xm.sig2z(da4d, z, znew)

    znew = -np.linspace(z.min(), z.max(), N)
    with pytest.raises(ValueError):
        xm.sig2z(da4d, z, znew)

def test_zeta():
    N = 16
    u = xr.DataArray(np.random.rand(N,N,N,N),
                    dims=['time','z','eta_rho','xi_u'],
                    coords={'time':range(N),'z':range(N),'eta_rho':range(N),
                            'xi_u':range(N)}
                    )
    v = xr.DataArray(np.random.rand(N,N,N,N),
                    dims=['time','z','eta_v','xi_rho'],
                    coords={'time':range(N),'z':range(N),'eta_v':range(N),
                            'xi_rho':range(N)}
                      )
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    xx = xr.DataArray(xx, dims=['y','x'], coords={'y':range(N), 'x':range(N)})
    yy = xr.DataArray(yy, dims=['y','x'], coords={'y':range(N), 'x':range(N)})
    with pytest.raises(ValueError):
        xm.rel_vorticity(u, v, xx, yy)
