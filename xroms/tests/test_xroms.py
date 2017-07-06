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
