import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt
import scipy.interpolate as naiso
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

def test_streamfunc():
    N = 16
    da4d = xr.DataArray(np.random.rand(N,N,N,N),
                    dims=['time','z','eta_rho','xi_u'],
                    coords={'time':range(N),'z':range(N),'eta_rho':range(N),
                            'xi_u':range(N)}
                    )
    z = xr.DataArray(np.random.rand(N,N,N),
                    dims=['z','eta_rho','xi_rho'],
                    coords={'z':range(N),'eta_rho':range(N),'xi_rho':range(N)}
                    )
    with pytest.raises(ValueError):
        xm.geo_streamfunc(da4d, z, 0.)

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

def test_qgpv():
    #######
    # Eady (q = 0)
    #######
    N = 16
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    zz = -np.arange(N)[:,np.newaxis,np.newaxis] * np.ones((N,N))
    v = xr.DataArray(np.zeros((N,N,N)),
                    dims=['s','eta_v','xi_rho'],
                    coords={'s':range(N),'eta_v':range(N),'xi_rho':range(N),
                           'z':(('s','eta_v','xi_rho'),zz),
                           'y':(('eta_v','xi_rho'),yy),
                           'x':(('eta_v','xi_rho'),xx)}
                    )
    u = xr.DataArray((N-np.arange(1,N+1))[:,np.newaxis,
                                            np.newaxis]*np.ones((N,N)),
                    dims=['s','eta_rho','xi_u'],
                    coords={'s':range(N),'eta_rho':range(N),'xi_u':range(N),
                           'z':(('s','eta_rho','xi_u'),zz),
                           'y':(('eta_rho','xi_u'),yy),
                           'x':(('eta_rho','xi_u'),xx)}
                    )
    zeta = xr.DataArray(xm.rel_vorticity(u, v, v.x, u.y),
                        dims=['s_rho','eta_rho','xi_rho'],
                        coords={'s_rho':range(N),'eta_rho':range(N-2),
                                'xi_rho':range(N-2),
                               'z':(('s_rho','eta_rho','xi_rho'),
                                    zz[:,1:-1,1:-1]),
                               'y':(('eta_rho','xi_rho'),yy[1:-1,1:-1]),
                               'x':(('eta_rho','xi_rho'),xx[1:-1,1:-1])}
                        )
    npt.assert_allclose(zeta.values, 0.)

    f = np.ones((N,N))
    N2 = np.ones(N-1)
    zN2 = -.5 * (np.arange(N)[1:]+np.arange(N)[:-1])
    eta = xr.DataArray(np.zeros((N-2,N-2)),
                       dims=zeta[0].dims, coords=zeta[0].coords
                      )

    dbdy = -f*u.diff('s')/u.z.diff('s')
    zb = .5 * (u.z[1:,0,0].values + u.z[:-1,0,0].values)
    b = np.cumsum(dbdy * u.y.diff('eta_rho')[0,0].values, axis=1)

    H = xr.DataArray(-(.1 + (N-1)*np.ones((N-2,N-2))),
                     dims=zeta[0].dims, coords=zeta[0].coords
                    )
    with pytest.raises(ValueError):
        xm.qgpv(zeta, xr.DataArray(b, dims=['s_b','eta_rho','xi_u'],
                                   coords={'s_b':range(N-1),
                                           'eta_rho':range(N),
                                           'xi_u':range(N)})[:,1:-1,1:-1],
                u.z, N2, zN2, f, eta, H)

    bnew = np.zeros_like(u)
    for j in range(N):
        for i in range(N):
            fb = naiso.interp1d(zb, b[:,j,i], fill_value='extrapolate')
            bnew[:,j,i] = fb(u.z.values[:,0,0])
    q = xm.qgpv(zeta, xr.DataArray(bnew, dims=['s_rho','eta_rho','xi_rho'],
                                   coords={'s_rho':range(N),
                                           'eta_rho':range(N),
                                           'xi_rho':range(N)})[:,1:-1,1:-1],
                zeta.z, N2, zN2, f[1:-1,1:-1], eta, H
               )
    npt.assert_allclose(q[1], 1.)

    H = -(-.1 + (N-1)*np.ones((N-2,N-2)))
    with pytest.raises(ValueError):
        xm.qgpv(zeta, xr.DataArray(bnew, dims=['s','eta_rho','xi_u'],
                                   coords={'s':range(N),
                                           'eta_rho':range(N),
                                           'xi_u':range(N)})[:,1:-1,1:-1],
                u.z, N2, zN2, f, eta, H)
