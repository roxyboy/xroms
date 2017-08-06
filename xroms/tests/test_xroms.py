import numpy as np
import pandas as pd
import xarray as xr
import numpy.testing as npt
import scipy.interpolate as naiso
import pytest
import xroms as xm
import xgcm.grid as xgd

# from . datasets import (all_datasets, nonperiodic_1d, periodic_1d, periodic_2d,
#                         nonperiodic_2d, all_2d, datasets, nonperiodic_4d_left)

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
    ds = xr.DataArray(np.random.rand(N,N,N,N),
                    dims=['time','z','eta_rho','xi_u'],
                    coords={'time':range(N),'z':range(N),'eta_rho':range(N),
                            'xi_u':range(N)}
                    ).to_dataset(name='u')
    ds['v'] = xr.DataArray(np.random.rand(N,N,N,N),
                    dims=['time','z','eta_v','xi_rho'],
                    coords={'time':range(N),'z':range(N),'eta_v':range(N),
                            'xi_rho':range(N)}
                      )
    xx, yy = np.meshgrid(np.arange(N-1), np.arange(N))
    ds_grid = xr.DataArray(xx, dims=['y','x'],
                           coords={'y':range(N), 'x':range(N-1)}
                          ).to_dataset(name='x_rho')
    ds_grid['y_rho'] = xr.DataArray(yy, dims=['y','x'],
                                coords={'y':range(N), 'x':range(N-1)})
    with pytest.raises(ValueError):
        xm.rel_vorticity(ds.u, ds.v, ds, ds_grid, xname='x_rho', yname='y_rho')

    #######
    # Eady (\zeta = 0)
    #######
    # xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    # # zz = -np.arange(N)[:,np.newaxis,np.newaxis] * np.ones((N,N))
    # ds_grid = xr.DataArray(xx, dims=['eta_v','xi_v'],
    #                        coords={'eta_v':range(N), 'xi_v':range(N)}
    #                       ).to_dataset('x_v')
    # ds_grid['y_u'] = xr.DataArray(yy, dims=['eta_u','xi_u'],
    #                             coords={'eta_u':range(N), 'xi_u':range(N)})
    # # ds_grid['z'] = xr.DataArray(zz, dims=['s','y','x'],
    # #                              coords={'z':range(N), 'y':range(N),
    # #                              'x':range(N)})
    # ds = xr.DataArray(np.ones((N,N,N)),
    #                   dims=['s','eta_v','xi_rho'],
    #                   coords={'s':range(N),'eta_v':range(N),'xi_rho':range(N)}
    #                  ).to_dataset(name='v')
    # ds['u'] = (N-1)**-1 * xr.DataArray((N-np.arange(1,N+1))[:,np.newaxis,
    #                                             np.newaxis]*np.ones((N,N)),
    #                                dims=['s','eta_rho','xi_u'],
    #                                coords={'s':range(N),'eta_rho':range(N),
    #                                        'xi_u':range(N)}
    #                               )
    # zeta = xm.rel_vorticity(ds, ds_grid)
    # npt.assert_allclose(zeta.values, 0.)

def test_qgpv():
    #######
    # Eady (q = f0)
    #######
    N = 16
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    zz = -np.arange(N)[:,np.newaxis,np.newaxis] * np.ones((N,N))
    # v = xr.DataArray(np.zeros((N,N,N)),
    #                 dims=['s','eta_v','xi_rho'],
    #                 coords={'s':range(N),'eta_v':range(N),'xi_rho':range(N),
    #                        'z':(('s','eta_v','xi_rho'),zz),
    #                        'y':(('eta_v','xi_rho'),yy),
    #                        'x':(('eta_v','xi_rho'),xx)}
    #                 )
    u = xr.DataArray((N-np.arange(1,N+1))[:,np.newaxis,
                                            np.newaxis]*np.ones((N,N)),
                    dims=['s','eta_rho','xi_u'],
                    coords={'s':range(N),'eta_rho':range(N),'xi_u':range(N),
                           'z':(('s','eta_rho','xi_u'),zz),
                           'y':(('eta_rho','xi_u'),yy),
                           'x':(('eta_rho','xi_u'),xx)}
                    )
    # zeta = xr.DataArray(xm.rel_vorticity(u, v, v.x, u.y),
    #                     dims=['s_rho','eta_rho','xi_rho'],
    #                     coords={'s_rho':range(N),'eta_rho':range(N-2),
    #                             'xi_rho':range(N-2),
    #                            'z':(('s_rho','eta_rho','xi_rho'),
    #                                 zz[:,1:-1,1:-1]),
    #                            'y':(('eta_rho','xi_rho'),yy[1:-1,1:-1]),
    #                            'x':(('eta_rho','xi_rho'),xx[1:-1,1:-1])}
    #                     )
    # npt.assert_allclose(zeta.values, 0.)
    zeta = xr.DataArray(np.zeros((N,N,N)),
                        dims=['s_rho','eta_rho','xi_rho'],
                        coords={'s_rho':range(N),'eta_rho':range(N),
                                'xi_rho':range(N),
                               'z':(('s_rho','eta_rho','xi_rho'),
                                    zz),
                               'y':(('eta_rho','xi_rho'),yy),
                               'x':(('eta_rho','xi_rho'),xx)}
                       )

    f = np.ones((N,N))
    N2 = np.ones(N-1)
    zN2 = -.5 * (np.arange(N)[1:]+np.arange(N)[:-1])
    eta = xr.DataArray(np.zeros((N,N)),
                       dims=zeta[0].dims, coords=zeta[0].coords
                      )

    dbdy = -f*u.diff('s')/u.z.diff('s')
    zb = .5 * (u.z[1:,0,0].values + u.z[:-1,0,0].values)
    b = np.cumsum(dbdy * u.y.diff('eta_rho')[0,0].values, axis=1)

    H = xr.DataArray(-(.1 + (N-1)*np.ones((N,N))),
                     dims=zeta[0].dims, coords=zeta[0].coords
                    )
    with pytest.raises(ValueError):
        xm.qgpv(zeta, xr.DataArray(b, dims=['s_b','eta_rho','xi_u'],
                                   coords={'s_b':range(N-1),
                                           'eta_rho':range(N),
                                           'xi_u':range(N)}),
                u.z, N2, zN2, f, eta, H)

    bnew = np.zeros_like(u)
    for j in range(N):
        for i in range(N):
            fb = naiso.interp1d(zb, b[:,j,i], fill_value='extrapolate')
            bnew[:,j,i] = fb(u.z.values[:,0,0])
    q = xm.qgpv(zeta, xr.DataArray(bnew, dims=['s_rho','eta_rho','xi_rho'],
                                   coords={'s_rho':range(N),
                                           'eta_rho':range(N),
                                           'xi_rho':range(N)}),
                zeta.z, N2, zN2, f, eta, H
               )
    npt.assert_allclose(q[1]-f.mean(), 0.)

    H = -(-.1 + (N-1)*np.ones((N,N)))
    with pytest.raises(ValueError):
        xm.qgpv(zeta, xr.DataArray(bnew, dims=['s','eta_rho','xi_u'],
                                   coords={'s':range(N),
                                           'eta_rho':range(N),
                                           'xi_u':range(N)}),
                u.z, N2, zN2, f, eta, H)
