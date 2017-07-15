import numpy as np
import xarray as xr
import pandas as pd
import functools as ft
import scipy.interpolate as naiso
import scipy.fftpack as fft
import scipy.integrate as intg
import scipy.signal as sig
import dask.array as dsar
import xgcm.grid as xgd

__all__ = ["sig2z","geo_streamfunc","geo_vel",
           "rel_vorticity","qgpv","pv_inversion"]

def _grid(ds, peri=False):
    return xgd.Grid(ds, periodic=peri)

def _interpolate(x,y,xnew):
    """
    Interpolates and flips the vertical coordinate as
    the bottom layer is at the top of the array in sigma coordinates.
    """
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
    zr : `numpy.array`
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
    if zr.shape != da.shape[-3:]:
        raise ValueError("`zr` should have the same "
                        "spatial dimensions as `da`.")

    dimd = da.dims
    N = da.shape
    nzi = len(zi)
    if len(N) == 4:
        dai = np.empty((N[0],nzi,N[-2],N[-1]))
        dim = [dimd[0],'z',dimd[-2],dimd[-1]]
        coord = {dimd[0]:da.coords[dimd[0]],
                'z':-zi, dimd[-2]:da.coords[dimd[-2]],
                dimd[-1]:da.coords[dimd[-1]]
                }
    elif len(N) == 3:
        dai = np.empty((nzi,N[-2],N[-1]))
        dim = ['z',dimd[-2],dimd[-1]]
        coord = {'z':-zi, dimd[-2]:da.coords[dimd[-2]],
                dimd[-1]:da.coords[dimd[-1]]
                }
    else:
        raise ValueError("The data should at least have three dimensions")
    dai[:] = np.nan

    zi = -zi[::-1] # ROMS has deepest level at index=0

    if nvar=='u':  # u variables
        zl = .5*(np.roll(np.roll(zr, -1, axis=-1), -1, axis=-2)
                 + np.roll(zr, -1, axis=-2)
                )
    elif nvar=='v': # v variables
        zl = .5*(np.roll(zr, -1, axis=-1)
                 + np.roll(np.roll(zr, -1, axis=-2), -1, axis=-1)
                )
    else:
        zl = zr

    for i in range(N[-1]):
        for j in range(N[-2]):
            # only bother for sufficiently deep regions
            if zl[:,j,i].min() < -1e2:
                # only interp on z above topo
                ind = np.argwhere(zi >= zl[:,j,i].min())
                if len(N) == 4:
                    for s in range(N[0]):
                        dai[s,:len(ind),j,i] = _interpolate(zl[:,j,i],
                                                            da[s,:,j,i].values,
                                                            zi[int(ind[0]):])
                else:
                    dai[:len(ind),j,i] = _interpolate(zl[:,j,i],
                                                     da[:,j,i].values,
                                                     zi[int(ind[0]):])

    return xr.DataArray(dai, dims=dim, coords=coord)

def geo_streamfunc(b, z, f0, inl=0., eta=None, ax=None):
    """
    Calculates the geostrophic streamfunction on \rho points
    based on the QG approximation.

    .. math::

     \psi = \frac{1}{f} \sum b \Delta z + \frac{g}{f} \eta

    Parameters
    ----------
    b : `xarray.DataArray`
        Buoyancy data
    z : `xarray.DataArray`
        The depths on which the buoyancy lies on
    f0 : `float`
        Coriolis parameter
    inl : float (optional)
        The initial value for the `scipy.interpolate.cumtrapz` function
    eta : `xarray.DataArray` (optional)
        Sea-surface height.
    ax : int (optional)
        The axis to take the integration over

    Returns
    -------
    psi : `xarray.DataArray`
        The geostrophic streamfunction on the
        same coordinate system as buoyancy
    """

    if b.ndim > 2:
        if b.dims[-3:] != z.dims:
            raise ValueError("`b` and `z` should have"
                            "the same spatial dimension.")

    g = 9.8
    psi = f0**-1 * intg.cumtrapz(b.values, x=z.values, axis=ax, initial=inl)
    if eta is not None:
        psi += g*f0**-1 * eta.values

    return xr.DataArray(psi, dims=b.dims, coords=b.coords)

def geo_vel(ds, ds_grid, psi, xname='x_rho', yname='y_rho',
            peri=False, shift=True):
    """
    Calculates the geostrophic velocity `ug` and `vg`
    on `u` and `v` points respectively
    from the geostrophic streamfunction.

    .. math

     u = -\frac{\partial \psi}{\partial y}
     v = \frac{\partial \psi}{\partial x}

    Parameters
    ----------
    ds : `xarray.Dataset`
        Object that includes all the data necessary
    ds : `xarray.Dataset`
        Object that includes the grid information
    psi : `xarray.Dataset`
        Geostrophic streamfunction
    shift : boolean (optional)
        If `True`, the geostrophic velocities will be interpolated
        to the center of each cell. If `False`, they will be on the
        same grid points as `u` and `v`.

    Returns
    -------
    ug, vg : `xarray.DataArray`
        Zonal and meridional geostrophic velocities.
    """

    grid = _grid(ds, peri=peri)
    x = xr.DataArray(ds_grid[xname].values, dims=psi[0,0].dims,
                     coords=psi[0,0].coords
                    )
    y = xr.DataArray(ds_grid[yname].values, dims=psi[0,0].dims,
                     coords=psi[0,0].coords
                    )
    ug = - grid.diff(psi,'Y') / grid.diff(y,'Y')
    vg = grid.diff(psi,'X') / grid.diff(x,'X')

    if shift:
        ug = grid.interp(ug, 'Y', boundary='fill')
        vg = grid.interp(vg, 'X', boundary='fill')

    # ug = - ((psi.shift(eta_rho=-1) - psi)
    #       / (psi.y.shift(eta_rho=-1) - psi.y)).isel(eta_rho=slice(None,-1))
    # vg = ((psi.shift(xi_rho=-1) - psi)
    #     / (psi.x.shift(xi_rho=-1) - psi.x)).isel(xi_rho=slice(None,-1))
    # ug = .25 * (ug + ug.shift(eta_rho=-1) + ug.shift(xi_rho=-1)
    #             + ug.shift(eta_rho=-1,xi_rho=-1)
    #            ).isel(eta_rho=slice(None,-1),
    #                   xi_rho=slice(None,-1)
    #                  )
    # vg = .25 * (vg + vg.shift(eta_rho=-1) + vg.shift(xi_rho=-1)
    #             + vg.shift(eta_rho=-1,xi_rho=-1)
    #            ).isel(eta_rho=slice(None,-1),
    #                   xi_rho=slice(None,-1)
    #                  )

    return ug, vg

def rel_vorticity(ds, ds_grid, uname='u', vname='v',
                  xname='x_v', yname='y_u', shift=True, peri=False):
    """
    Calculates the relative vorticity. ROMS applies a C-grid so the
    vorticity will be on \psi points.

    .. math::

     \zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

    Parameters
    ----------
    ds : `xarray.Dataset`
        Object that includes all the data necessary
    ds : `xarray.Dataset`
        Object that includes the grid information
    shift : boolean (optional)
        If `True`, the geostrophic velocities will be interpolated
        to the center of each cell. If `False`, they will be on the
        same grid points as `u` and `v`.

    Returns
    -------
    zeta : `xarray.DataArray`
        Relative vorticity. If `shift` is False, `\zeta` will be
        returned on `\psi` points.
    """

    grid = _grid(ds, peri=peri)
    u = ds[uname][:,::-1]
    v = ds[vname][:,::-1]
    x = ds_grid[xname]
    y = ds_grid[yname]
    if u.shape[-2:] != y.shape or v.shape[-2:] != x.shape:
        raise ValueError("The dimensions of u and y and/or "
                         "v and x do not match")
    x = xr.DataArray(x.values, dims=v[0,0].dims, coords=v[0,0].coords)
    y = xr.DataArray(y.values, dims=u[0,0].dims, coords=u[0,0].coords)
    # if u.shape != v.shape:
    #     raise ValueError("`u` and `v` should have the same shape.")
    dvdx = grid.diff(v,'X') / grid.diff(x,'X')
    dudy = grid.diff(u,'Y') / grid.diff(y,'Y')
    zeta = dvdx - dudy

    if shift:
        zeta = grid.interp(grid.interp(zeta, 'X', boundary='extend'),
                           'Y', boundary='extend'
                          )

    return zeta

def _interp_vgrid(nz,ny,nx,z,H):
    """Generate grid for vertical finite differences"""
    dzr = np.zeros((nz+1,ny,nx))
    dzr[0] = -z[0].values
    dzr[1:-1] = -z.diff('s_rho').values
    dzr[-1] = -H + z[-1].values
    dzp = np.zeros((nz,ny,nx))
    dzp[0] = dzr[0] + dzr[1]*.5
    dzp[1:-1] = .5*(dzr[1:-2]+dzr[2:-1])
    dzp[-1] = dzr[-2]*.5 + dzr[-1]

    # interface positions between rho points, where b will be interpolated
    zp = np.zeros_like(dzr)
    zp[0] = z[0]*.5
    zp[1:] = -np.cumsum(dzp, axis=-3) # zp[-1] = -H

    return dzr, dzp, zp

def qgpv(zeta, b, z, N2, zN2, f, eta, H, dim=None, coord=None):
    """
    Calculates the quasi-geostrophic PV on \rho points.

    .. math::

     q_surf = f0/H \times (b_surf/N^2 + \eta)
     q_int = f + \zeta + f0 \times \frac{d}{dz} (\frac{b}{N^2})

    Parameters
    ----------
    zeta : `xarray.DataArray`
        Relative vorticity.
    b : `xarray.DataArray`
        Buoyancy on \rho points.
    z : `xarray.DataArray`
        Depths which buoyancy is on.
    N2 : `numpy.array`
        Background buoyancy frequency squared.
    zN2 : `numpy.array`
        Depths which the background buoyancy frequency squared is on.
    f : `numpy.array`
        Coriolis parameter
    eta : `xarray.DataArray`
        Sea-surface height.
    H : `xarray.DataArray`
        Bathymetry depth

    Returns
    -------
    q : `xarray.DataArray`
        QGPV on \rho points
    """

    # if zeta.dims[-2:] != ('lat_psi', 'lon_psi'):
    #     raise ValueError("`zeta` should be on \psi points.")
    if b.dims[-3:] != z.dims:
        raise ValueError("`b` and `z` should have "
                        "the same spatial dimension.")
    if (H > z[-1]).values.sum() != 0:
        raise ValueError("Bathymetry shouldn't be shallower "
                        "than the last grid point of `b`")

    N = z.shape
    N2_intrp = np.zeros((N[0]+1,N[1],N[2]))
    N2_intrp[:] = np.nan
    if b.ndim == 3:
        b_intrp = N2_intrp.copy()
    elif b.ndim == 4:
        b_intrp = np.zeros((b.shape[0], N[0]+1, N[1], N[2]))
        b_intrp[:] = np.nan

    dzr, dzp, zp = _interp_vgrid(N[0],N[1],N[2],z,H)

    fN = naiso.interp1d(zN2, N2, fill_value='extrapolate')
    for j in range(N[-2]):
        for i in range(N[-1]):
            N2_intrp[:,j,i] = fN(zp[:,j,i])
            if b.ndim == 3:
                b_intrp[:,j,i] = _interpolate(z[:,j,i], b[:,j,i],
                                             zp[:,j,i])[::-1]
            elif b.ndim == 4:
                for t in range(b.shape[0]):
                    b_intrp[t,:,j,i] = _interpolate(z[:,j,i], b[t,:,j,i],
                                                   zp[:,j,i])[::-1]

    # # move zeta to \rho points
    # zeta = .25 * (zeta + zeta.shift(lat_psi=-1) + zeta.shift(lon_psi=-1)
    #              + zeta.shift(lat_psi=-1, lon_psi=-1)
    #              ).isel(lat_psi=slice(None,-1), lon_psi=slice(None,-1)
    #                    ).values

    f0 = f.mean()
    q_int = (f + zeta
             + f0 * (np.diff(b_intrp * N2_intrp**-1, axis=-3)
                     * np.diff(zp, axis=-3)**-1
                    )
            )
    q = np.empty_like(b_intrp)
    q[:] = np.nan
    if q.ndim == 4:
        q[:,0] = f0*(-H.values**-1) * (b_intrp[:,0]*N2_intrp[0]**-1
                                      + eta.values)
        q[:,1:] = q_int
    elif q.ndim == 3:
        q[0] = f0*(-H.values**-1) * (b_intrp[0]*N2_intrp[0]**-1
                                    + eta.values)
        q[1:] = q_int

    return xr.DataArray(q, dims=dim, coords=coord)

def _hanning(nx,ny):
    """Hanning window"""
    return sig.hanning(nx) * sig.hanning(ny)[:,np.newaxis]

def pv_inversion(psi, N2, zN2, H, f0, dx, dy):
    """
    PV inversion

    Parameters
    ----------
    psi : `xarray.DataArray`
        Geostrophic streamfunction
    N2 : `numpy.array`
        Background buoyancy frequency
    zN2 : `numpy.array`
        Depths at which `N2` is on
    H : `xarray.DataArray`
        Bathymetry Depths (meters)
    f0 : float
        f-plane coriolis parameter
    dx, dy : float
        Representative grid scale of the domain.
        Used for getting the wavenumbers.

    Returns
    -------
    qgpv : `xarray.DataArray`
        Quasi-geostrophic potential vorticity
    """
    g = 9.8
    N = psi.shape

    dzr, dzp, zp = _interp_vgrid(N[0],N[1],N[2],psi.z,H)

    fN = naiso.interp1d(zN2, N2, fill_value='extrapolate')
    A = np.zeros((N[0]+1, N[0]+1, N[1], N[2]))
    for j in range(N[-2]):
        for i in range(N[-1]):
            N2_intrp = fN(zp[:,j,i])

            r = f0**2/(N2_intrp*dzr)
            rm = (r[0] + f0**2/g)/H  # coef of psi_s
            ru = -r[0]/H

            Adn = r[:-1]/dzp
            Aup = np.zeros(N[0])
            Aup[0] = ru
            Aup[1:] = r[1:-1]/dzp[:-1]
            Amid = np.zeros(N[0]+1)
            Amid[0] = rm
            Amid[1:-1] = -(r[:-2]+r[1:-1])/dzp[:-1]
            Amid[-1] = -r[-2]/dzp[-1]

            A[:,:,j,i] = np.diag(Adn,-1) + np.diag(Amid) + np.diag(Aup,1)

            # # for use with tridiag, append Aup and Adn with 0s
            # Aupa = np.append(Aup, 0.)
            # Adna = np.append(0, Adn)

    psigk = np.zeros_like(psi)
    for k in range(N[-3]):
        psigk[k] = fft.fftshift(fft.fft2(psi[k]
                                * _hanning(N[-1],N[-2]))
                               ) * (N[-1]*N[-2])**-1
        zetag[k] = np.real(fft.ifft2(fft.ifftshift(-K2_ * psigk
                                                * N[-1]*N[-2]))
                        )

    qgpvk = np.zeros_like(psi)
    kx = np.arange(-int(N[-1]/2)+1,int(N[-1]/2)) * 2*np.pi/(N[-1]*dx)  # need dimensional wavenumbers!
    ky = np.arange(-int(N[-2]/2)+1,int(N[-2]/2)) * 2*np.pi/(N[-2]*dy)
    kx_, ky_ = np.meshgrid(kx, ky);
    K2_ = kx_**2 + ky_**2 ;
    for j in range(len(ky)):
        for i in range(len(kx)):
            Ak = A[:,:,j,i].copy()
            Ak[1:,1:] -= np.eye(N[0])*K2_[j,i]
            qgpvk[:,j,i] = np.dot(Ak, psigk[:,j,i])

    qg = np.zeros_like(psi)
    for k in range(N[0]):
        qgpv[k] = np.real(fft.ifft2(fft.ifftshift(qgpvk[k])))*(N[-2]*N[-1])

    return xr.DataArray(qgpv, dims=psi.dims, coords=psi.coords)
