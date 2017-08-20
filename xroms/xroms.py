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
import warnings

__all__ = ["set_coords","sig2z","geo_streamfunc","geo_vel",
           "rel_vorticity","generalized_qgpv","pv_inversion"]

# convert everything that doesn't have a time dimension to coord
def set_coords(ds):
    new_coords = [varname for varname in ds.data_vars
                  if ('time' not in ds[varname].dims)]
    ds_new = ds.set_coords(new_coords)
    return ds_new

def _grid(ds, peri):
    return xgd.Grid(ds, periodic=peri)

def _interpolate(x,y,xnew):
    """
    Interpolates and flips the vertical coordinate so as
    the bottom layer is at the top of the array in sigma coordinates.
    """
    f = naiso.interp1d(x,y,
                    fill_value='extrapolate')
    return f(xnew)[::-1]

def sig2z(da, zr, zi, nvar=None, dim=None, coord=None):
    """
    Interpolate variables on \sigma coordinates onto z coordinates.

    Parameters
    ----------
    da : `dask.array`
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
    dai : `dask.array`
        The data interpolated onto a spatial uniform z coordinate
    """

    if np.diff(zi)[0] < 0. or zi.max() <= 0.:
        raise ValueError("The values in `zi` should be postive and increasing.")
    if np.any(np.absolute(zr[0]) < np.absolute(zr[-1])):
        raise ValueError("`zr` should have the deepest depth at index 0.")
    if zr.shape != da.shape[-3:]:
        raise ValueError("`zr` should have the same "
                        "spatial dimensions as `da`.")

    if dim == None:
        dim = da.dims
    if coord == None:
        coord = da.coords
    N = da.shape
    nzi = len(zi)
    if len(N) == 4:
        dai = np.empty((N[0],nzi,N[-2],N[-1]))
        # dim = [dimd[0],'z',dimd[-2],dimd[-1]]
        # coord = {dimd[0]:da.coords[dimd[0]],
        #         'z':-zi, dimd[-2]:da.coords[dimd[-2]],
        #         dimd[-1]:da.coords[dimd[-1]]
        #         }
    elif len(N) == 3:
        dai = np.empty((nzi,N[-2],N[-1]))
        # dim = ['z',dimd[-2],dimd[-1]]
        # coord = {'z':-zi, dimd[-2]:da.coords[dimd[-2]],
        #         dimd[-1]:da.coords[dimd[-1]]
        #         }
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

    if type(z)==xr.DataArray and b.ndim > 2:
        if b.dims[-3:] != z.dims:
            raise ValueError("`b` and `z` should have "
                             "the same spatial dimension.")
        z = z.values

    g = 9.8
    psi = f0**-1 * intg.cumtrapz(b.values, x=z, axis=ax, initial=inl)
    if eta is not None:
        psi += g*f0**-1 * eta.values[np.newaxis,:,:]

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
    ds_grid : `xarray.Dataset`
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

    grid = _grid(ds, peri)
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

def rel_vorticity(u, v, ds, ds_grid,
                  xname='x_v', yname='y_u', shift=True, peri=False):
    """
    Calculates the relative vorticity. ROMS applies a C-grid so the
    vorticity will be on \psi points.

    .. math::

     \zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

    Parameters
    ----------
    u, v : `xarray.DataArray`
        Zonal and meridional velocity respectively
    ds_grid : `xarray.Dataset`
        Object that includes the grid information
    shift : boolean (optional)
        If `True`, the geostrophic velocities will be shifted
        half a cell.

    Returns
    -------
    zeta : `xarray.DataArray`
        Relative vorticity.
    """

    grid = _grid(ds, peri)
    x = ds_grid[xname]
    y = ds_grid[yname]
    if u.dims[-2:] != y.dims or v.dims[-2:] != x.dims:
        warnings.warn("The dimensions of `u` and `y` and/or "
                      "`v` and `x` do not match.")
        x = xr.DataArray(x.values, dims=v[0,0].dims, coords=v[0,0].coords)
        y = xr.DataArray(y.values, dims=u[0,0].dims, coords=u[0,0].coords)
    # if u.shape != v.shape:
    #     raise ValueError("`u` and `v` should have the same shape.")
    dvdx = grid.diff(v,'X') / grid.diff(x,'X')
    dudy = grid.diff(u,'Y') / grid.diff(y,'Y')
    if dvdx.dims != dudy.dims:
        dvdx = grid.interp(dvdx,'X',boundary='fill')
        dudy = grid.interp(dudy,'Y',boundary='fill')

    zeta = dvdx - dudy

    if shift:
        zeta = grid.interp(grid.interp(zeta, 'X', boundary='extend'),
                           'Y', boundary='extend'
                          )

    return zeta

def _interp_vgrid(nz,ny,nx,z,H):
    """
    Generate grid for vertical finite differences.

    .. math::

     dzr = [-z[0], z[0]-z[1], z[1]-z[2], ..., z[-1]-H]
     dzp &= [dzr[0]+dzr[1]/2, (dzr[1]+dzr[2])/2, ..., (dzr[-3]+dzr[-2])/2, dzr[-2]/2+dzr[-1]]\\
         &= [(z[0]+z[1])/2, (z[0]-z[2])/2, (z[1]-z[3])/1, ..., (z[-2]+z[-1])/2-H]
     zp = [z[0]/2, (z[0]+z[1])/2, (z[1]+z[2])/2, ..., (z[-2]+z[-1])/2, H]
    """
    dzr = np.zeros((nz+1,ny,nx))
    dzr[0] = -z[0].values
    dzr[1:-1] = -z.diff('s_rho').values
    dzr[-1] = np.absolute(H) + z[-1].values
    dzp = np.zeros((nz,ny,nx))
    dzp[0] = dzr[0] + dzr[1]*.5
    dzp[1:-1] = .5*(dzr[1:-2]+dzr[2:-1])
    dzp[-1] = dzr[-2]*.5 + dzr[-1]

    # interface positions between rho points, where b will be interpolated
    zp = np.zeros_like(dzr)
    zp[0] = z[0]*.5
    zp[1:] = -np.cumsum(dzp, axis=-3) # zp[-1] = H

    return dzr, dzp, zp

def generalized_qgpv(zeta, b, z, N2, zN2, f, eta, H,
         dim=None, coord=None, bottom='flat',
         intrp='both', native_grid=True):
    """
    Calculates the quasi-geostrophic PV on \rho points.

    .. math::

     q+ = f0/H \times (b[0]/N[0]^2 + \eta)
     q = \zeta + f0 \times \frac{d}{dz} (\frac{b}{N^2})
     q- = f0/H \times (b[-1]/N[-1]^2 + \eta)

    Parameters
    ----------
    zeta : `xarray.DataArray`
        Relative vorticity.
    b : `xarray.DataArray`
        Buoyancy on `rho` points.
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
    bottom : `str`
        If bottom is `flat`, the quiescent bottom boundary condition
        will be applied. If `sloped`, boundary condition with
        a sloped bottom will be applied.
    intrp : `str`
        If `both`, b and N2 will be interpolated in the vertical axis.
        If 'N2', only N2 will be interpolated.
    native_grid : `bool`
        If `True`, the coordinate is assumed to be in the native grid
        of the outputs.

    Returns
    -------
    q : `xarray.DataArray`
        QGPV on `rho` points with the Coriolis parameter added.
    """

    # if zeta.dims[-2:] != ('lat_psi', 'lon_psi'):
    #     raise ValueError("`zeta` should be on \psi points.")
    if b.dims[-3:] != z.dims:
        raise ValueError("`b` and `z` should have "
                        "the same spatial dimension.")
    if native_grid:
        if (H > z[-1]).values.sum() != 0:
            raise ValueError("Bathymetry shouldn't be shallower "
                            "than the last grid point of `b`")

    N = z.shape
    fN = naiso.interp1d(zN2, N2, fill_value='extrapolate')
    if intrp == 'both':
        dzr, dzp, zp = _interp_vgrid(N[0],N[1],N[2],z,H)
        zi = zp.copy()
        N2_intrp = np.zeros((N[0]+1,N[1],N[2]))
        N2_intrp[:] = np.nan
        if b.ndim == 3:
            b_intrp = N2_intrp.copy()
        elif b.ndim == 4:
            b_intrp = np.zeros((b.shape[0], N[0]+1, N[1], N[2]))
            b_intrp[:] = np.nan

        for j in range(N[-2]):
            for i in range(N[-1]):
                N2_intrp[:,j,i] = fN(zp[:,j,i])
                if b.ndim == 3:
                    b_intrp[:,j,i] = _interpolate(z[:,j,i], b[:,j,i],
                                                 zi[:,j,i])[::-1]
                elif b.ndim == 4:
                    for t in range(b.shape[0]):
                        b_intrp[t,:,j,i] = _interpolate(z[:,j,i], b[t,:,j,i],
                                                       zi[:,j,i])[::-1]
    elif intrp == 'N2':
        zi = z
        N2_intrp = np.zeros((N[0],N[1],N[2]))
        N2_intrp[:] = np.nan
        for j in range(N[-2]):
            for i in range(N[-1]):
                N2_intrp[:,j,i] = fN(z[:,j,i])
        b_intrp = b

    # # move zeta to \rho points
    # zeta = .25 * (zeta + zeta.shift(lat_psi=-1) + zeta.shift(lon_psi=-1)
    #              + zeta.shift(lat_psi=-1, lon_psi=-1)
    #              ).isel(lat_psi=slice(None,-1), lon_psi=slice(None,-1)
    #                    ).values

    f0 = f.mean()
    q = np.zeros((b.shape[0], N[0]+2, N[1], N[2]))
    ddzbN2 = (np.diff(b_intrp * N2_intrp**-1, axis=-3)
              * np.diff(zi, axis=-3)**-1
             )
    if zeta.shape == b_intrp.shape:
        ddzbN2_intrp = np.zeros_like(b)
        for j in range(N[-2]):
            for i in range(N[-1]):
                if b.ndim == 3:
                    ddzbN2_intrp[:,j,i] = _interpolate(z[:,j,i],
                                                 ddzbN2[:,j,i],
                                                 zi[:,j,i])[::-1]
                elif b.ndim == 4:
                    for t in range(b.shape[0]):
                        ddzbN2_intrp[t,:,j,i] = _interpolate(z[:,j,i],
                                                       ddzbN2[t,:,j,i],
                                                       zi[:,j,i])[::-1]
    else:
        ddzbN2_intrp = ddzbN2

    qgpv = zeta + f0 * ddzbN2_intrp
    if q.ndim == 4:
        q[:,0] = f0*np.absolute(H.values)**-1 * (b_intrp[:,0]*N2_intrp[0]**-1
                                      + eta.values)
        q[:,1:-1] = qgpv
        if bottom=='flat':
            pass
        elif bottom=='sloped':
            q[:,-1] = f0*np.absolute(H.values)**-1 * (b_intrp[:,-1]
                                                      * N2_intrp[-1]**-1
                                      + eta.values)
        else:
            raise NotImplementedError("Unknown bottom boundary condition "
                                      "specified.")
    elif q.ndim == 3:
        q[0] = f0*np.absolute(H.values)**-1 * (b_intrp[:,0]*N2_intrp[0]**-1
                                      + eta.values)
        q[1:-1] = qgpv
        if bottom=='flat':
            pass
        elif bottom=='sloped':
            q[-1] = f0*np.absolute(H.values)**-1 * (b_intrp[:,-1]
                                                    * N2_intrp[-1]**-1
                                      + eta.values)
        else:
            raise NotImplementedError("Unknown bottom boundary condition "
                                      "specified.")

    return xr.DataArray(q, dims=dim, coords=coord)

def _hanning(nx,ny):
    """Hanning window"""
    return sig.hanning(nx) * sig.hanning(ny)[:,np.newaxis]

def pv_inversion(psi, z, N2, zN2, H, f0, dx, dy,
                dim=None, coord=None, window=False):
    """
    QGPV inversion from the geostrophic streamfunction.

    .. math::

     The second-order derivative of an arbitrary variable `phi` is

     \frac{d^2 \phi}{dz^2} = &\frac{1}{dzp[i]} \bigg( \frac{\phi_i}{dzr[i]} \\
                             &- (\frac{1}{dzr[i]} + \frac{1}{dzr[i+1]})\phi_{i+1}
                             &+ \frac{\phi_{i+2}}{dzr[i+1]} \bigg)

     so the inversion matrix for the vertical derivative becomes

     A = \left( \begin{array}{ccccccc}
            -\frac{r[0] + f_0^2/g}{H} & \frac{r[0]}{H} & 0 & 0 & 0 & ... & 0 \\
            \frac{r[0]}{dzp[0]} & -\frac{r[0]+r[1]}{dzp[0]} & \frac{r[1]}{dzp[0]} & 0 & 0 & ... & 0 \\
            0 & \frac{r[1]}{dzp[1]} & -\frac{r[1]+r[2]}{dzp[1]} & \frac{r[2]}{dzp[1]} & 0 & ... & 0\\
            ... \end{array}
         \right)

     where $r = \frac{f_0^2}{N^2} \frac{1}{dzr}$.

    Parameters
    ----------
    psi : `xarray.DataArray`
        Geostrophic streamfunction
    z : `xarray.DataArray`
        Depths as which `psi` is on
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
    if psi.ndim != 3:
        raise ValueError("`psi` is expected to have only "
                        "three (spatial) dimensions.")
    if psi.dims != z.dims:
        raise ValueError("`psi` and `z` should have "
                        "the same dimensions.")
    N = psi.shape
    psi_intrp = np.zeros((N[0]+1, N[1], N[2]))

    dzr, dzp, zp = _interp_vgrid(N[0], N[1], N[2], z, H)

    fN = naiso.interp1d(zN2, N2, fill_value='extrapolate')
    A = np.zeros((N[0]+1, N[0]+1, N[1], N[2]))
    for j in range(N[-2]):
        for i in range(N[-1]):
            N2_intrp = fN(zp[:,j,i])
            psi_intrp[:,j,i] = _interpolate(z[:,j,i], psi[:,j,i],
                                            zp[:,j,i])[::-1]

            r = f0**2 / (N2_intrp * dzr[:,j,i])
            rm = (r[0] + f0**2/g)/np.absolute(H[j,i])  # coef of psi_s
            ru = -r[0]/np.absolute(H[j,i])

            Adn = r[:-1]/dzp[:,j,i]
            Aup = np.zeros(N[0])
            Aup[0] = ru
            Aup[1:] = r[1:-1]/dzp[:-1,j,i]
            Amid = np.zeros(N[0]+1)
            Amid[0] = rm
            Amid[1:-1] = -(r[:-2]+r[1:-1])/dzp[:-1,j,i]
            Amid[-1] = -r[-2]/dzp[-1,j,i]

            A[:,:,j,i] = np.diag(Adn,-1) + np.diag(Amid) + np.diag(Aup,1)

    kx = fft.fftshift(fft.fftfreq(N[2], dx)) * 2.*np.pi
    ky = fft.fftshift(fft.fftfreq(N[1], dy)) * 2.*np.pi
    K2_ = kx[np.newaxis,:]**2 + ky[:,np.newaxis]**2

    psigk = np.zeros_like(psi_intrp, dtype=complex)
    zetag = np.zeros_like(psi)
    for k in range(N[0]+1):
        if window:
            psigk[k] = fft.fftshift(fft.fft2(psi_intrp[k]
                                             * _hanning(N[-1],N[-2])
                                            )
                                   ) * (N[-1]*N[-2])**-1
        else:
            psigk[k] = fft.fftshift(fft.fft2(psi_intrp[k])
                                   ) * (N[-1]*N[-2])**-1
        if k > 0:
            zetag[k-1] = np.real(fft.ifft2(fft.ifftshift(-K2_ * psigk[k]
                                                         * N[-1]*N[-2]))
                                )

    qgpvk = np.zeros_like(psi_intrp, dtype=complex)
    for j in range(len(ky)):
        for i in range(len(kx)):
            Ak = A[:,:,j,i].copy()
            Ak[1:,1:] -= np.eye(N[0])*K2_[j,i]
            qgpvk[:,j,i] = np.dot(Ak, psigk[:,j,i])

    qgpv = np.zeros_like(psi_intrp)
    for k in range(N[0]+1):
        qgpv[k] = np.real(fft.ifft2(fft.ifftshift(qgpvk[k])))*(N[-2]*N[-1])

    return xr.DataArray(zetag, dims=psi.dims, coords=psi.coords), \
           xr.DataArray(qgpv, dims=dim, coords=coord)

def _simodesA(A,dz,K2,nmodes,atop):

    """
    .. math::

     Compute m-th eigenvectors phi_m and eigenvalues mu_m satisfying

        [\frac{d}{dz} S \frac{d}{dz} - K^2] phi_m = -\mu_m^2 phi_m

     with boundary conditions

        (S/H) \frac{d}{dz} phi_m = mu_m^2/atop phi_m  @  z=ztop
        (S/H) \frac{d}{dz} phi_m = 0                  @  z=zbot

     where H = fluid depth (=sum(dz)), S = f^2/N^2 and K is the
     horizontal wavenumber.

    Parameters
    ----------
    A : `numpy.array`
        psi-q matrix without K^2 part, such that:

        A psi -[0, K2*psi(2:nz)] = Q = [b, q(z)],
        b = f^2/(H N^2) \frac{d \psi}{dz} at z=0

    dz : `numpy.array`
        Vector of 'layer thicknesses' summing to H.
        len(dz) = A.shape[0] - 1.

    Returns
    -------
    phi(z,m,K) : `numpy.array`
    lambda(m,K) : `numpy.array`
        \sqrt{mu^2-K^2}

        where 1 < m < nmodes <= length(dz).

    Input parameters atop is nondimensional weight for surface mode.
    """
    # nz = length(A(:,1));
    # H = sum(dz);
    #
    # mu = zeros(nmodes,1);
    # phi = zeros(nz,nmodes);
    # lam = zeros(size(mu));
    #
    # Ak = A - diag([0 K2*ones(1,nz-1)]);
    # B = diag([1; dz/H]);
    # F = -eye(nz); F(1,1) = 1;
    # P =  eye(nz); P(1,1) = atop;
    #
    # FPA = F*P*Ak;
    #
    # [V,D] = eig(FPA);
    #
    # [mutemp,ri] = sort(sqrt(-diag(D)),'ascend');
    #
    # phitemp = V(:,ri(1:nmodes));
    # mu = mutemp(1:nmodes);
    # lambda = sqrt(mu.^2-K2);
    #
    # % Normalize based on (B5)
    #
    # FBA = F*B*Ak;
    #
    # for j=1:nmodes
    #   Nm = transpose(phitemp(:,j))*FBA*phitemp(:,j);
    #   phi(:,j) = phitemp(:,j)/sqrt(Nm);
    #   phi(:,j) = sign(phi(1,j))*phi(:,j);
    # end
    #
    #
    # #  Normalize phi as in old version of simodes paper
    # # for m=1:nmodes
    # #   prod = phi(2:end,m)'*(phi(2:end,m).*dz(2:end)/H)+phi(1,m)'*phi(1,m)/atop;
    # #   phi(:,m) = phi(:,m)/sqrt(prod);
    # #   if (phi(1,m)<0)
    # #     phi(:,m) = -phi(:,m);
    # #   end
    # # end

def SA_modes(ds, ds_grid, q, b, f, N2, zN2, H, dx, dy,
            yname='y_rho', peri=False, nK=5000):
    """
    Derives the decomposition of surface-aware modes.
    """
    if b.ndim != 3:
        raise ValueError("`b` should only have three (spatial) dimensions.")
    N = b.shape
    grid = _grid(ds, peri)
    y = ds_grid[yname]
    dy = grid.interp(grid.diff(y,'Y'),'Y',boundary='fill').mean(dim='xi_rho')
    bm = b.mean(dim='xi_rho')
    bm_y = np.squeeze(bm.values * dy.values**-1)

    f0 = np.mean(f)
    beta = (f.mean(dim='xi_rho')[-1].values
            - f.mean(dim='xi_rho')[0].values
           ) / (y.mean(dim='xi_rho')[-1].values
                - y.mean(dim='xi_rho')[0].values)

    dzr, dzp, zp = _interp_vgrid(N[-3], N[-2], N[-1], z, H)

    b_intrp = np.zeros((N[-3]+1,N[-2],N[-1]))
    N2_intrp = np.zeros((N[-3]+1,N[-2],N[-1]))
    N2_bar = np.zeros((N[-3]+1,N[-2]))
    fN = naiso.interp1d(zN2, N2, fill_value='extrapolate')
    for j in range(N[-2]):
        N2_bar[:,j] = fN(np.mean(zp, axis=-1)[:,j])
        N2_intrp[:,j,i] = fN(zp[:,j,i])
        b_intrp[:,j,i] = _interpolate(z[:,j,i], b[:,j,i],
                                     zp[:,j,i])[::-1]

    Q_y = beta + f0*(bm_y[:,:-1] / N2_bar[np.newaxis,:-1]
                     - bm_y[:,1:] / N2_bar[np.newaxis,1:]
                    ) / np.mean(dzp, axis=-1)[np.newaxis,:]
    Q_ym = np.mean(Q_y, axis=-1)

    q_y = grid.interp(grid.diff(q.mean(dim='xi_rho'),'Y'),
                      'Y',boundary='fill'
                     ).values * dy.values**-1
    q_ym = np.squeeze(np.mean(q_y, axis=-1))

    nmodes = 5
    kx = fft.fftshift(fft.fftfreq(N[-1], dx)) * 2.*np.pi
    ky = fft.fftshift(fft.fftfreq(N[-2], dy)) * 2.*np.pi
    k2v = np.unique(kx[np.newaxis,:]**2
                    + ky[:,np.newaxis]**2)[1:]
    kd2 = (2.*np.pi*f0)**2 / (N2_intrp.mean(axis=-3)*H**2)
    atop = .1*k2v[1]/k2d

    dzs = np.zeros(N[-3]+1,N[-2],N[-1])
    dzs[0] = .5 * dzr[0]
    dzs[1] = .5 * (dzr[0] + dzr[1])
    dzs[2:] = dzp[1:]

    phi = np.zeros((N[-3]+1,nmodes,nK,N[-2],N[-1]))
    lam = np.zeros_like(phi[0])
    s = np.zeros(nK,N[-2],N[-1])
