import os
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, simps, cumtrapz
from scipy.interpolate import SmoothBivariateSpline
import astropy.units as u

######################################################################
## Absolute and Relative Error used in quad
## Integration over theta
## If you change you need to regenerate the library
EPSABS = 1.49e-5
EPSREL = 1.49e-4

class NFWModel(object):
    r"""
    A class that predicts off-centered nfw profiles.
    
    Initializing a class is easy.  You need a cosmology object like those created by astropy,
    since we need to know overdensities.  Once you have one:
    >>>  from offset_nfw import NFWModel
    >>>  nfw_model = NFWModel(cosmology)
    
    If you change the x_range and Nsize you need to new interpolation tables
    To do that, you should run the python script:
    >>>  python eval_parallel.py
    If you want to use tables you generated in another directory, simply pass the directory name:
    >>>  nfw_model = NFWModel(cosmology, dir='nfw_tables')
    
    Parameters
    ----------
    cosmology : astropy.cosmology instance
        A cosmology object that can return distances and densities for computing $\Sigma\Crit$ and
        $\rho_m$ or $\rho_c$.
    dir : str
        The directory where the saved tables should be stored (will be interpreted through
        ``os.path``). [default: './data/']
    rho : str
        Which type of overdensity to use for the halo, `'rho_m'` or `'rho_c'`.  These correspond to
        measuring the overdensity relative to the matter density ($\rho_m$) or the critical density
        ($\rho_c$). [default: 'rho_m']
    delta : float
        The overdensity at which the halo mass is defined. [default: 200]
    x_range : tuple
        The min-max range of x (=r/r_s) for the interpolation table. Precision is not guaranteed for 
        values other than the default.  [default: (0.001, 1000)]
    """
    def __init__(self, cosmology, mydir='./data/', rho='rho_c',
                delta=200, Nsize=5000, x_range=(0.001, 1000)):
        # size to create the integration vector
        self.x_range = x_range
        dx = 2*np.log(x_range[1]/x_range[0])
        self.nsize = Nsize

        if not os.path.exists(mydir):
            raise RuntimeError("Nonexistent save directory passed to NFWModel")
        self.dir = mydir

        if not (hasattr(cosmology, "critical_density") and 
                hasattr(cosmology, "critical_density0") and
                hasattr(cosmology, "Om")):
            raise RuntimeError("Must pass working cosmology object to NFWModel")
        self.cosmology = cosmology

        if not rho in ['rho_c', 'rho_m']:
            raise RuntimeError("Only rho_c and rho_m currently implemented")
        self.rho = rho
        self.delta = delta
        
        # Useful quantity in scaling profiles
        self._rmod = (3./(4.*np.pi)/self.delta)**0.33333333

        self.table_file_root = mydir
        xlow, xhig = x_range
        self.table_fname = os.path.join(self.table_file_root,
                                        'offset_nfw_table_%i_%.0e_%.0e'%(Nsize,xlow,xhig))

        # if sigma:
        self._loadTables()
    
    def nfw_norm(self, M, c, z):
        """ Return the normalization for delta sigma and sigma. """
        c = np.asarray(c)
        z = np.asarray(z)
        deltac=self.delta/3.*c*c*c/(np.log(1.+c)-c/(1.+c))
        rs = self.scale_radius(M, c, z)
        return rs*deltac*self.reference_density(z)

    def scale_radius(self, M, c, z):
        """ Return the scale radius in comoving Mpc. """
        c = np.asarray(c)
        z = np.asarray(z)
        if not isinstance(M, u.Quantity):
            M = (M*u.Msun)
        rs = self._rmod/c*(M/self.reference_density(z))**0.33333333
        return rs.to(u.Mpc**0.99999999)  # to deal with fractional powers

    def reference_density(self, z):
        """Return the reference density for this halo: that is, critical density for rho_c,
        or matter density for rho_m. Physical NOT COMOVING"""
        if self.rho=='rho_c':
            dens = self.cosmology.critical_density(z)
            return dens.to('Msun/Mpc^3')
        else:
            dens = self.cosmology.Om0*self.cosmology.critical_density0
            return (dens*(1.+z)**3).to('Msun/Mpc^3')

    def sigma(self, r, M, c, z, r_mis=0, kernel='gamma'):
        """Return an optionally miscentered NFW sigma from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        kernel: str
            Kernal for convolution. Options: gamma, single or rayleigh.
        
        Returns
        -------
        float or np.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z).value
        x = np.atleast_1d(r/rs)
        x_mis = np.atleast_1d(r_mis/rs)
        norm = self.nfw_norm(M, c, z).value
        
        # for a given kernel it gets the respective interpolation table
        u_profile = self.getProfileFunction(kernel,'sigma')

        zeromask = x_mis==0.
        if np.all(zeromask):
            return_vals = self.unitary_sigma_theory(x)
        else:
            return_vals = u_profile(x, x_mis)
        return_vals = norm*return_vals
        return return_vals

    def deltasigma(self, r, M, c, z, r_mis=0, kernel='gamma'):
        """Return an optionally miscentered NFW sigma from an internal interpolation table.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        r_mis : float
            The distance (in length units) between the reported center of the halo and the actual
            assumed to be in Mpc/h.
            center of the halo.  Whatever definition (comoving/not, etc) you used for your cosmology
            object should be replicated here.  This can be an object with astropy units of length;
            if not it is assumed to be in Mpc/h.  If this is an iterable, all other non-r parameters
            must be either iterables with the same length or floats.
        
        kernel: str
            Kernal for convolution. Options: gamma, single or rayleigh.
        
        Returns
        -------
        float or np.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z).value
        x = np.atleast_1d(r/rs)
        x_mis = np.atleast_1d(r_mis/rs)
        norm = self.nfw_norm(M, c, z).value
        
        # for a given kernel it gets the respective interpolation table
        u_profile = self.getProfileFunction(kernel,'deltasigma')

        zeromask = x_mis==0.
        if np.all(zeromask):
            return_vals = self.unitary_deltasigma_theory(x)
        else:
            return_vals = u_profile(x, x_mis)
        return_vals = norm*return_vals
        return return_vals

    def sigma_theory(self, r, M, c, z):
        """Return an NFW sigma from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        
        Returns
        -------
        float or np.ndarray
            Returns the value of sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r = r*u.Mpc
        x = np.atleast_1d((r/rs).decompose().value)
        norm = self.nfw_norm(M, c, z)
        return_vals = norm*self.unitary_sigma_theory(x)
        return return_vals

    def deltasigma_theory(self, r, M, c, z):
        """Return an NFW delta sigma from theory.
        
        Parameters
        ----------
        r : float or iterable
            The radius or radii (in length, not angular, units) at which to evaluate the function.
            Whatever definition (comoving/not, etc) you used for your cosmology object should be
            replicated here.  This can be an object with astropy units of length; if not it is
            assumed to be in Mpc/h.
        M : float or iterable
            The mass of the halo at the overdensity definition given at class initialization. If
            this is an iterable, all other non-r parameters must be either iterables with the same
            length or floats. This can be an object with astropy units of mass; if not it is assumed
            to be in h Msun.
        c : float or iterable
            The concentration of the halo at the overdensity definition given at class 
            initialization.  If this is an iterable, all other non-r parameters must be either
            iterables with the same length or floats.
        z : float or iterable
            The redshift of the halo.  If this is an iterable, all other non-r parameters must be
            either iterables with the same length or floats.
        
        Returns
        -------
        float or np.ndarray
            Returns the value of delta sigma at the requested parameters. If every parameter was a
            float, this is a float. If only r OR some of the non-r parameters were iterable, this is
            a 1D array with the same length as the iterable parameters.  If both r and another
            parameter were iterable, with the non-r parameter having shape ``(n1, n2, ..., nn)``
            (which of course may be only one item!), then this returns an array of shape
            ``(n1, n2, ..., nn, len(r))``.
        """
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r = r*u.Mpc
        x = np.atleast_1d((r/rs).decompose().value)
        
        norm = self.nfw_norm(M, c, z)
        return_vals = norm*self.unitary_deltaSigma_theory(x)
        return return_vals

    def sigma_to_deltasigma(self, r, sigma):
        """central_value is default 0; central_value = [something floating-point] will be used; 
        central_value = 'interp' will use the innermost value of sigma.  central_value must have same
        units as sigma, if given explicitly."""
        if hasattr(r, 'unit'):
            r_unit = r.unit
            r = r.value
        else:
            r_unit = 1
        if hasattr(sigma, 'unit'):
            sigma_unit = sigma.unit
            sigma = sigma.value
        else:
            sigma_unit = 1
        sigma_r = 2*np.pi*r*sigma
        sum_sigma = cumtrapz(sigma_r, r, initial=0)*sigma_unit*r_unit**2
        sum_area = np.pi*(r**2-r[0]**2)*r_unit**2
        deltasigma = np.zeros_like(sum_sigma)
        # Linearly interpolate central value, which is nan due to sum_area==0
        if len(deltasigma.shape)==1:
            deltasigma[1:] = sum_sigma[1:]/sum_area[1:] - sigma[1:]*sigma_unit        
            deltasigma[0] = 2*deltasigma[1]-deltasigma[2]
        else:
            deltasigma[:,1:] = sum_sigma[:,1:]/sum_area[1:] - sigma[:,1:]*sigma_unit        
            deltasigma[:,0] = 2*deltasigma[:,1]-deltasigma[:,2]
        return deltasigma


    def unitary_sigma_theory(self,x):
        """Unitary NFW \Sigma Profile

        Brainerd and Wright 1999 
        """
        return f_nfw(x)

    def unitary_deltasigma_theory(self,x):
        """Unitary NFW \Sigma Profile
        
        Brainerd and Wright 1999 
        """
        return g_nfw(x)

    def _loadTables(self):
        self._buildTables()
        is_miscentered = os.path.isfile(self.table_fname+'_miscentered.npz')
        is_gamma = os.path.isfile(self.table_fname+'_gamma.npz')
        
        if is_miscentered:
            lib = np.load(self.table_fname+'_miscentered.npz')
            self._miscentered_sigma = lib['sigma_mis']
            self._miscentered_sigma_err = lib['sigma_mis_err']
            self._setupMiscenteredSigma()
            self._buildMiscenteredDeltaSigma()
            self._setupMiscenteredDeltaSigma()

        if is_gamma:
            lib = np.load(self.table_fname+'_gamma.npz')
            self._gamma_sigma = lib['sigma_gamma']
            self.table_tau = lib['tau']
            self._setupGammaSigma()
            self._buildGammaDeltaSigma()
            self._setupGammaDeltaSigma()

    def _buildTables(self):
        if not os.path.isfile(self.table_fname+'_miscentered.npz'):
            print('Grid Table Not Found %s'%(self.table_fname+'_miscentered.npz'))
            print('Please generate a new table for new x_range and Nsize')

            self.table_x = np.logspace(np.log10(self.x_range[0]), np.log10(self.x_range[1]), max(2,self.nsize))
            #self.table_x = thetaFunctionSampling(self.table_x2,int(self.nsize))
            self.x_min = np.min(self.table_x)
            self.x_max = np.max(self.table_x)
            self.dx = np.log(self.table_x[1]/self.table_x[0])*self.table_x
        else:
            lib = np.load(self.table_fname+'_miscentered.npz')
            self.table_x = lib['x']
            self.table_xmis = lib['xmis']
            # lib = np.load(self.table_fname+'_gamma.npz')
            # self.table_tau = lib['tau']


    def _setupMiscenteredSigma(self):
        clipx = self.table_xmis#np.clip(self.table_xmis, self.table_x[0], self.table_x[-1])
        iXmis = lambda xmis: int(np.interp(xmis, clipx, np.arange(clipx.size)))
        self.u_miscentered_sigma = lambda x, xmis: interp1d(np.log(self.table_x),self._miscentered_sigma[iXmis(xmis)],fill_value='extrapolate')(np.log(x))

    def _setupGammaSigma(self):
        clipx = self.table_tau#np.clip(self.table_tau, self.table_x[0], self.table_x[-1])
        iTau = lambda tau: int(np.interp(tau, clipx, np.arange(clipx.size)))
        self.u_gamma_sigma = lambda x, tau: interp1d(np.log(self.table_x),self._gamma_sigma[iTau(tau)],fill_value='extrapolate')(np.log(x))
        pass

    def _setupMiscenteredDeltaSigma(self, save=True):
        clipx = np.clip(self.table_xmis, self.table_x[0], self.table_x[-1])
        iXmis = lambda xmis: int(np.interp(xmis, clipx, np.arange(clipx.size)))
        self.u_miscentered_deltasigma = lambda x, xmis: interp1d(np.log(self.table_x),self._miscentered_deltasigma[iXmis(xmis)],fill_value=0.,bounds_error=False)(np.log(x))
    
    def _setupGammaDeltaSigma(self):
        clipx = np.clip(self.table_tau, self.table_x[0], self.table_x[-1])
        iTau = lambda tau: int(np.interp(tau, clipx, np.arange(clipx.size)))
        self.u_gamma_deltasigma = lambda x, tau: interp1d(np.log(self.table_x),self._gamma_deltasigma[iTau(tau)],fill_value=0.,bounds_error=False)(np.log(x))
        pass

    def probGamma(self, R, Rs):
        return np.exp(-R/Rs)*R/Rs**2

    def _buildMiscenteredDeltaSigma(self, save=True):
        self._miscentered_deltasigma = np.array([self.sigma_to_deltasigma(self.table_x, ms) for ms in self._miscentered_sigma])
    
    def _buildGammaDeltaSigma(self, save=True):
        self._gamma_deltasigma = np.array([self.sigma_to_deltasigma(self.table_x, ms) for ms in self._gamma_sigma])

    def getProfileFunction(self, kernel, ptype):
        """ Given a kernel and profile type string return the respective interpolation function

        E.g. self.getProfileFunction('gamma','sigma') -> self.u_gamma_sigma
        Basically, gives the gamma sigma interpolation function
        """
        sigma_profile_dict = {'single': 'miscentered','gamma':'gamma','rayleigh':'rayleigh'}
        var_name = 'u_%s_%s'%(sigma_profile_dict[kernel],ptype)
        return getattr(self, var_name)
    
    def getProfileGrid(self, kernel, ptype):
        """ Given a kernel and profile type string return the respective interpolation function

        E.g. self.getProfileGrid('gamma','sigma') -> self._gamma_sigma_table 
        Basically, gives the gamma sigma interpolation function
        """
        sigma_profile_dict = {'single': 'miscentered','gamma':'gamma','rayleigh':'rayleigh'}
        var_name = '_%s_%s'%(sigma_profile_dict[kernel],ptype)
        return getattr(self, var_name)

    
    def generate_miscentered_sigma_parallel(self,x,nsize=4000):
        """generate_miscentered_sigma_parallel 

        Uses a rectangular grid instead of a square one.

        Args:
            x (array): the R/R_s radii array used
        """
        xmis = self.table_x
        res = np.zeros((xmis.size, x.size))
        error = np.zeros(xmis.size)
        
        for i,rmis in enumerate(xmis):
            ratio = x/rmis

            # close to the peak
            wsel, = np.where( (ratio<=101.) & (ratio>=1/101.) )
            xint = thetaFunctionSampling(x[wsel],rmis,nsize)
            _integral, err = sigmaMis(xint,rmis)

            isnan = np.isnan(_integral)
            intF = interp1d(np.log(xint[~isnan]), np.log(_integral[~isnan]))

            # at x100 Rmis we can safely assume the asymptotic solution
            yhig = f_nfw(x)
            ylow = f_nfw(rmis)
            integral = np.where(ratio>=100.,yhig,ylow)
            wmid, = np.where( (ratio<=100.) & (ratio>=1/100.) )
            integral[wmid] = np.exp(intF(np.log(x[wmid])))

            res[i] = integral
            error[i] = np.nanmean(err/_integral)

        self.table_xmis = x
        self._miscentered_sigma = res
        self._miscentered_sigma_err = error
        self.miscentered_dict = self._todict('miscentered','sigma')
        pass
    
    def generate_gamma_sigma_parallel(self, tauvec, nsize=int(4./EPSREL)):
        """generate_gamma_sigma_parallel 

        Integrate over the xmis of signle mis-centered cluster
        For a Gamma distribution P(xmis,tau)

        Args:
            tauvec (array): the gamma distribution tau value
        """
        _single = self._miscentered_sigma.copy()
        nsize = self.table_x.size
        x = self.table_xmis

        res = np.zeros((tauvec.size,nsize))
        for i,tau in enumerate(tauvec):
            res[i] = np.trapz(self.probGamma(x, tau)[:,np.newaxis]*_single,x=x,axis=0)

        self.table_tau = tauvec
        self._gamma_sigma = res
        self.gamma_dict = self._todict('gamma','sigma')
        pass

    def _todict(self, kernel, ptype):
        if kernel=='miscentered':
            out = {'x1': self.table_xmis, 'x2': self.table_x,
                   'vec': getattr(self, '_%s_%s'%('miscentered',ptype)),
                   'vec_err': getattr(self, '_%s_%s_err'%('miscentered',ptype))
                   }
            return out

        if kernel=='gamma':
            out = {'x1': self.table_x, 'x2': self.table_tau,
                   'vec': getattr(self, '_%s_%s'%('gamma',ptype)),
                   'vec_err': np.full((self.table_x.size,),np.nan)}
            return out

    def generate_miscentered_deltasigma_parallel(self,x,nsize=int(4./EPSREL)):
        """generate_miscentered_deltasigma_parallel 

        Uses a rectangular grid instead of a square one.

        Args:
            x (array): the R/R_s radii array used
        """
        xmis = self.table_x
        res = np.zeros((xmis.size, x.size))
        error = np.zeros(xmis.size)
        
        for i,rmis in enumerate(xmis):
            ratio = x/rmis

            # close to the peak
            wsel, = np.where( (ratio<=101.) & (ratio>=1/101.) )
            xint = thetaFunctionSampling(x[wsel],rmis,nsize)
            _integral, err = deltaSigmaMis(xint, rmis)

            isnan = np.isnan(_integral)
            intF = interp1d(np.log(xint[~isnan]), np.log(_integral[~isnan]))

            # at x100 Rmis we can safely assume the asymptotic solution
            yhig = g_nfw(x)
            ylow = g_nfw(rmis)
            integral = np.where(ratio>=100.,yhig,ylow)
            wmid, = np.where( (ratio<=100.) & (ratio>=1/100.) )
            integral[wmid] = np.exp(intF(np.log(x[wmid])))

            res[i] = integral
            error[i] = np.nanmean(err/_integral)

        self.table_xmis = x
        self._miscentered_deltasigma = res
        self._miscentered_deltasigma_err = error
        self.miscentered_delta_dict = self._todict('miscentered','deltasigma')
        pass
    
    def generate_gamma_deltasigma_parallel(self, tauvec, nsize=int(4./EPSREL)):
        """generate_gamma_deltasigma_parallel 

        Integrate over the xmis of signle mis-centered cluster
        For a Gamma distribution P(xmis,tau)

        Args:
            tauvec (array): the gamma distribution tau value
        """
        _single = self._miscentered_deltasigma.copy()
        nsize = self.table_x.size
        x = self.table_xmis

        res = np.zeros((tauvec.size,nsize))
        for i,tau in enumerate(tauvec):
            res[i] = np.trapz(self.probGamma(x, tau)[:,np.newaxis]*_single,x=x,axis=0)

        self.table_tau = tauvec
        self._gamma_deltasigma = res
        self.gamma_delta_dict = self._todict('gamma','deltasigma')
        pass
    
    def to_tsv(self, profile_type='sigma',kernel='single'):
        ### file name
        froot = self.table_fname
        fname1 = froot+'_log_%s_%s.txt'%(profile_type, kernel)
        fname2 = froot+'_%s_logx.txt'%(kernel)
        fname3 = froot+'_%s_logxmis.txt'%(kernel)

        # vectors 
        vecgrid = self.getProfileGrid(kernel,profile_type)
        xvec = self.table_x
        yvec = {'single':self.table_xmis, 'gamma':self.table_tau}

        write_vec(fname2, xvec, np.log)
        write_vec(fname3, yvec[kernel], np.log)
        write_grid(fname1, vecgrid, np.log)

def write_vec(fname, vec, func=np.array):
    # Save x data
    xvec = func(vec)
    with open(fname, 'w') as x_file:
        x_file.write('\n'.join(str(x) for x in xvec))

def write_grid(fname, gridvec, func=np.array):
    # Save x data
    zvec = func(gridvec)
    with open(fname, 'w') as z_file:
        for row in zvec:
            z_file.write(' '.join(str(val) for val in row))
            z_file.write('\n')

## Auxialiary Functions
def funcQuadVal(r,rm,func,eps=EPSREL/10.):
    """Integrates a function (Sigma, DSigma) over theta for r, rm
    """
    integral1, err1 = quad(func,0.,(1-eps)*np.pi,args=(r,rm),epsrel=EPSREL,epsabs=EPSABS)
    integral2, err2 = quad(func,(1+eps)*np.pi,2*np.pi,args=(r,rm),epsrel=EPSREL,epsabs=EPSABS)
    return (integral1+integral2)/2./np.pi, (err1+err2)/2.

def _sigmaMisQuad(t,r,rs):
    return f_nfw(np.sqrt(r**2+rs**2+2*r*rs*np.cos(t)))

def _deltaSigmaMisQuad(t,r,rs):
    return g_nfw(np.sqrt(r**2+rs**2+2*r*rs*np.cos(t)))

def deltaSigmaMis(rvec,rmis,eps=1e-5):
    """ NFW Off-Centered Profile (Quad Integration)
    rvec : array
        the radius value r/r_s
    rmis : float
        the off-centering radii value
    """
    res = np.zeros_like(rvec)
    error = np.zeros_like(rvec)
    for i in range(rvec.size):
        integral, err = funcQuadVal(rvec[i],rmis, _deltaSigmaMisQuad)
        res[i] = integral
        error[i] = err
    return res, error

def sigmaMis(rvec,rmis,eps=1e-5):
    """ NFW Off-Centered Profile (Quad Integration)
    rvec : array
        the radius value r/r_s
    rmis : float
        the off-centering radii value
    """
    res = np.zeros_like(rvec)
    error = np.zeros_like(rvec)
    for i in range(rvec.size):
        integral, err = funcQuadVal(rvec[i],rmis, _sigmaMisQuad)
        res[i] = integral
        error[i] = err
    return res, error

### Compute \Sigma Analyticaly
def f_greater_than(x):
    res = 1. / (x**2 - 1.0)
    res *=(1- 2.0 / np.sqrt(x**2 - 1.0) * np.arctan(np.sqrt((x - 1.0) / (1.0 + x))))
    return res

def f_less_than(x):
    res = 1. / (x**2 - 1.0)
    res *=(1- 2.0 / np.sqrt(1.0 - x**2) * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x))))
    return res
        
def f_nfw(x, eps=1e-9):
    """
    Analytical normalized Sigma profile
    Wright & Brained 2000

    Args:
        x (array): Rp/Rs where Rs is the scale radius, Rs = R200/c
    """
    if (isinstance(x,float))or(isinstance(x,int)): 
        x = np.array([x])
        
    res = 1/3.*np.ones_like(x)

    ix = np.where(x <= 1-eps)[0]
    res[ix] = f_less_than(x[ix])
    
    ix = np.where(x>=1+eps)[0]
    res[ix] = f_greater_than(x[ix])
    return res

### Compute \Delta \Sigma Analyticaly
def g_less_than(x,core=4e-3):
    # below core the solution fails numereically
    x = np.where(x<core, core, x)
    term1 = 8.0*np.arctanh(np.sqrt((1.0-x)/(1.0+x)))/(x**2*np.sqrt(1.0-x**2))
    term2 = 4.0/x**2 * np.log(x/2.0)
    term3 = -2.0/(x**2-1.0)
    term4 = 4.0*np.arctanh(np.sqrt((1.0-x)/(1.0+x)))/((x**2-1.0)*np.sqrt(1.0-x**2))
    return term1 + term2 + term3 + term4

def g_greater_than(x, xmax=1e9):
    # above xmax the function achieves machine precision
    x = np.where(x>xmax, xmax, x)
    term1 = 8.0*np.arctan(np.sqrt((x-1.0)/(1.0+x)))/(x**2*np.sqrt(x**2-1.0))
    term2 = 4.0/x**2 * np.log(x/2.0)
    term3 = -2.0/(x**2-1.0)
    term4 = 4.0*np.arctan(np.sqrt((x-1.0)/(1.0+x)))/((x**2-1.0)**(3.0/2.0))
    return term1 + term2 + term3 + term4

def g_nfw(x, eps=1e-9):
    """gNFW Eqn 15 and 16 Wright & Brained 2000

    Analytical normalized shear/deltaSigma profile

    Args:
        x (array): Rp/Rs where Rs is the scale radius, Rs = R200/c
    """
    if (isinstance(x,float))or(isinstance(x,int)): 
        x = np.array([x])
            
    res = np.zeros_like(x)
    ix = np.where(np.abs(x-1) <= eps)[0]
    res[ix] = 10./3. + 4*np.log(1/2.) 
    
    ix = np.where(x <= 1-eps)[0]
    res[ix] = g_less_than(x[ix])
    
    ix = np.where(x>=1+eps)[0]
    res[ix] = g_greater_than(x[ix])
    return res

def get_adaptive_bin(xvec,fvec,Nsize=100):
    fcsum = np.cumsum(fvec,axis=0)
    cdf = fcsum/np.max(fcsum)
    x_adp_bin = np.interp(np.linspace(0.,1.,Nsize),cdf,np.log(xvec))
    return np.exp(x_adp_bin)

def asymmetric_laplace_bparam_pdf(x, sigma, beta):
    mu = 0.
    B0 = np.sqrt(1 + beta**2 / 4)
    B_plus = B0 + beta / 2
    B_minus = B0 - beta / 2

    pdf = np.zeros_like(x)
    pdfl = (1 / (2 * sigma * B0)) * np.exp((x - mu) / (sigma * B_minus))
    pdfh = (1 / (2 * sigma * B0)) * np.exp(-(x - mu) / (sigma * B_plus))
    
    return np.where(x<=0,pdfl,pdfh)

func = lambda x,A,lbd,kappa: A*asymmetric_laplace_bparam_pdf(x,lbd,kappa)
def thetaFunctionSampling(x,xm0,Nsize,pars=[0.67140937, 0.46993814, 0.15819616]):
    thetaEff = func(np.log10(x/xm0),*pars)
    xgrid = get_adaptive_bin(x,thetaEff,Nsize)
    return xgrid
