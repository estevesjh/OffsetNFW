import os
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad, simps
from scipy.interpolate import SmoothBivariateSpline
import astropy.units as u

import time
t0 = time.time()
def tupdate(s):
    t = time.time()-t0
    if t<60:
        print("%.1f seconds have elapsed at point %s"%(t,s))
    elif t<60*60:
        print("%.2f minutes have elapsed at point %s"%(t/60.,s))
    else:
        print("%i hours and %.2f minutes have elapsed %s"%(int(t/3600), t/60., s))
    pass

class NFWModel(object):
    r"""
    A class that generates offset (miscentered) NFW halo profiles.  The basic purpose of this class
    is to generate internal interpolation tables for fast computation of the common NFW lensing 
    quantities, but it includes direct computation of the non-miscentered versions for completeness.
    
    Initializing a class is easy.  You need a cosmology object like those created by astropy,
    since we need to know overdensities.  Once you have one:
    >>>  from offset_nfw import NFWModel
    >>>  nfw_model = NFWModel(cosmology)
    
    However, this won't have any internal interpolation tables (unless you've already created them
    in the directory you're working in).  To do that, you should use the class method 
    :ref:`generate``:
    >>>  nfw_model.generate()
    If you want to use tables you generated in another directory, simply pass the directory name:
    >>>  nfw_model = NFWModel(cosmology, dir='nfw_tables')
    
    Parameters
    ----------
    cosmology : astropy.cosmology instance
        A cosmology object that can return distances and densities for computing $\Sigma\Crit$ and
        $\rho_m$ or $\rho_c$.
    dir : str
        The directory where the saved tables should be stored (will be interpreted through
        ``os.path``). [default: '.']
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
    def __init__(self, cosmology, mydir='.', rho='rho_c', nsize=100,
                delta=200, Nsize=100, x_range=(0.001, 1000),
                sigma=True, gamma=True):
        # size to create the integration vector
        self.x_range = x_range
        dx = 2*np.log(x_range[1]/x_range[0])
        self.nsize = nsize
        print('Nsize :', self.nsize)

        if not os.path.exists(mydir):
            raise RuntimeError("Nonexistent save directory passed to NFWModel")
        self.dir = mydir

        if not rho in ['rho_c', 'rho_m']:
            raise RuntimeError("Only rho_c and rho_m currently implemented")
        self.rho = rho
        self.delta = delta
        # Useful quantity in scaling profiles
        self._rmod = (3./(4.*np.pi)/self.delta)**0.33333333

        self.table_file_root = os.path.join(self.dir, '.offset_nfw_table')
        self.table_file_root = self.table_file_root+'_nsize_%i_xrange_%.4f_%.1f'%(
                                                        self.nsize, x_range[0], x_range[1])

        self._loadTables(sigma, gamma)
        self.do_sigma = sigma
        self.do_gamma = gamma
    
    def nfw_norm(self, M, c, z):
        """ Return the normalization for delta sigma and sigma. """
        c = np.asarray(c)
        z = np.asarray(z)
        if not isinstance(M, u.Quantity):
            M = (np.asarray(M)*u.Msun).to(u.g)
        deltac=self.delta/3.*c*c*c/(np.log(1.+c)-c/(1.+c))
        rs = self.scale_radius(M, c, z)
        return rs*deltac*self.reference_density(z)
    
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
        rs = self.scale_radius(M, c, z)
        if not isinstance(r, u.Quantity):
            r = r*u.Mpc
        x = np.atleast_1d((r/rs).decompose().value)
        if isinstance(r_mis, float):
            if not isinstance(r_mis, u.Quantity):
                r_mis = r_mis*u.Mpc
        else:
            raise RuntimeError("r_mis should be a float number")

        x_mis = np.atleast_1d((r_mis/rs).decompose().value)
        norm = self.nfw_norm(M, c, z)
        
        # for a given kernel it gets the respective interpolation table
        sigma_profile = self.getProfileVariable(kernel,'sigma')

        zeromask = x_mis==0.
        if np.all(zeromask):
            return_vals = self.unitary_sigma_theory(x)
        else:
            clipx = np.clip(x_mis, self.table_x[0], None)
            return_vals = sigma_profile(x, clipx)
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
        float or numpy.ndarray
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
        float or numpy.ndarray
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
        sum_sigma = scipy.integrate.cumtrapz(sigma_r, r, initial=0)*sigma_unit*r_unit**2
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

    def unitary_deltaSigma_theory(self,x):
        """Unitary NFW \Sigma Profile
        
        Brainerd and Wright 1999 
        """
        return g_nfw(x)

    def _buildTables(self):
        self.table_x = np.logspace(np.log10(self.x_range[0]), np.log10(self.x_range[1]), max(2,self.nsize))
        #self.table_x = thetaFunctionSampling(self.table_x2,int(self.nsize))
        self.x_min = np.min(self.table_x)
        self.x_max = np.max(self.table_x)
        self.dx = np.log(self.table_x[1]/self.table_x[0])*self.table_x

    def _buildMiscenteredSigma(self, save=True, force=False):
        if (not force) and os.path.isfile(self.table_file_root+'_miscentered_sigma.npy'):
            self._miscentered_sigma = np.load(self.table_file_root+'_miscentered_sigma.npy')
        else:
            self.generate_miscentered_sigma(save)
            #self._miscentered_sigma = np.load(self.table_file_root+'_miscentered_sigma.npz')
        self._setupMiscenteredSigma()
    
    def _buildGammaSigma(self, save=True, force=False):
        is_file1 = os.path.isfile(self.table_file_root+'_gamma_sigma.npy')

        if (not force) and (is_file1):
            self._gamma_sigma = np.load(self.table_file_root+'_gamma_sigma.npy')
        else:
            self.generate_gamma_sigma(save)
            #self._gamma_sigma = np.load(self.table_file_root+'_gamma_sigma.npz')
        self._setupGammaSigma()

    def _buildMiscenteredDeltaSigma(self, save=True):
        self._miscentered_deltasigma = np.array([self.sigma_to_deltasigma(self.table_x, ms) for ms in self._miscentered_sigma])
        if save:
            np.save(self.table_file_root+'_miscentered_deltasigma.npy', self._miscentered_deltasigma)
    
    def _buildGammaDeltaSigma(self, save=True):
        self._gamma_deltasigma = np.array([self.sigma_to_deltasigma(self.table_x, ms) for ms in self._gamma_sigma])
        if save:
            np.save(self.table_file_root+'_gamma_deltasigma.npy', self._miscentered_deltasigma)

    def generate_miscentered_sigma(self, save=True):
        res = np.zeros((self.table_x.size,self.table_x.size))
        error = np.zeros(self.table_x.size)
        
        x = self.table_x
        for i,rmis in enumerate(x):
            ratio = x/rmis

            # close to the peak
            wmid, = np.where( (ratio<=100.) & (ratio>=1/100.) )
            xint = thetaFunctionSampling(x[wmid],rmis,int(self.nsize))
            integral, err = sigmaMis(xint,rmis)
            intF = interp1d(np.log(xint), np.log(integral), fill_value='extrapolate')

            # at x100 Rmis we can safely assume the asymptotic solution
            yhig = f_nfw(self.table_x)
            ylow = f_nfw(rmis)
            integral = np.where(ratio>=100.,yhig,ylow)
            integral[wmid] = np.exp(intF(np.log(x[wmid])))

            res[i] = integral
            error[i] = np.nanmean(err)

        self._miscentered_sigma = res
        if save:
            np.save(self.table_file_root+'_miscentered_sigma.npy', self._miscentered_sigma)
            np.save(self.table_file_root+'_miscentered_sigma_error.npy', error)
        pass

    def _setupMiscenteredSigma(self):
        # TODO: Implement 2d interpolation
        pass
        # x2, x2 = np.meshgrid(self.table_x,self.table_x)
        # kx = ky = 3
        # xflat = np.log(x2.flatten())
        # yflat = xflat
        # zflat = np.log(self._miscentered_sigma.flatten())

        # bad = np.isnan(zflat)|np.isinf(zflat)
        # fit = SmoothBivariateSpline( xflat[~bad], yflat[~bad], zflat[~bad], kx=kx, ky=ky)
        # self._miscentered_sigma_table = lambda x,xm: np.exp(fit(np.log(x),np.log(xm)))

    def _setupGammaSigma(self):
        # TODO: Implement 2d interpolation
        pass
        # self._gamma_sigma_table = interp2d(
        #     np.log(self.table_x), np.log(self.table_x), self._gamma_sigma, kind='cubic')

    def _setupMiscenteredDeltaSigma(self):
        # TODO: Implement 2d interpolation
        pass
    
    def _setupGammaDeltaSigma(self):
        # TODO: Implement 2d interpolation
        pass
        # self._miscentered_deltasigma_table = scipy.interpolate.RegularGridInterpolator(
        #     (numpy.log(self.table_x), numpy.log(self.table_x)), self._miscentered_deltasigma)


    def probGamma(self, R, Rs):
        return np.exp(-R/Rs)*R/Rs**2

    def generate_gamma_sigma(self, save=True):
        # we restrict the  xmisc range
        # a factor is x10 smaller
        # at the border we cannot integrate over 
        # the P(R,Rmis)
        res = np.zeros((self.nsize,self.nsize))
        x = self.table_x
        
        try:
            _res = self._miscentered_sigma.copy()
        except:
            raise RuntimeError("Nonexistent mis-centered sigma")

        self._gamma_sigma = np.array([simps(self.probGamma(x, xm)[:,np.newaxis]*_res,x=x,axis=0) for xm in x])
        if save:
            np.save(self.table_file_root+'_gamma_sigma.npy', self._gamma_sigma)
        pass
        
    def generate(self, sigma=True, gamma=False, save=True,
                       force=False):
        """
        Generate internal interpolation tables using the settings specified when the NFWModel
        instance was created.  Note that this method does **not** check for existing tables before
        writing over them.
        
        Parameters
        ----------
        sigma : bool
            Generate tables for computing $\Sigma$ and related quantities ($\kappa$). 
            [default: True]
        gamma : bool
            Generate tables for Gamma miscentering distribution. 
            [default: True]
        """
        if sigma:
            tupdate("before tables")
            self._buildTables()
            tupdate("before sigle miscentering")
            self._buildMiscenteredSigma(save, force=force)
        if gamma:
            tupdate("before gamma")
            self._buildGammaSigma(save=save, force=force)
        tupdate("finished")

    def _loadTables(self, sigma=True, gamma=True):
        self._buildTables()
        
        if not os.path.isfile(self.table_file_root+'_miscentered_sigma.npy'):
            print("Nonexistent NFW tables, please run nfw_model.generate()")

        if sigma:
            try:
                if not hasattr(self, '_miscentered_sigma'):
                    self._miscentered_sigma = np.load(self.table_file_root+'_miscentered_sigma.npy')
                self._setupMiscenteredSigma()
            except IOError:
                pass
            try:
                if not hasattr(self, '_gamma_sigma'):
                    self._gamma_sigma = np.load(self.table_file_root+'_gamma_sigma.npy')
                self._setupGammaSigma()
            except IOError:
                pass

    def getProfileVariable(self, kernel, ptype):
        """ Given a kernel and profile type string return the respective interpolation function

        E.g. self.getProfileVariable('gamma','sigma') -> self._gamma_sigma_table 
        Basically, gives the gamma sigma interpolation function
        """
        sigma_profile_dict = {'single': '_miscentered','gamma':'_gamma','rayleigh':'_rayleigh'}
        var_name = sigma_profile_dict[kernel]+'_%s_table'%(ptype)
        return getattr(self, var_name)
import scipy
## Auxialiary Functions
def funcQuadVal(r,rm,func,eps=1e-5):
    """Integrates a function (Sigma, DSigma) over theta for r, rm
    """
    # integral, err = quad(func,0.,2.*np.pi,args=(r,rm),epsrel=1.49e-3,epsabs=1e-3)
    # return integral/2./np.pi

    integral1, err1 = quad(func,0.,(1-eps)*np.pi,args=(r,rm),epsrel=1.49e-5,epsabs=1e-5)
    integral2, err2 = quad(func,(1+eps)*np.pi,2*np.pi,args=(r,rm),epsrel=1.49e-5,epsabs=1e-5)
    return (integral1+integral2)/2./np.pi, (err1+err2)/2.

def _sigmaMisQuad(t,r,rs):
    return f_nfw(np.sqrt(r**2+rs**2+2*r*rs*np.cos(t)))

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
    res = 1/3.*np.ones_like(x)
    res = np.where(x<1.,f_less_than(x),res)
    res = np.where(x>1.,f_greater_than(x),res)
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
