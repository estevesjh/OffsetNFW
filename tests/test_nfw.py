import numpy
import numpy.testing
import astropy.cosmology
import astropy.units as u
try:
    import offset_nfw
except ImportError:
    import sys
    sys.path.append('..')
    import offset_nfw
    

# A "cosmology" object that passes initialization tests.
class fake_cosmo(object):
    critical_density0 = 1
    Om0 = 1
    def critical_density(self, x):
        return 1
    def angular_diameter_distance(self, x):
        return 1
    def angular_diameter_distance_z1z2(self, x, y):
        return 1
    def Om(self, x):
        return 1

m_c_z_test_list = [(1E14, 4, 0.2), (1E13, 4, 0.2), (1E15, 4, 0.2),
                   (1E14, 2, 0.2), (1E14, 6, 0.2), 
                   (1E14, 4, 0.05), (1E14, 4, 0.5), (1E14, 4, 4)]
cosmo = astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3)
        

def test_object_creation():
    # Need a cosmology object
    numpy.testing.assert_raises(TypeError, offset_nfw.NFWModel)
    # Need a REAL cosmology object
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, None) 
    cosmology_obj = fake_cosmo()
    offset_nfw.NFWModel(cosmology_obj)  # This should pass
    # Existing directory
    offset_nfw.NFWModel(cosmology_obj, dir='.')
    # Non-existing directory
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, dir='_random_dir')
    # Wrong rho type
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, rho='rho_dm')
    # Nonsensical delta
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, delta=-200)
    # Non-working ranges
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=3)
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=(3,4,5))
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, x_range=('a', 'b'))
    # Should work
    offset_nfw.NFWModel(cosmology_obj, x_range=[3,4])
    # Non-working ranges
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, miscentering_range=3)
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, miscentering_range=(3,4,5))
    numpy.testing.assert_raises(RuntimeError, offset_nfw.NFWModel, cosmology_obj, miscentering_range=('a', 'b'))
    # Should work
    offset_nfw.NFWModel(cosmology_obj, miscentering_range=[3,4])
    
    obj = offset_nfw.NFWModel(cosmology_obj, '.', 'rho_m', delta=150, precision=0.02, x_range=(0.1,2), 
                       miscentering_range=(0.1,2), comoving=False)
    numpy.testing.assert_equal(obj.cosmology, cosmology_obj)
    numpy.testing.assert_equal(obj.dir, '.')
    numpy.testing.assert_equal(obj.rho, 'rho_m')
    numpy.testing.assert_equal(obj.delta, 150)
    numpy.testing.assert_equal(obj.precision, 0.02)
    numpy.testing.assert_equal(obj.x_range, (0.1,2))
    numpy.testing.assert_equal(obj.miscentering_range, (0.1,2))
    numpy.testing.assert_equal(obj.comoving, False)
    
    # Should work
    offset_nfw.NFWModel(astropy.cosmology.FlatLambdaCDM(H0=100, Om0=0.3))

def test_form_iterables():
    """ Test the piece that makes iterables out of input parameters """
    nfw_obj = offset_nfw.NFWModel(fake_cosmo())
    
    len_rad = 10
    len_many = 4
    radial_bins = numpy.linspace(0.1, 1., num=len_rad)
    single = 1.
    many = numpy.linspace(0,3,num=len_many)

    # Test some singles
    res = nfw_obj._form_iterables(radial_bins, single)
    numpy.testing.assert_equal(res, (radial_bins, single))
    res = nfw_obj._form_iterables(radial_bins, single, single)
    numpy.testing.assert_equal(res, (radial_bins, single, single))

    # Test a single multi-point value
    res = nfw_obj._form_iterables(radial_bins, many)
    numpy.testing.assert_equal(res[0], numpy.array([radial_bins]*len_many))
    numpy.testing.assert_equal(res[1], numpy.array([many]*len_rad).T)
    
    # Test a mix of multi-point and single-point objects
    res = nfw_obj._form_iterables(radial_bins, many, single)
    numpy.testing.assert_equal(res[0], numpy.array([radial_bins]*len_many))
    numpy.testing.assert_equal(res[1], numpy.array([many]*len_rad).T)
    numpy.testing.assert_equal(res[2], numpy.full((len(many), len(radial_bins)), single))
    res = nfw_obj._form_iterables(radial_bins, many, single, single)
    numpy.testing.assert_equal(res[0], numpy.array([radial_bins]*len_many))
    numpy.testing.assert_equal(res[1], numpy.array([many]*len_rad).T)
    numpy.testing.assert_equal(res[2], numpy.full((len(many), len(radial_bins)), single))
    numpy.testing.assert_equal(res[3], numpy.full((len(many), len(radial_bins)), single))
    res = nfw_obj._form_iterables(radial_bins, many, many, single)
    numpy.testing.assert_equal(res[0], numpy.array([radial_bins]*len_many))
    numpy.testing.assert_equal(res[1], numpy.array([many]*len_rad).T)
    numpy.testing.assert_equal(res[2], numpy.array([many]*len_rad).T)
    numpy.testing.assert_equal(res[3], numpy.full((len(many), len(radial_bins)), single))
    
    # Make sure things fail for multi-point objects of different lengths
    numpy.testing.assert_raises(RuntimeError, nfw_obj._form_iterables, radial_bins, many, [1,2], 1)
    
def test_scale_radii():
    """ Test scale radius measurement. """
    # Test against some precomputed values
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    numpy.testing.assert_equal(nfw_1.scale_radius(1E14, 4, 0.2), 0.250927041991*u.Mpc)
    numpy.testing.assert_equal(nfw_1.scale_radius(1E15, 3, 0.2), 0.720807898577*u.Mpc)
    numpy.testing.assert_equal(nfw_1.scale_radius(1E13, 5, 0.2), 0.0931760124926*u.Mpc)
    numpy.testing.assert_equal(nfw_1.scale_radius(1E14, 4, 0.1), 0.250927041991*u.Mpc)
    numpy.testing.assert_equal(nfw_1.scale_radius(1E14, 4.5, 0.3), 0.213527269104*u.Mpc)
    nfw_2 = offset_nfw.NFWModel(cosmo, delta=150, rho='rho_c')
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4, 0.2), 0.22798234765*u.Mpc)
    numpy.testing.assert_equal(nfw_2.scale_radius(1E15, 3, 0.2), 0.65489743799*u.Mpc)
    numpy.testing.assert_equal(nfw_2.scale_radius(1E13, 5, 0.2), 0.0846560255292*u.Mpc)
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4, 0.1), 0.22798234765*u.Mpc)
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4.5, 0.3), 0.19400239891*u.Mpc)
    nfw_3 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_m')
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4, 0.2), 0.281924022285*u.Mpc)
    numpy.testing.assert_equal(nfw_3.scale_radius(1E15, 3, 0.2), 0.809849191419*u.Mpc)
    numpy.testing.assert_equal(nfw_3.scale_radius(1E13, 5, 0.2), 0.104686031501*u.Mpc)
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4, 0.1), 0.281924022285*u.Mpc)
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4.5, 0.3), 0.25059913092*u.Mpc)
    nfw_4 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_m', comoving=False)
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4, 0.2), 
                               1.2*nfw_4.scale_radius(1E14, 4, 0.2))
    numpy.testing.assert_equal(nfw_3.scale_radius(1E15, 3, 0.2), 
                               1.2*nfw_4.scale_radius(1E15, 3, 0.2))
    numpy.testing.assert_equal(nfw_3.scale_radius(1E13, 5, 0.2), 
                               1.2*nfw_4.scale_radius(1E13, 5, 0.2))
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4, 0.1), 
                               1.1*nfw_4.scale_radius(1E14, 4, 0.1))
    numpy.testing.assert_equal(nfw_3.scale_radius(1E14, 4.5, 0.3), 
                               1.3*nfw_4.scale_radius(1E14, 4.5, 0.3))
    nfw_5 = offset_nfw.NFWModel(cosmo, delta=150, rho='rho_c', comoving=False)
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4, 0.2), 
                               1.2*nfw_5.scale_radius(1E14, 4, 0.2))
    numpy.testing.assert_equal(nfw_2.scale_radius(1E15, 3, 0.2), 
                               1.2*nfw_5.scale_radius(1E15, 3, 0.2))
    numpy.testing.assert_equal(nfw_2.scale_radius(1E13, 5, 0.2), 
                               1.2*nfw_5.scale_radius(1E13, 5, 0.2))
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4, 0.1), 
                               1.1*nfw_5.scale_radius(1E14, 4, 0.1))
    numpy.testing.assert_equal(nfw_2.scale_radius(1E14, 4.5, 0.3), 
                               1.3*nfw_5.scale_radius(1E14, 4.5, 0.3))
    
    try:
        import galsim
        for m, c, z in m_c_z_test_list:
            nfw_comp = galsim.NFWModel(m, c, z, omega_m=cosmo.Om0)
            numpy.testing.assert_equal(nfw_1.scale_radius(m, c, z), nfw_comp.rs)
    except ImportError:
        pass
    
    
def test_against_galsim_theory():
    """ Test against the GalSim implementation of NFWs. """
    try:
        import galsim
        nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
        
        z_source = 4.95
        
        radbins = numpy.exp(numpy.linspace(numpy.log(0.01), numpy.log(100), num=100))
        for m, c, z in [(1E14, 4, 0.2), (1E13, 4, 0.2), (1E15, 4, 0.2),
                        (1E14, 2, 0.2), (1E14, 6, 0.2), 
                        (1E14, 4, 0.05), (1E14, 4, 0.5), (1E14, 4, 4)]:
            galsim_nfw = galsim.NFWModel(m, c, z, omega_m = cosmo.Om0)
            angular_pos_x = radbins/cosmo.angular_diameter_distance(z)*206265
            angular_pos_y = numpy.zeros_like(angular_pos_x)
            numpy.testing.assert_equal(galsim_offset_nfw.getShear((angular_pos_x, angular_pos_y), z_source), 
                                       nfw_1.gamma_theory(radbins, m, c, z, z_source))
            numpy.testing.assert_equal(galsim_offset_nfw.getConvergence((angular_pos_x, angular_pos_y), z_source), 
                                       nfw_1.kappa_theory(radbins, m, c, z, z_source))
            numpy.testing.assert_equal(galsim_offset_nfw.getShear((angular_pos_x, angular_pos_y), z_source, reduced=True), 
                                       nfw_1.g_theory(radbins, m, c, z, z_source))
        
    except ImportError:
        import warnings
        warnings.warn("Could not test against GalSim -- import failure")
        return True
    
def test_against_cluster_lensing_theory():
    """ Test against the cluster-lensing implementation of NFWs. """
    try:
        import clusterlensing
    except ImportError:
        import warnings
        warnings.warn("Could not test against cluster-lensing -- import failure")

def test_sigma_to_deltasigma_theory():
    """ Test that the numerical sigma -> deltasigma produces the theoretical DS. """
    radbins = numpy.exp(numpy.linspace(numpy.log(0.01), numpy.log(100), num=100))
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    for m,c, _ in m_c_z_test_list:
        ds = nfw_1.deltasigma_theory(radbins, 1E14, 4)
        sig = nfw_1.sigma_theory(radbins, 1E14, 4)
        numpy.testing.assert_equal(ds, nfw_1.sigma_to_deltasigma(radbins, sig))
    #TODO: do again, miscentered
        
def test_z_ratios_theory():
    """ Test that the theoretical shear changes properly with redshift"""
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    base = nfw_1.gamma_theory(1., 1.E14, 4, 0.1, 0.15)
    new_z = numpy.linspace(0.15, 1.1, nsteps=20)
    new_gamma = nfw_1.gamma_theory(1, 1.E14, 4, 0.1, new_z)
    new_gamma /= base
    numpy.testing.assert_equal(new_gamma, cosmo.angular_diameter_distance_z1z2(0.1, new_z)/cosmo.angular_diameter_distance_z1z2(0.1, 0.15))

    base = nfw_1.sigma_theory(1., 1.E14, 4, 0.1, 0.15)
    new_sigma = nfw_1.sigma_theory(1, 1.E14, 4, 0.1, new_z)
    new_sigma /= base
    numpy.testing.assert_equal(new_sigma, cosmo.angular_diameter_distance_z1z2(0.1, new_z)/cosmo.angular_diameter_distance_z1z2(0.1, 0.15))
    
    #TODO: do again, miscentered
    
def test_g():
    """ Test that the theoretical returned g is the appropriate reduced shear """
    nfw_1 = offset_nfw.NFWModel(cosmo, delta=200, rho='rho_c')
    z_source = 4.95
    for m, c, z in m_c_z_test_list:
        numpy.testing.assert_equal(nfw_1.g_theory(m, c, z, z_source), nfw_1.gamma_theory(m, c, z, z_source)/(1+nfw_1.kappa_theory(m, c, z, z_source)))
    
def test_Upsilon():
    """ Test that the theoretical Upsilon is the appropriate value. """
    pass
    
def setup_table():
    """ Generate a small interpolation table so we can test its outputs. """
    
    nfw_halo = offset_nfw.NFWModel(cosmo, generate=True, x_range=(0.1, 2), miscentering_range=(0, 0.2))
    
if __name__=='__main__':
    #setup_table()
    test_object_creation()
    test_form_iterables()
    test_scale_radii()
    test_z_ratios_theory()
    test_against_galsim_theory()
    test_against_clusterlensing_theory()
    test_sigma_to_deltasigma_theory()
    test_g()
    test_Upsilon()