""" This code generates the sigma mis-centered library file
    of the NFWModel. The precision of the grid can be increased by Nsize.
    The range of the grid can also be adjusted. However, the precision of the integration
    is hard code inside the NFWModel().generate_miscentered_sigma_parallel() code. 
    To modify you should EPSABS = 1e-4; EPSREL = 1.49e-3 in the nfw.py file.
"""
import numpy as np
import sys
import os

import astropy.cosmology
from multiprocessing import Pool
from nfw import NFWModel
import astropy.cosmology
import time
t0 = time.time()

def parallel_generate_model(xmis_range):
    cosmology = astropy.cosmology.Planck15
    model = NFWModel(cosmology, Nsize=nsize_per_task, x_range=xmis_range)

    # integration over R/R_s vector
    xvec = np.logspace(np.log10(xlow),np.log10(xhig), Nsize)
    model.generate_miscentered_sigma_parallel(xvec)
    return model.miscentered_dict

def parallel_generate_model2(tau_range):
    cosmology = astropy.cosmology.Planck15
    model = NFWModel(cosmology, Nsize=Nsize, x_range=(xlow,xhig))
    
    tauvec = np.logspace(np.log10(tau_range[0]),np.log10(tau_range[1]),nsize_per_task)
    model.generate_gamma_sigma_parallel(tauvec)
    return model.gamma_dict

def tupdate(s):
    t = time.time()-t0
    if t<60:
        print("%.1f seconds have elapsed at point %s"%(t,s))
    elif t<60*60:
        print("%.2f minutes have elapsed at point %s"%(t/60.,s))
    else:
        print("%i hours and %.2f minutes have elapsed %s"%(int(t/3600), t/60., s))
    pass

def append(mylist,var,operation=np.hstack):
    res = []
    for out in mylist:
        res.append(out[var])
    return operation(res)

def join_all(fname, datas):
    x = append(datas,'x1',np.unique)
    xmis = append(datas,'x2')
    sigma_mis_err = append(datas,'vec_err')
    sigma_mis = append(datas,'vec',np.vstack)

    print('Saved: %s'%fname)
    np.savez(fname, x=x, xmis=xmis, 
            sigma_mis=sigma_mis, sigma_mis_err=sigma_mis_err)

def join_all_gamma(fname, datas):
    x = append(datas,'x1',np.unique)
    tau = append(datas,'x2')
    sigma_gamma = append(datas,'vec',np.vstack)

    print('Saved: %s'%fname)
    np.savez(fname, x=x, tau=tau, 
            sigma_gamma=sigma_gamma)

def check_path():
    if not os.path.isdir(path):
        os.makedirs(path)

def buildMiscenteredSigma():
    # split the xmis vector; 
    # the addtional factors 1/10 and 100 
    # are setup to have good integration limits for the gamma function
    # the gamma distribution is asymmetrical, with a larger upper tail
    x_ranges = np.logspace(np.log10(xlow)-1., np.log10(xhig) + 2., ntasks + 1)
    x_ranges = [(x_ranges[i], x_ranges[i+1]) for i in range(ntasks)]
    check_path()

    # Create a multiprocessing pool with the desired number of processes
    pool = Pool(processes=nCores)
    # Perform parallel generation of models using map function
    datas = pool.map(parallel_generate_model, x_ranges)
    # Close the multiprocessing pool
    pool.close()
    pool.join()
    tupdate("end multprocessing")

    tupdate("joining all")
    join_all(fnameOut+'_miscentered.npz', datas)

def buildGammaSigma():
    x_ranges = np.logspace(np.log10(xlow), np.log10(xhig), ntasks + 1)
    x_ranges = [(x_ranges[i], x_ranges[i+1]) for i in range(ntasks)]
    check_path()
    
    # Create a multiprocessing pool with the desired number of processes
    pool = Pool(processes=nCores)
    # Perform parallel generation of models using map function
    datas = pool.map(parallel_generate_model2, x_ranges)
    # Close the multiprocessing pool
    pool.close()
    pool.join()
    tupdate("end multprocessing")

    tupdate("joining all")
    join_all_gamma(fnameOut+'_gamma.npz', datas)

def main():
    tupdate("Build Miscentered Sigma")
    buildMiscenteredSigma()

    tupdate("Build Gamma Sigma")
    buildGammaSigma()
    
if __name__ == "__main__":
    ########## SETUP ############
    xlow, xhig = 1e-2, 1e5
    Nsize = 10000
    ntasks = 20
    nsize_per_task = 500
    nCores = 20
    path = 'data'
    fnameOut = path+'/offset_nfw_table_%i_%.0e_%.0e'%(Nsize,xlow,xhig)
    #############################
    main()