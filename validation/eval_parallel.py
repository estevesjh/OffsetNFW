import numpy as np
import sys
import os
path = os.getcwd().split('/validation')[0]
print(path)
sys.path.append(path)

import astropy.cosmology
from multiprocessing import Pool
from offset_nfw.nfw import NFWModel
import astropy.cosmology

def parallel_generate_model(x_range):
    cosmology = astropy.cosmology.Planck15
    model = NFWModel(cosmology, sigma=True, gamma=True, nsize=20, x_range=x_range)
    model.generate(save=False)
    return model

if __name__ == "__main__":
    size = 4

    xlow, xhig = 1e-1, 1e1
    x_ranges = np.logspace(np.log10(xlow), np.log10(xhig), size + 1)
    x_ranges = [(x_ranges[i], x_ranges[i+1]) for i in range(size)]

    # Create a multiprocessing pool with the desired number of processes
    pool = Pool()

    # Perform parallel generation of models using map function
    all_models = pool.map(parallel_generate_model, x_ranges)

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    print('Saving Results')
    # Process the gathered models as needed
    for model in all_models:
        # Perform further processing or analysis
        froot = path + model.table_file_root[1:]
        print('Saved: %s'%(froot+'_miscentered_sigma.npy'))
        np.save(froot+'_miscentered_sigma.npy', model._miscentered_sigma)
        np.save(froot+'_miscentered_sigma_error.npy', model._miscentered_sigma_err)