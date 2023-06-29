import sys
import os
path = os.getcwd().split('/validation')[0]
print(path)
sys.path.append(path)

import astropy.cosmology
from mpi4py import MPI
import numpy as np
from offset_nfw.nfw import NFWModel
def parallel_generate_model(x_range):
    cosmology = astropy.cosmology.Planck15
    model = NFWModel(cosmology, sigma=True, gamma=True, nsize=20, x_range=x_range)
    model.generate(save=False)
    return model

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    xlow, xhig = 1e-1, 1e1
    x_ranges = np.logspace(np.log10(xlow), np.log10(xhig), size + 1)
    x_ranges = [(x_ranges[i], x_ranges[i+1]) for i in range(size)]

    # Divide the x_ranges among the processes using MPI scatter
    local_x_range = comm.scatter(x_ranges, root=0)

    # Perform parallel generation of models
    local_model = parallel_generate_model(local_x_range)

    # Gather the local models from all processes
    all_models = comm.gather(local_model, root=0)

    if rank == 0:
        print('Saving Results')
        # Process the gathered models as needed
        for model in all_models:
            # Perform further processing or analysis
            froot = path + model.table_file_root[1:]
            print('Saved: %s'%(froot+'_miscentered_sigma.npy'))
            np.save(froot+'_miscentered_sigma.npy', model._miscentered_sigma)
            np.save(froot+'_miscentered_sigma_error.npy', model._miscentered_sigma_err)