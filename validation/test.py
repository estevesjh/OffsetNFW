import astropy.cosmology
import matplotlib.pyplot as plt

import sys
import os
path = os.getcwd().split('/validation')[0]
print(path)
sys.path.append(path)
from offset_nfw.nfw import NFWModel
import astropy.cosmology
cosmology = astropy.cosmology.Planck15
model = NFWModel(cosmology, sigma=True, gamma=True, nsize=100, x_range=(0.01,10.))
model.generate(sigma=True)
model.generate(gamma=True)

# plt.loglog()
# for ix in [50,60,70]:
#     plt.plot(model.table_x, model._miscentered_sigma[ix])

# # check sampling
# Nsize = model.table_x.size
# xm0 = 1.
# x = model.table_x

# xgrid = thetaFunctionSampling(x,xm0,Nsize)
# _ = plt.hist(np.log10(x),bins=20)
# plt.hist(np.log10(xgrid),bins=_[1])
# # plt.xscale('log')
