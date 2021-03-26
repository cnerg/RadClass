import h5py
import numpy as np

def create_file(filename):
    # Creating sample dataset
    f = h5py.File(filename, "w")

    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(1000,)
    timestamps = np.random.rand(1000,)
    spectra = np.random.rand(10,1000)
    
    # group structure for MUSE files
    grp1 = f.create_group('upper_group')
    grp2 = grp1.create_group('lower_group')
    
    # data structure for MUSE files
    dset1 = grp2.create_dataset('live', (1000,))
    dset2 = grp2.create_dataset('timestamps', (1000,))
    dset3 = grp2.create_dataset('spectra', (10,1000))

    # store randomized data in test file
    dset1[...] = live
    dset2[...] = timestamps
    dset3[...] = spectra

    # close test file
    f.close()