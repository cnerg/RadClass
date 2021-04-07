import h5py

def create_file(filename, datapath, labels, live, timestamps, spectra):
    # Creating sample dataset
    f = h5py.File(filename, "w")
    
    # data structure for MUSE files
    dset1 = f.create_dataset(datapath + labels['live'], (1000,))
    dset2 = f.create_dataset(datapath + labels['timestamps'], (1000,))
    dset3 = f.create_dataset(datapath + labels['spectra'], (1000,1000))

    # store randomized data in test file
    dset1[...] = live
    dset2[...] = timestamps
    dset3[...] = spectra

    # close test file
    f.close()