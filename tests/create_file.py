import h5py


def create_file(filename, datapath, labels, live, timestamps, spectra):
    # Creating sample dataset
    f = h5py.File(filename, "w")

    # data structure for MUSE files
    dset1 = f.create_dataset(datapath + labels['live'], (1000,),
                             data=live, dtype='float64')
    dset2 = f.create_dataset(datapath + labels['timestamps'], (1000,),
                             data=timestamps, dtype='float64')
    dset3 = f.create_dataset(datapath + labels['spectra'], (1000, 1000),
                             data=spectra, dtype='float64')

    # close test file
    f.close()
