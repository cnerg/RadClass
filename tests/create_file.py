import h5py


def create_file(filename, datapath, labels, live, timestamps, spectra, timesteps, energy_bins):
    # Creating sample dataset
    f = h5py.File(filename, "w")

    # data structure for MUSE files
    dset1 = f.create_dataset(datapath + labels['live'], (timesteps,),
                             data=live, dtype='float64')
    dset2 = f.create_dataset(datapath + labels['timestamps'], (timesteps,),
                             data=timestamps, dtype='float64')
    dset3 = f.create_dataset(datapath + labels['spectra'], (timesteps, energy_bins),
                             data=spectra, dtype='float64')

    # close test file
    f.close()
