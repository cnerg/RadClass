import h5py

filename = 'testfile.h5'
datapath = '/uppergroup/lowergroup/'
labels = {'live': '2x4x16LiveTimes',
          'timestamps': '2x4x16Times',
          'spectra': '2x4x16Spectra'}

energy_bins = 1000
timesteps = 1000
livetime = 0.9


def create_file(live, timestamps, spectra):
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
