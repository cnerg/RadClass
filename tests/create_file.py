import h5py

def create_file(filename, node, live, timestamps, spectra):
    # Creating sample dataset
    f = h5py.File(filename, "w")
    
    # group structure for MUSE files
    grp1 = f.create_group('auto_prefixed_datasets')
    grp2 = grp1.create_group(node)
    
    # data structure for MUSE files
    dset1 = grp2.create_dataset('2x4x16LiveTimes', (1000,))
    dset2 = grp2.create_dataset('2x4x16Times', (1000,))
    dset3 = grp2.create_dataset('2x4x16Spectra', (10,1000))

    # store randomized data in test file
    dset1[...] = live
    dset2[...] = timestamps
    dset3[...] = spectra

    # close test file
    f.close()