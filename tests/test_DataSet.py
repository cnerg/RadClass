import numpy as np
import os

import RadClass.DataSet as ds
import tests.create_file as file


def test_init_database():
    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(file.timesteps,)
    timestamps = np.random.rand(file.timesteps,)
    spectra = np.random.rand(file.timesteps, file.energy_bins)

    file.create_file(live, timestamps, spectra)

    processor = ds.DataSet(file.labels)

    processor.init_database(file.filename, file.datapath)

    # remove file
    processor.close()
    os.remove(file.filename)

    # checks if arrays were saved to file and read back correctly
    np.testing.assert_almost_equal(processor.live, live, decimal=5)
    np.testing.assert_almost_equal(processor.timestamps, timestamps, decimal=5)


def test_data_slice():
    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(file.timesteps,)
    timestamps = np.random.rand(file.timesteps,)
    spectra = np.random.rand(file.timesteps, file.energy_bins)

    file.create_file(live, timestamps, spectra)

    processor = ds.DataSet(file.labels)
    # load file into processor's "memory"
    processor.init_database(file.filename, file.datapath)

    # query 3 random rows in the fake spectra matrix
    rows = np.random.choice(range(10), 3, replace=False)
    # sorted() for correct index syntax
    real_slice = spectra[sorted(rows)]
    test_slice = processor.data_slice(file.datapath, sorted(rows))

    # remove file
    processor.close()
    os.remove(file.filename)

    # check the entire array for approximately equal
    np.testing.assert_almost_equal(test_slice, real_slice, decimal=5)


def test_close():
    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(file.timesteps,)
    timestamps = np.random.rand(file.timesteps,)
    spectra = np.random.rand(file.timesteps, file.energy_bins)

    file.create_file(live, timestamps, spectra)

    processor = ds.DataSet(file.labels)
    # load file into processor's "memory"
    processor.init_database(file.filename, file.datapath)

    processor.close()
    # fails if file was not closed
    # therefore processor.file = True because it is still a file
    assert not processor.file

    os.remove(file.filename)
