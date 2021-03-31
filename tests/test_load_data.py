import numpy as np
import os

import RadClass.DataSet as ds
from tests.create_file import create_file

def test_init_database():
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(1000,)
    timestamps = np.random.rand(1000,)
    spectra = np.random.rand(10,1000)

    create_file(filename, datapath, labels, live, timestamps, spectra)

    processor = ds.DataSet(labels)

    processor.init_database(filename, datapath)

    # remove file
    os.remove(filename)

    # checks if arrays were saved to file and read back correctly
    np.testing.assert_almost_equal(processor.live, live, decimal = 5)
    np.testing.assert_almost_equal(processor.timestamps, timestamps, decimal = 5)

def test_data_slice():
    # Creating sample dataset
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(1000,)
    timestamps = np.random.rand(1000,)
    spectra = np.random.rand(10,1000)

    create_file(filename, datapath, labels, live, timestamps, spectra)

    processor = ds.DataSet(labels)

    # query 3 random rows in the fake spectra matrix
    rows = np.random.choice(range(10), 3, replace = False)
    # sorted() for correct index syntax
    real_slice = spectra[sorted(rows)]
    test_slice = processor.data_slice(filename, datapath, sorted(rows))

    # remove file
    os.remove(filename)

    # check the entire array for approximately equal
    np.testing.assert_almost_equal(test_slice, real_slice, decimal = 5)
