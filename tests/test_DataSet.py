import numpy as np
import pytest
import os

import RadClass.DataSet as ds
import tests.test_data as test_data

# randomized data to store (smaller than an actual MUSE file)
live = np.random.rand(test_data.timesteps,)
timestamps = np.random.rand(test_data.timesteps,)
spectra = np.random.rand(test_data.timesteps, test_data.energy_bins)


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


def test_init_database():
    processor = ds.DataSet(test_data.labels)

    processor.init_database(test_data.filename, test_data.datapath)

    # remove file
    processor.close_file()

    # checks if arrays were saved to file and read back correctly
    np.testing.assert_almost_equal(processor.live, live, decimal=5)
    np.testing.assert_almost_equal(processor.timestamps, timestamps, decimal=5)


def test_data_slice():
    processor = ds.DataSet(test_data.labels)
    # load file into processor's "memory"
    processor.init_database(test_data.filename, test_data.datapath)

    # query 3 random rows in the fake spectra matrix
    rows = np.random.choice(range(10), 3, replace=False)
    # sorted() for correct index syntax
    real_slice = spectra[sorted(rows)]
    test_slice = processor.data_slice(test_data.datapath, sorted(rows))

    # remove file
    processor.close_file()

    # check the entire array for approximately equal
    np.testing.assert_almost_equal(test_slice, real_slice, decimal=5)


def test_close():
    processor = ds.DataSet(test_data.labels)
    # load file into processor's "memory"
    processor.init_database(test_data.filename, test_data.datapath)

    processor.close_file()
    # fails if file was not closed
    # therefore processor.file = True because it is still a file
    assert not processor.file
