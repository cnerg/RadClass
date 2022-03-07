import numpy as np
import pytest
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
from RadClass.DiffSpectra import DiffSpectra
import tests.test_data as test_data

# initialize sample data
start_date = datetime(2019, 2, 2)
delta = timedelta(seconds=1)
timestamps = np.arange(start_date,
                       start_date + (test_data.timesteps * delta),
                       delta).astype('datetime64[s]').astype('float64')

live = np.full((len(timestamps),), test_data.livetime)
spectra = np.arange(test_data.timesteps)
spectra = np.full((test_data.energy_bins, spectra.shape[0]), spectra).T


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


def test_difference():
    stride = 10
    integration = 10

    # run handler script with analysis parameter
    post_analysis = DiffSpectra()
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, post_analysis=post_analysis,
                          store_data=True)
    classifier.run_all()

    diff_spectra = post_analysis.diff_spectra
    print(post_analysis.diff_spectra.shape)
    print(diff_spectra)
    exp = np.ones_like(diff_spectra)
    np.testing.assert_equal(diff_spectra, exp)
