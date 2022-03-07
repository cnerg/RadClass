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
    # small stride since there are less data samples in test_data
    diff_stride = 2
    post_analysis = DiffSpectra(stride=diff_stride)
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, post_analysis=post_analysis,
                          store_data=True)
    classifier.run_all()

    diff_spectra = post_analysis.diff_spectra
    # DiffSpectra length test:
    # there should be one difference spectrum for each timestamp with
    # diff_stride number of spectra before it
    exp_len = classifier.storage.shape[0] - diff_stride
    np.testing.assert_equal(diff_spectra.shape[0], exp_len)

    # DiffSpectra value test:
    # for test_data, the minimum background will always be the first spectrum
    # in a window (because the spectra increase 1, 2, 3, 4, etc.)
    # therefore the diff spectra element values will always be
    # spectra[i] - spectra[i-diff_stride]
    # (the algebra for n(n+1)/2 integrated intervals is simplified below)
    diff_value = 3*stride**2 - integration**2 \
        + (2*stride*integration) + stride - integration
    exp_spectra = np.full((exp_len, test_data.energy_bins),
                          diff_value/(2*test_data.livetime))
    np.testing.assert_almost_equal(diff_spectra[:, 1:], exp_spectra, decimal=2)

    # DiffSpectra timestamp test:
    # there should be one difference spectrum for each timestamp with
    # diff_stride number of spectra before it (i.e. spectra for every timestamp
    # after the first diff_stride spectra)
    exp_ts = classifier.storage[diff_stride:, 0]
    np.testing.assert_equal(diff_spectra[:, 0], exp_ts)
