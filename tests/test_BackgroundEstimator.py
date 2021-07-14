import numpy as np
import pytest
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
from RadClass.BackgroundEstimator import BackgroundEstimator
import tests.test_data as test_data

start_date = datetime(2019, 2, 2)
delta = timedelta(seconds=1)
timestamps = np.arange(start_date,
                       start_date + (test_data.timesteps * delta),
                       delta).astype('datetime64[s]').astype('float64')

live = np.full((len(timestamps),), test_data.livetime)
# randomly order incremental "spectra"
values = np.arange(test_data.timesteps)
spectra = np.array([np.full((test_data.energy_bins,), x) for x in values])


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


def test_init():
    confidence = 0.8
    ofilename = 'test_ofilename'
    store_all = True

    bckg = BackgroundEstimator(confidence=confidence,
                               ofilename=ofilename,
                               store_all=store_all,
                               energy_bins=test_data.energy_bins)

    np.testing.assert_equal(confidence, bckg.confidence)
    np.testing.assert_equal(ofilename, bckg.ofilename)
    np.testing.assert_equal(store_all, bckg.store_all)
    np.testing.assert_equal(test_data.energy_bins, bckg.energy_bins)


def test_estimation():
    stride = 100
    integration = 100

    confidence = 0.8
    ofilename = 'bckg_results'
    bckg = BackgroundEstimator(confidence=confidence, ofilename=ofilename)
    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, store_data=True, analysis=bckg)
    classifier.run_all()

    bckg.write()

    # the resulting observation should be:
    #   counts * integration / live-time
    expected = (((integration-1)**2 + (integration-1)) /
                (2*test_data.livetime)) * test_data.energy_bins
    results = np.genfromtxt('bckg_results.csv', delimiter=',', skip_header=1)
    np.testing.assert_almost_equal(results[0][1], expected, decimal=3)

    time_idx = np.where(values == 0)[0][0]
    np.testing.assert_equal(results[0][0], timestamps[time_idx])

    expected_num = round((test_data.timesteps / integration) *
                         (1 - confidence))
    np.testing.assert_equal(len(results), expected_num)

    os.remove('bckg_results.csv')


def test_spectral_storage():
    stride = 100
    integration = 100

    confidence = 0.8
    ofilename = 'bckg_results'
    bckg = BackgroundEstimator(confidence=confidence, ofilename=ofilename,
                               store_all=True,
                               energy_bins=test_data.energy_bins)
    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, store_data=True, analysis=bckg)
    classifier.run_all()

    bckg.write()

    expected = np.full((test_data.energy_bins,),
                       ((integration-1)**2 + (integration-1)) /
                       (2*test_data.livetime))
    results = np.genfromtxt('bckg_results.csv', delimiter=',', skip_header=1)
    np.testing.assert_almost_equal(results[0][2:], expected, decimal=3)

    os.remove('bckg_results.csv')
