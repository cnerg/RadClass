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
spectra = np.arange(test_data.timesteps)
spectra = np.full((test_data.energy_bins, spectra.shape[0]), spectra).T


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


def test_init():
    confidence = 0.8

    bckg = BackgroundEstimator(confidence=confidence)

    np.testing.assert_equal(confidence, bckg.confidence)


def test_estimation():
    stride = 10
    integration = 10

    confidence = 0.8
    bckg = BackgroundEstimator(confidence=confidence)
    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, store_data=True, analysis=bckg)
    classifier.run_all()

    bckg.estimate()

    # the resulting observation should be:
    #   counts * integration / live-time
    expected = (((integration-1)**2 + (integration-1)) /
                (2*test_data.livetime)) * test_data.energy_bins / integration
    np.testing.assert_almost_equal(bckg.background.iloc[0][1],
                                   expected,
                                   decimal=3)

    time_idx = np.where(spectra == 0)[0][0]
    np.testing.assert_equal(bckg.background.iloc[0][0], timestamps[time_idx])

    expected_num = round((test_data.timesteps / integration) *
                         (1 - confidence))
    np.testing.assert_equal(len(bckg.background), expected_num)


def test_write():
    stride = 10
    integration = 10

    confidence = 0.8
    bckg = BackgroundEstimator(confidence=confidence)
    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, store_data=True, analysis=bckg)
    classifier.run_all()

    ofilename = 'bckg_test'
    bckg.write(ofilename=ofilename)

    results = np.genfromtxt(ofilename+'.csv', delimiter=',')[1:]
    print(results)

    # the resulting observation should be:
    #   counts * integration / live-time
    expected = (((integration-1)**2 + (integration-1)) /
                (2*test_data.livetime)) * test_data.energy_bins / integration
    np.testing.assert_almost_equal(results[0][1], expected, decimal=3)

    time_idx = np.where(spectra == 0)[0][0]
    np.testing.assert_equal(results[0][0], timestamps[time_idx])

    expected_num = round((test_data.timesteps / integration) *
                         (1 - confidence))
    np.testing.assert_equal(results.shape[0], expected_num)

    os.remove(ofilename+'.csv')
