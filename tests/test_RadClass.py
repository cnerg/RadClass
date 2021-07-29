import numpy as np
import pytest
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
import tests.test_data as test_data

# initialize sample data
start_date = datetime(2019, 2, 2)
delta = timedelta(seconds=1)
timestamps = np.arange(start_date,
                       start_date + (test_data.timesteps * delta),
                       delta).astype('datetime64[s]').astype('float64')

live = np.full((len(timestamps),), test_data.livetime)
spectra = np.full((len(timestamps), test_data.energy_bins),
                  np.full((1, test_data.energy_bins), 10.0))


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


class NullAnalysis():
    changed = False

    # tracks whether this class is called properly
    def run(self, data):
        self.changed = True


def test_analysis():
    stride = 60
    integration = 60

    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=NullAnalysis())
    classifier.run_all()

    np.testing.assert_equal(True, classifier.analysis.changed)


def test_init():
    stride = 60
    integration = 60
    cache_size = 10000

    classifier = RadClass(stride=stride, integration=integration,
                          datapath=test_data.datapath,
                          filename=test_data.filename, cache_size=cache_size,
                          labels=test_data.labels)

    np.testing.assert_equal(stride, classifier.stride)
    np.testing.assert_equal(integration, classifier.integration)
    np.testing.assert_equal(test_data.datapath, classifier.datapath)
    np.testing.assert_equal(test_data.filename, classifier.filename)
    np.testing.assert_equal(cache_size, classifier.cache_size)
    np.testing.assert_equal(test_data.labels, classifier.labels)


def test_integration():
    stride = 60
    integration = 60

    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename)
    classifier.run_all()

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = spectra * integration / test_data.livetime
    results = classifier.storage.to_numpy()[1]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)


def test_cache():
    stride = 60
    integration = 60
    cache_size = 100

    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename,
                          cache_size=cache_size)
    classifier.run_all()

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = spectra * integration / test_data.livetime
    results = classifier.storage.to_numpy()[1]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)


def test_stride():
    stride = 100
    integration = 50

    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename)
    classifier.run_all()

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = spectra * integration / test_data.livetime
    expected_samples = int(test_data.timesteps /
                           (integration + (stride - integration)))
    np.testing.assert_almost_equal(classifier.storage.iloc[0],
                                   expected[0],
                                   decimal=2)
    np.testing.assert_equal(len(classifier.storage), expected_samples)


def test_write():
    stride = 60
    integration = 60
    filename = 'test_results.csv'

    # run handler script
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename)
    classifier.run_all()
    classifier.write(filename)

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = spectra * integration / test_data.livetime
    results = np.genfromtxt(filename, delimiter=',')[1, 1:]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)

    os.remove(filename)
