import numpy as np
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
import tests.create_file as file

class NullAnalysis():
    changed = False
    
    # tracks whether this class is called properly
    def run(self, data):
        self.changed = True

def test_analysis():
    # randomized data to store (smaller than an actual MUSE file)
    live = np.random.rand(file.timesteps,)
    timestamps = np.random.rand(file.timesteps,)
    spectra = np.random.rand(file.timesteps, file.energy_bins)

    # create sample test file with above simulated data
    file.create_file(live, timestamps, spectra)

    stride = 60
    integration = 60

    # run handler script
    classifier = RadClass(stride, integration, file.datapath, file.filename,
                          analysis = NullAnalysis(), store_data=True)
    classifier.run_all()

    np.testing.assert_equal(True, classifier.analysis.changed)

def test_init():
    stride = 60
    integration = 60
    store_data = True
    cache_size = 10000

    classifier = RadClass(stride = stride, integration = integration,
                          datapath = file.datapath, filename = file.filename,
                          store_data=store_data, cache_size=cache_size,
                          labels = file.labels)

    np.testing.assert_equal(stride, classifier.stride)
    np.testing.assert_equal(integration, classifier.integration)
    np.testing.assert_equal(file.datapath, classifier.datapath)
    np.testing.assert_equal(file.filename, classifier.filename)
    np.testing.assert_equal(store_data, classifier.store_data)
    np.testing.assert_equal(cache_size, classifier.cache_size)
    np.testing.assert_equal(file.labels, classifier.labels)


def test_integration():
    # initialize sample data
    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)
    timestamps = np.arange(start_date, start_date + (file.timesteps * delta), delta).astype('datetime64[s]').astype('float64')

    live = np.full((len(timestamps),), file.livetime)
    spectra = np.full((len(timestamps), file.energy_bins), np.full((1, file.energy_bins), 10.0))

    # create sample test file with above simulated data
    file.create_file(live, timestamps, spectra)

    stride = 60
    integration = 60

    # run handler script
    classifier = RadClass(stride, integration, file.datapath,
                          file.filename, store_data=True)
    classifier.run_all()

    # the resulting 1-hour observation should be:
    #   counts * integration / live-time
    expected = spectra * integration / file.livetime
    results = np.genfromtxt('results.csv', delimiter=',')[1, 1:]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)

    os.remove(file.filename)
    os.remove('results.csv')
