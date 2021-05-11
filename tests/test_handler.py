import numpy as np
import os
from datetime import datetime, timedelta

from RadClass.analysis import RadClass
from tests.create_file import create_file


def test_integration():
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)
    timestamps = np.array([])
    for i in range(1000):
        timestamps = np.append(timestamps, start_date.timestamp())
        start_date += delta
    
    livetime = 0.9
    live = np.full((len(timestamps),), 0.9)
    spectra = np.full((len(timestamps), 1000), np.full((1, 1000) ,10.0))

    # create sample test file with above simulated data
    create_file(filename, datapath, labels, live, timestamps, spectra)

    stride = 60
    integration = 60

    # run handler script
    classifier = RadClass(stride, integration, datapath, filename, store_data=True)
    classifier.run_all()

    # the resulting 1-hour observation should be counts * integration / live-time
    expected = spectra * integration / livetime
    results = np.genfromtxt('results.csv', delimiter=',')[1, 1:]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)

    os.remove(filename)
    os.remove('results.csv')

def test_cache():
    filename = 'testfile.h5'
    datapath = '/uppergroup/lowergroup/'
    labels = {'live': '2x4x16LiveTimes',
              'timestamps': '2x4x16Times',
              'spectra': '2x4x16Spectra'}

    start_date = datetime(2019, 2, 2)
    delta = timedelta(seconds=1)
    timestamps = np.array([])
    for i in range(1000):
        timestamps = np.append(timestamps, start_date.timestamp())
        start_date += delta
    
    livetime = 0.9
    live = np.full((len(timestamps),), 0.9)
    spectra = np.full((len(timestamps), 1000), np.full((1, 1000) ,10.0))

    # create sample test file with above simulated data
    create_file(filename, datapath, labels, live, timestamps, spectra)

    stride = 60
    integration = 60
    cache_size = 100

    # run handler script
    classifier = RadClass(stride, integration, datapath, filename, store_data=True, cache_size=cache_size)
    classifier.run_all()

    # the resulting 1-hour observation should be counts * integration / live-time
    expected = spectra * integration / livetime
    results = np.genfromtxt('results.csv', delimiter=',')[1, 1:]
    np.testing.assert_almost_equal(results, expected[0], decimal=2)

    os.remove(filename)
    os.remove('results.csv')
