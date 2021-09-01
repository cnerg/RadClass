import numpy as np
import pytest
import os
from datetime import datetime, timedelta

from RadClass.RadClass import RadClass
from RadClass.H0 import H0
import tests.test_data as test_data

# initialize sample data
start_date = datetime(2019, 2, 2)
delta = timedelta(seconds=1)
timestamps = np.arange(start_date,
                       start_date + (test_data.timesteps * delta),
                       delta).astype('datetime64[s]').astype('float64')

live = np.full((len(timestamps),), test_data.livetime)
sample_val = 1.0
spectra = np.full((len(timestamps), test_data.energy_bins),
                  np.full((1, test_data.energy_bins), sample_val))
# setting up for rejected null hypothesis
rejected_H0_time = test_data.timesteps//2
spectra[rejected_H0_time:] = 100.0


@pytest.fixture(scope="module", autouse=True)
def init_test_file():
    # create sample test file with above simulated data
    yield test_data.create_file(live, timestamps, spectra)
    os.remove(test_data.filename)


def test_init():
    significance = 0.1
    gross = False
    energy_bins = 10
    analysis = H0(significance=significance,
                  gross=gross,
                  energy_bins=energy_bins)

    np.testing.assert_equal(analysis.significance, significance)
    np.testing.assert_equal(analysis.gross, gross)
    np.testing.assert_equal(analysis.triggers.shape, (0, energy_bins+1))


def test_gross():
    stride = 10
    integration = 10

    # run handler script with analysis parameter
    analysis = H0()
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()

    np.testing.assert_equal(analysis.triggers[0][0],
                            timestamps[-rejected_H0_time])
    # there should only be one rejected hypothesis
    np.testing.assert_equal(analysis.triggers.shape[0], 1)


def test_channel():
    stride = 10
    integration = 10

    # run handler script with analysis parameter
    analysis = H0(gross=False, energy_bins=test_data.energy_bins)
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()

    np.testing.assert_equal(analysis.triggers[0][0],
                            timestamps[-rejected_H0_time])
    # there should only be one rejected hypothesis
    np.testing.assert_equal(analysis.triggers.shape[0], 1)
    # columns = 1 for timestamp + energy_bins
    np.testing.assert_equal(analysis.triggers.shape[1], test_data.energy_bins+1)


def test_write_gross():
    stride = 10
    integration = 10
    filename = 'h0test.csv'

    # run handler script with analysis parameter
    analysis = H0()
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()
    analysis.write(filename)

    results = np.genfromtxt(filename, delimiter=',')[1:]
    # 5 columns are required since the index is also saved (via pandas)
    np.testing.assert_equal(results.shape, (1, 5))

    os.remove(filename)


def test_write_channel():
    stride = 10
    integration = 10
    filename = 'h0test.csv'

    # run handler script with analysis parameter
    analysis = H0(gross=False, energy_bins=test_data.energy_bins)
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()
    analysis.write(filename)

    results = np.genfromtxt(filename, delimiter=',')[1:]
    # 2 extra columns are required for timestamp and index (via pandas)
    np.testing.assert_equal(results.shape, (1, test_data.energy_bins+2))

    os.remove(filename)
