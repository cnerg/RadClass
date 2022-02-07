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

    np.testing.assert_equal(analysis.log_significance, np.log10(significance))
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

    obs_timestamp = analysis.triggers[0][0]
    exp_timestamp = timestamps[-(rejected_H0_time+integration)]
    np.testing.assert_equal(obs_timestamp,
                            exp_timestamp)
    # there should only be one rejected hypothesis
    obs_rows = analysis.triggers.shape[0]
    exp_rows = 1
    np.testing.assert_equal(obs_rows, exp_rows)


def test_channel():
    stride = 10
    integration = 10

    # run handler script with analysis parameter
    analysis = H0(gross=False, energy_bins=test_data.energy_bins)
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()

    obs_timestamp = analysis.triggers[0][0]
    exp_timestamp = timestamps[-(rejected_H0_time+integration)]
    np.testing.assert_equal(obs_timestamp,
                            exp_timestamp)
    # there should only be one rejected hypothesis
    obs_rows = analysis.triggers.shape[0]
    exp_rows = 1
    np.testing.assert_equal(obs_rows, exp_rows)
    # columns = 1 for timestamp + energy_bins
    obs_cols = analysis.triggers.shape[1]
    exp_cols = test_data.energy_bins+1
    np.testing.assert_equal(obs_cols, exp_cols)


def test_write_gross():
    stride = 10
    integration = 10
    filename = 'h0test_gross.csv'

    # run handler script with analysis parameter
    analysis = H0()
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()
    analysis.write(filename)

    results = np.loadtxt(filename, delimiter=',')
    # expected shape is only 1D because only 1 entry is expected
    obs = results.shape
    exp = (4,)
    np.testing.assert_equal(obs, exp)

    os.remove(filename)


def test_write_channel():
    stride = 10
    integration = 10
    filename = 'h0test_channel.csv'

    # run handler script with analysis parameter
    analysis = H0(gross=False, energy_bins=test_data.energy_bins)
    classifier = RadClass(stride, integration, test_data.datapath,
                          test_data.filename, analysis=analysis)
    classifier.run_all()
    analysis.write(filename)

    results = np.loadtxt(filename, delimiter=',')
    # 1 extra columns are required for timestamp
    # expected shape is only 1D because only 1 entry is expected
    obs = results.shape
    exp = (test_data.energy_bins+1,)
    np.testing.assert_equal(obs, exp)

    os.remove(filename)


def test_zero_counts_gross():
    # accept all pvals
    significance = 1.0
    energy_bins = 10

    analysis = H0(significance=significance,
                  gross=True,
                  energy_bins=energy_bins)

    spectrum1 = np.zeros((10,))
    spectrum2 = np.zeros((10,))
    # run twice for initialization
    analysis.run_gross(spectrum1, 1)
    analysis.run_gross(spectrum2, 2)

    obs = analysis.triggers
    exp = np.array([[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_equal(obs, exp)


def test_zero_counts_channel():
    # accept all pvals
    significance = 1.0
    energy_bins = 10

    analysis = H0(significance=significance,
                  gross=False,
                  energy_bins=energy_bins)

    spectrum1 = np.zeros((10,))
    spectrum2 = np.zeros((10,))
    spectrum2[0] = 100.0
    # run twice for initialization
    analysis.run_channels(spectrum1, 1)
    analysis.run_channels(spectrum2, 2)

    obs_pvals = analysis.triggers[0][1:]
    obs_ts = analysis.triggers[0][0]
    np.testing.assert_equal(obs_ts, 1)
    # only the first channel will have failed
    np.testing.assert_equal(np.count_nonzero(obs_pvals), 1)
    # check that the first channel alone is rejected
    np.testing.assert_equal(np.sum(obs_pvals), obs_pvals[0])
