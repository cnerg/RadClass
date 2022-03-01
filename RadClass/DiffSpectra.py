import numpy as np


class DiffSpectra:
    '''
    Estimates a minimum background spectrum for each
    timestamp/spectrum and saves a difference spectrum
    between the two.

    Attributes:
    stride: The number of timestamps prior to the current
        spectrum for finding the minimum background
    '''

    def __init__(self, stride=10):
        self.stride = stride

    def run(self, data):
        # index all windows of length self.diff_stride into a tmp array
        windows = [data[i-self.stride:self.stride] for i in range(self.stride-1, data.shape[0])]
        # filter out any windows without enough bckg samples
        # i.e. beginning of spectra w/ idx < self.diff_stride
        windows = [x for x in windows if x.shape[0] == self.stride]
        # save the smallest (by gross counts) spectra
        # for each background window (skipping timestamp in first col)
        bckg_spectra = [x[np.argmin(np.sum(x[:,1:], axis=1))] for x in windows]

        # skipping timestamp in first col, calculate difference spectra
        self.diff_spectra = data[self.stride:,1:] - np.asarray(bckg_spectra[:,1:])
        # save final data with timestamps
        self.diff_spectra = np.c_[bckg_spectra[:,0], bckg_spectra]