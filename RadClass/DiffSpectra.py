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
        windows = [data[i-self.stride:i] for i in range(self.stride, data.shape[0])]
        # save the smallest (by gross counts) spectra
        # for each background window (skipping timestamp in first col)
        bckg_spectra = [x[np.argmin(np.sum(x[:, 1:], axis=1))] for x in windows]

        # skipping timestamp in first col, calculate difference spectra
        # for each spectrum with the bckg_spectra for the window prior to it
        self.diff_spectra = data[self.stride:, 1:] - np.asarray(bckg_spectra)[:, 1:]
        # save final data with timestamps from original data
        self.diff_spectra = np.c_[data[self.stride:, 0], self.diff_spectra]

    def write(self, filename):
        '''
        Write results to file using numpy.savetxt() method.
        filename should include the file extension.
        '''
        with open(filename, 'a') as f:
            header = ''
            # build/include header if file is new
            if f.tell() == 0:
                header = np.append(['timestamp'],
                                   np.arange(len(self.diff_spectra[0])).astype(str))
                header = ', '.join(col for col in header)
            np.savetxt(fname=f,
                       X=self.diff_spectra,
                       delimiter=',',
                       header=header)
