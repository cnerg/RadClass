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
        # initializing spectra, with bckg to be subtracted below
        self.diff_spectra = data[self.stride:].copy()
        # compute the gross integrated count-rate for every
        # row/spectrum in the dataset
        sums = np.sum(data[:, 1:], axis=1)
        # loop over every index in the difference spectra to be calculated
        # that is, from self.stride to the end of the dataset
        # (since those are the spectra with the requisite window
        # of length self.stride before it)
        for i in range(self.diff_spectra.shape[0]):
            # from a window of sums (i.e. gross integrated count-rates)
            # of length self.stride find the minimum
            # select the smallest spectrum as bckg_spectrum
            # sums includes all data [:] but bckg_spectrum is computed
            # for [self.stride:], so indexing data is offset by i
            bckg_spectrum = data[np.argmin(sums[i:i+self.stride])+i]
            # subtract background from stored spectrum for the appropriate row
            self.diff_spectra[i, 1:] -= bckg_spectrum[1:]

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
