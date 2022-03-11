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
        self.diff_spectra = data[self.stride:].copy()
        sums = np.sum(data[:, 1:], axis=1)
        for i in range(self.diff_spectra.shape[0]):
            bckg_spectrum = data[np.argmin(sums[i:i+self.stride])+i]
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
