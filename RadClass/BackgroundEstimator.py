import numpy as np


class BackgroundEstimator:
    '''
    Integrates a gross count rate from a spectrum and orders samples in
        ascending order of magnitude. Saves a certain percentage samples
        to file with corresponding gross count rate.

    Attributes:
    header: Column names used for writing to file.
    background: Numpy array used to store timestamps and count rates.
    confidence: Percentage of samples to disregard. 1-confidence number of
        samples are saved. For example, if confidence=0.95, the lowest 5% of
        count rates/timestamps are saved.
    spectra: Optional NumPy array that stores spectral data from energy bins.
    energy_bins: The number of energy bins defined by the data under analysis.
    '''

    # columns: timestamp, count-rate
    header = ['timestamp', 'count-rate']
    background = np.empty((0, 2))

    def __init__(self, confidence=0.95):
        self.confidence = confidence

    def sort(self):
        '''
        Wrapper function for numpy array sort. Organizes samples in ascending
            order of gross count rate, i.e. column 1.
        '''

        # sort rows of background by the second column order
        return self.background[np.argsort(self.background[:,1])]

    def estimate(self):
        '''
        Selects a subset of numpy array with the smallest gross count rates.
        If not writing to file, this must be called manually to prune samples.

        Attributes:
        cutoff: count-rate value for the percentile of samples
        '''

        # find number of background samples
        cutoff = np.percentile(self.background[:, 1], (1-self.confidence)*100)
        self.background = self.background[self.background[:, 1] < cutoff]

        # sort for convenience the smallest count-rates
        self.background = self.sort()

    def run(self, data, timestamp):
        '''
        Method called by parent class during analysis. Integrates/collects a
            gross count rate and stores it in an iterim NumPy array.

        Attributes:
        count_rate: gross count rate from individual observation.
        '''

        count_rate = np.sum(data)

        # using numpy to build lightweight matrix instead of pandas
        self.background = np.vstack([self.background, [timestamp, count_rate]])

    def write(self, ofilename='bckg_results'):
        '''
        Writes results to file.
        Calls estimate() first to prune data array, only saving
            (1-confidence)% of samples.

        Attributes:
        ofilename: string, exluding file extension, for filename to save.
        '''

        # until estimate() is called, all background count-rates are saved
        self.estimate()
        np.savetxt(fname=ofilename+'.csv',
                   X=self.background,
                   delimiter=',',
                   header=', '.join(col for col in self.header))
