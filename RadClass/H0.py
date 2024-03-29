import numpy as np

from scipy import stats


class H0:
    '''
    Applies binomial hypothesis test to data processed with RadClass.
    Capable of applying a hypothesis test to gross or channel-wise count-rate.
    This test relies on the assumption that count-rates measured consecutively
    in time will not be statistical different from each other. That is, they
    should come from the same baseline background environment. If they are
    different (and therefore the null hypothesis is rejected), presumably a
    radioactive event occurred (e.g. SNM transfer).

    For information on testing regime, see doi: 10.2307/2332612.
    For information on scipy's binomial test, see:
    docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html

    Attributes:
    significance: Significance level for hypothesis test. If the resulting
        binomial test's p-value is smaller than significance, null hypothesis
        is rejected.
    gross: Boolean; determines whether only the gross count-rate is analyzed
        (True) or each energy channel individually (False).
    x1: The first timestamp's count-rate to be tested.
    x2: The second timestamp's count-rate to be compared with x1.
    triggers: Three things are saved in this array. x1, x2, and the p-value for
        rejected null hypotheses per the inputted significance level.
    trigger_times: The timestamp (in Unix epoch timestamp) for rejected null
        hypothesis via triggers.

    Returns:
    pvals: Maximum possible value is 1.0. pvals associated with rejected null
        hypotheses (pval <= significance) are stored as log_10(pval).
        Therefore, maximum possible -stored- value is 0.0. This avoids
        float rounding for extremely small pvals.
    '''

    def __init__(self, significance=0.05, gross=True, energy_bins=1000):
        # default: store log10 value of significance for comparison
        self.log_significance = np.log10(significance)
        self.gross = gross
        self.x1 = None

        # arrays are structured as matrices (nxm) for writing to file
        if self.gross:
            # four values are saved: timestamp, x1, x2, and p-val
            self.triggers = np.empty((0, 4))
        else:
            # all p-vals for rejected hypothesis are saved in this sparse array
            self.triggers = np.empty((0, energy_bins+1))

    def binom(self, x1, n, p):
        '''
        Private method for running binomial test.

        Return:
        lpval: log base 10 p-value result of scipy.stats.binomtest.
        '''
        # np.log10(1E-350), chosen to be smaller than any possible result
        min_lpval = -350.0
        # scipy.stats.binomtest will fail if n (# of trials)
        # is less than 1 (possible for high-energy bins)
        if int(n) < 1:
            lpval = 0.0
        else:
            pval = stats.binomtest(int(x1), int(n), p,
                                   alternative='two-sided').pvalue
            if pval == 0.0:
                lpval = min_lpval
            else:
                lpval = np.log10(pval)
        return lpval

    def run_gross(self, data, timestamp):
        '''
        Applies scipy.stats.binomtest (requires v. 1.7.0 or greater)
        to the gross, integrated count-rate. Spectral data is
        passed by RadClass.RadClass and integrated along all
        channels. Count-rates at x1 and x2 are tracked with
        associated timestamp for next binomtest at next timestamp.
        '''
        data = np.sum(data)

        # only needed for the first initialization
        if self.x1 is None:
            self.x1 = data
            self.x1_timestamp = timestamp
        else:
            self.x2 = data
            n = self.x1 + self.x2
            p = 0.5
            lpval = self.binom(self.x1, n, p)

            # only save instances with rejected null hypotheses
            if lpval <= self.log_significance:
                self.triggers = np.append(self.triggers,
                                          [[self.x1_timestamp, lpval,
                                           self.x1, self.x2]],
                                          axis=0)

            # saving data for the next integration step
            self.x1 = self.x2
            self.x1_timestamp = timestamp

    def run_channels(self, data, timestamp):
        '''
        Applies scipy.stats.binomtest (requires v. 1.7.0 or greater)
        to the channel/energy-wise count-rate. Spectral data is
        passed by RadClass.RadClass and binomtest is applied to
        each channel individually. Count-rates, x1 and x2, for
        every spectral bin are tracked with associated timestamp
        for next binomtest at next timestamp.
        '''
        # only needed for the first initialization
        if self.x1 is None:
            self.x1 = data
            self.x1_timestamp = timestamp
        else:
            self.x2 = data
            rejections = np.zeros_like(data)
            nvec = self.x1 + self.x2
            p = 0.5
            for i, (x1, n) in enumerate(zip(self.x1, nvec)):
                lpval = self.binom(x1, n, p)

                if lpval <= self.log_significance:
                    rejections[i] = lpval

            if np.sum(rejections) != 0:
                self.triggers = np.append(self.triggers,
                                          [np.insert(rejections,
                                           0, self.x1_timestamp)],
                                          axis=0)

            # saving data for the next integration step
            self.x1 = self.x2
            self.x1_timestamp = timestamp

    def run(self, data, timestamp):
        '''
        Method required by RadClass. Called at each integration step.
        Completes hypothesis testing for passed data and stores results.
        Wrapper method that chooses run_gross or run_channel based on user
        input variable: gross.
        '''
        run_method = { True: self.run_gross, False: self.run_channels }
        run_method[self.gross](data, timestamp)

    def write(self, filename):
        '''
        Writes results of hypothesis test to file using numpy.savetxt.
        '''

        with open(filename, 'a') as f:
            header = ''
            # build/include header if file is new
            if f.tell() == 0:
                if self.gross:
                    header = ['timestamps', 'pval', 'x1', 'x2']
                elif not self.gross:
                    header = np.append(['timestamp'],
                                       np.arange(self.triggers.shape[0]-1).astype(str))
                header = ', '.join(col for col in header)
            np.savetxt(fname=f,
                       X=self.triggers,
                       delimiter=',',
                       header=header)
