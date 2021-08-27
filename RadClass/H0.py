import numpy as np
import pandas as pd

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
    docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom_test.html

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
    '''

    def __init__(self, significance=0.05, gross=True, energy_bins=1000):
        self.significance = significance
        self.gross = gross
        self.x1 = None

        # arrays are structured as matrices (nxm) for writing to file
        if self.gross:
            # four values are saved: timestamp, x1, x2, and p-val
            self.triggers = np.empty((0, 4))
        else:
            # all p-vals for rejected hypothesis are saved in this sparse array
            self.triggers = np.empty((0, energy_bins+1))

    def run_gross(self, data, timestamp):
        data = np.sum(data)

        # only needed for the first initialization
        if self.x1 is None:
            self.x1 = data
        else:
            self.x2 = data
            n = self.x1 + self.x2
            p = 0.5
            pval = stats.binom_test(self.x2, n, p, alternative='two-sided')

            # only save instances with rejected null hypothesesf
            if pval <= self.significance:
                self.triggers = np.append(self.triggers,
                                            [[timestamp, pval,
                                            self.x1, self.x2]],
                                            axis=0)

            # saving data for the next integration step
            self.x1 = self.x2
    
    def run_channels(self, data, timestamp):
        # only needed for the first initialization
        if self.x1 is None:
            self.x1 = data
        else:
            self.x2 = data
            rejections = np.zeros_like(data)
            nvec = self.x1 + self.x2
            p = 0.5
            for i, (x1, n) in enumerate(zip(self.x1, nvec)):
                pval = stats.binom_test(x1, n, p,
                                            alternative='two-sided')
                if pval <= self.significance:
                    rejections[i] = pval
            if np.nonzero(rejections)[0].size != 0:
                self.triggers = np.append(self.triggers,
                                        [np.insert(rejections,
                                                    0, timestamp)],
                                        axis=0)

            # saving data for the next integration step
            self.x1 = self.x2

    run_method = { True: run_gross, False: run_channels }

    def run(self, data, timestamp):
        '''
        Method required by RadClass. Called at each integration step.
        Completes hypothesis testing for passed data and stores results.
        Wrapper method that chooses run_gross or run_channel based on user
        input variable: gross.
        '''
        self.run_method[self.gross](self, data, timestamp)

    def write(self, filename):
        '''
        Writes results of hypothesis test to file by constructing a Pandas
        DataFrame and saving to csv with name filename.
        '''

        results = pd.DataFrame()
        results['timestamps'] = self.triggers[:, 0]
        if len(self.triggers[0]) == 4:
            results['pval'] = self.triggers[:, 1]
            results['x1'] = self.triggers[:, 2]
            results['x2'] = self.triggers[:, 3]
        else:
            for i in range(len(self.triggers[0, 1:])):
                results[str(i)] = self.triggers[:, i]

        results.to_csv(filename, sep=',')
