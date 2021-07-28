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
        self.trigger_times = np.empty((0, 1))
        if self.gross:
            # three values are saved: x1, x2, and p-val for rejected hypotheses
            self.triggers = np.empty((0, 3))
        else:
            # all p-vals for rejected hypothesis are saved in this sparse array
            self.triggers = np.empty((0, energy_bins))

    def run(self, data, timestamp):
        '''
        Method required by RadClass. Called at each integration step.
        Completes hypothesis testing for passed data and stores results.
        '''

        if self.gross:
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
                                              [[pval, self.x1, self.x2]],
                                              axis=0)
                    self.trigger_times = np.append(self.trigger_times,
                                                   [[timestamp]],
                                                   axis=0)

                # saving data for the next integration step
                self.x1 = self.x2
        else:
            # only needed for the first initialization
            if self.x1 is None:
                self.x1 = data
            else:
                self.x2 = data
                pvals = np.empty_like(data)
                rejections = np.zeros_like(data)
                for i in range(len(self.x2)):
                    n = self.x1[i] + self.x2[i]
                    p = 0.5
                    pvals[i] = stats.binom_test(self.x1[i], n, p,
                                                alternative='two-sided')
                    if pvals[i] <= self.significance:
                        rejections[i] = pvals[i]
                if np.nonzero(rejections)[0].size != 0:
                    self.triggers = np.append(self.triggers,
                                              [rejections],
                                              axis=0)
                    self.trigger_times = np.append(self.trigger_times,
                                                   [[timestamp]],
                                                   axis=0)

                # saving data for the next integration step
                self.x1 = self.x2

    def write(self, filename):
        '''
        Writes results of hypothesis test to file by constructing a Pandas
        DataFrame and saving to csv with name filename.
        '''

        results = pd.DataFrame()
        results['timestamps'] = self.trigger_times[:, 0]
        if len(self.triggers[0]) == 3:
            results['pval'] = self.triggers[:, 0]
            results['x1'] = self.triggers[:, 1]
            results['x2'] = self.triggers[:, 2]
        else:
            for i in range(len(self.triggers[0])):
                results[str(i)] = self.triggers[:, i]

        results.to_csv(filename, sep=',')
