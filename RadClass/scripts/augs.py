import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import loguniform
from beads.beads.beads import beads


# DANS: Data Augmentations for Nuclear Spectra feature-Extraction
# TODO: standardize return to either include background or not
class DANSE:
    def __init__(self):
        self.BEADS_PARAMS = dict( 
            fc=4.749e-2,
            r=4.1083,
            df=2,
            lam0=3.9907e-4,
            lam1=4.5105e-3,
            lam2=3.3433e-3,
        )

    def _estimate(self, X_bckg, mode):
        '''
        Background estimation method used in background and sig2bckg.
        NOTE: Two background subtraction modes are supported: 'min' and 'mean'.
        'min': take the minimum gross count-rate spectrum from X.
        'mean': take the average count-rate for each bin from X.

        Inputs:
        X_bckg: array-like; 2D spectral array of measurements, uses the mode
            input to complete background superimposition.
        mode: str; two background subtraction modes are currently supported:
            'min': take the minimum gross count-rate spectrum from X.
            'mean': take the average count-rate for each bin from X.
        '''

        if mode == 'min':
            idx = np.argmin(np.sum(X_bckg, axis=0))
            X_bckg = X_bckg[idx]
        elif mode == 'mean':
            X_bckg = np.mean(X_bckg, axis=1)
        elif mode == 'beads':
            X_bckg = beads(X_bckg, **self.BEADS_PARAMS)[1]

        return X_bckg

    def background(self, X, X_bckg, subtraction=False,
                   event_idx=None, mode='mean'):
        '''
        Superimposes an even signature onto various forms of background
        distributions. This action does require an accurate estimation of a
        typical baseline, but several methods are employed.
        X is assumed to be background adjusted by default. That is, it should
        have its original background already removed.
        If subtraction=True and event_idx is not None, X should be 2D.
        event_idx indicates the row in X that indicates the event spectrum.
        The other rows in X are then used to estimate a background distribution
        for subtraction.
        NOTE: Two background subtraction modes are supported: 'min' and 'mean'.
        'min': take the minimum gross count-rate spectrum from X.
        'mean': take the average count-rate for each bin from X.

        Inputs:
        X: array-like; If 1D, taken as an event spectrum (must be previously
            background-subtracted). If 2D, subtraction=True, and event_idx not
            None, a background subtraction is conducted prior to
            superimposition.
        X_bckg: array-like; If 1D, add this spectrum to the event spectrum of X
            as the superimposed background. If 2D, use the mode input to
            complete background superimposition.
        subtraction: bool; If True, conduct background subtraction on X
            (event_idx must not be None)
        event_idx: int(p); row index for event spectrum in X used for
            background subtraction.
        mode: str; two background subtraction modes are currently supported:
            'min': take the minimum gross count-rate spectrum from X.
            'mean': take the average count-rate for each bin from X.
        '''

        X = X.copy()
        modes = ['min', 'mean', 'beads']
        # input error checks
        if subtraction and event_idx is None and X.ndim > 1:
            raise ValueError('If subtraction=True and len(X)>1, \
                        event_idx must be specified.')
        elif subtraction and event_idx is not None and X.ndim <= 1:
            raise ValueError('X must be 2D to do background subtraction.')
        elif X_bckg.ndim > 1 and mode == 'beads':
            raise ValueError('mode == {} does not support \
                         multiple backgrounds'.format(mode))
        elif mode not in modes:
            raise ValueError('Input mode not supported.')

        # subtract a background estimation if it wasn't done prior
        if subtraction:
            if event_idx is not None:
                bckg = np.delete(X, event_idx, axis=0)
                X = X[event_idx]
            else:
                # estimate the baseline from the event itself
                bckg = X.copy()
            bckg = self._estimate(bckg, mode)
            # ensure no negative counts
            X = (X-bckg).clip(min=0)

        # estimate a background/baseline if multiple spectra are provided
        # if mode == 'beads':
        #     warnings.warn('mode == {} assumes X_bckg has already \
        #         undergone BEADS estimation.'.format(mode))
        if X_bckg.ndim > 1 and mode != 'beads':
            X_bckg = self._estimate(X_bckg, mode)

        # ensure no negative counts
        return (X + X_bckg).clip(min=0)

    def resample(self, X):
        '''
        Resamples spectra according to a Poisson distribution.
        Gamma radiation detection is approximately Poissonian.
        Each energy bin of a spectrum could be resampled using the original
        count-rate, lambda_i, as the statistical parameter for a distribution:
        Pois_i(lambda_i). Randomly sampling from this distribution would
        provide a new count-rate for that energy bin that is influenced, or
        augmented, by the original sample.

        Inputs:
        X: array-like; can be a vector of one spectrum, a matrix of many
            matrices (rows: spectra, cols: instances), or a subset of either.
            X serves as the statistical parameters for each distribution.
        Return:
        augmentation: array-like, same shape as X; the augmented spectra using
            channel resampling (see above)
        '''

        # lambda = n*p using constant probability
        p = 0.5
        n = X / p
        # augmentation = np.random.poisson(lam=X)
        # using binomial distribution for accurate low-count sampling
        augmentation = np.random.binomial(n=n.astype(int), p=p)

        return augmentation

    def sig2bckg(self, X, X_bckg, r=(0.5, 2.), subtraction=False,
                 event_idx=None, mode='mean'):
        '''
        Estimate and subtract background and scale signal-to-noise of event
        signature. The return is a spectrum with an estimated background and
        a perturbed signal intensity.
        Scaling ratio is 1/r^2. Therefore, r<1 makes the signal more intense
        and r>1 makes the signal smaller.
        X is assumed to be background adjusted by default. That is, it should
        have its original background already removed.
        If subtraction=True and event_idx is not None, X should be 2D.
        event_idx indicates the row in X that indicates the event spectrum.
        The other rows in X are then used to estimate a background distribution
        for subtraction.
        NOTE: Two background subtraction modes are supported: 'min' and 'mean'.
        'min': take the minimum gross count-rate spectrum from X.
        'mean': take the average count-rate for each bin from X.

        Inputs:
        X: array-like; If 1D, taken as an event spectrum (must be previously
            background-subtracted). If 2D, subtraction=True, and event_idx not
            None, a background subtraction is conducted prior to
            superimposition.
        X_bckg: array-like; If 1D, add this spectrum to the event spectrum of X
            as the superimposed background. If 2D, use the mode input to
            complete background superimposition.
        r: tuple; [min, max) scaling ratio. Default values ensure random
            scaling that is no more than 2x larger or smaller than the original
            signal. See numpy.random.uniform for information on interval.
            NOTE: Enforce a specific value with (r1, r2) where r1=r2.
        subtraction: bool; If True, conduct background subtraction on X
            (event_idx must not be None)
        event_idx: int(p); row index for event spectrum in X used for
            background subtraction.
        mode: str; two background subtraction modes are currently supported:
            'min': take the minimum gross count-rate spectrum from X.
            'mean': take the average count-rate for each bin from X.
        '''

        X = X.copy()
        modes = ['min', 'mean', 'beads']
        # input error checks
        if subtraction and event_idx is None and X.ndim > 1:
            raise ValueError('If subtraction=True and len(X)>1, \
                        event_idx must be specified.')
        elif subtraction and event_idx is not None and X.ndim <= 1:
            raise ValueError('X must be 2D to do background subtraction.')
        elif X_bckg.ndim > 1 and mode == 'beads':
            raise ValueError('mode == {} does not support \
                         multiple backgrounds'.format(mode))
        elif mode not in modes:
            raise ValueError('Input mode not supported.')

        if r[0] <= 0 or r[1] <= 0:
            raise ValueError('{} must be positive.'.format(r))

        # subtract a background estimation if it wasn't done prior
        if subtraction:
            if event_idx is not None:
                bckg = np.delete(X, event_idx, axis=0)
                X = X[event_idx]
            else:
                # estimate the baseline from the event itself
                bckg = X.copy()
            bckg = self._estimate(bckg, mode)
            # ensure no negative counts
            X = (X-bckg).clip(min=0)

        # estimate a background/baseline if multiple spectra are provided
        # if mode == 'beads':
        #     warnings.warn('mode == {} assumes X_bckg has already \
        #         undergone BEADS estimation.'.format(mode))
        if X_bckg.ndim > 1 and mode != 'beads':
            X_bckg = self._estimate(X_bckg, mode)

        # even random choice between upscaling and downscaling
        r = loguniform.rvs(r[0], r[1], size=1)
        X *= r

        # ensure no negative counts
        return (X + X_bckg).clip(min=0)

    def _gauss(self, x, amp, mu, sigma):
        '''
        Fit equation for a Gaussian distribution.

        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        '''

        return amp * np.exp(-((x - mu) / 4 / sigma)**2)

    def _emg(self, x, amp, mu, sigma, tau):
        """
        Exponentially Modifed Gaussian (for small tau). See:
        https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        tau: float; exponent relaxation time
        """

        term1 = np.exp(-0.5 * np.power((x - mu) / sigma, 2))
        term2 = 1 + (((x - mu) * tau) / sigma**2)
        return amp * term1 / term2

    def _lingauss(self, x, amp, mu, sigma, m, b):
        '''
        Includes a linear term to the above function. Used for modeling
        (assumption) linear background on either shoulder of a gamma photopeak.

        Inputs:
        x: array-like; 1D spectrum array of count-rates
        amp: float; amplitude = A/sigma*sqrt(2*pi)
        mu: float; mean
        sigma: float; standard deviation
        m: float; linear slope for background/baseline
        b: float; y-intercept for background/baseline
        '''

        return amp * np.exp(-0.5 * np.power((x - mu) / sigma, 2.)) + m*x + b

    def _fit(self, roi, X):
        '''
        Fit function used by resolution() for fitting a Gaussian function
        on top of a linear background in a specified region of interest.
        TODO: Add a threshold for fit 'goodness.' Return -1 if failed.

        Inputs:
        roi: tuple; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        '''

        # binning of data (default usually 0->1000 bins)
        ch = np.arange(0, len(X))
        region = X[roi[0]:roi[1]]

        # initial guess for fit
        max_y = np.max(region)
        max_z = ch[roi[0]:roi[1]][np.argmax(region)]
        # [amp, mu, sigma, m, b]
        p0 = [max_y, max_z, 1., 0, X[roi[0]]]

        # prevents nonsensical fit parameters (fail otherwise)
        lower_bound = [0, 0, 0, -np.inf, -np.inf]
        upper_bound = [np.inf, X.shape[0]-1, np.inf, np.inf, np.inf]
        bounds = (lower_bound, upper_bound)
        coeff, var_matrix = curve_fit(self._lingauss,
                                      ch[roi[0]:roi[1]],
                                      region,
                                      p0=p0,
                                      bounds=bounds)

        return coeff

        # # as calculated exactly from Gaussian statistics
        # fwhm = 2*np.sqrt(2*np.log(2))*coeff[1]
        # return fwhm

    def _crude_bckg(self, roi, X):
        '''
        Linear estimation of background using the bounds of an ROI.
        Uses point-slope formula and the bounds for the ROI region to create
        an array of the expected background.

        Inputs:
        roi: tuple; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        '''

        lower_bound = roi[0]
        upper_bound = roi[1]

        y1 = X[lower_bound]
        y2 = X[upper_bound]
        slope = (y2 - y1) / (upper_bound - lower_bound)

        y = slope * (np.arange(lower_bound, upper_bound) - lower_bound) + y1

        return y, slope, y1

    def _escape_int(self, E):
        '''
        Computes the ratio of escape peak/photopeak intensity as
        a function of photopeak energy (> 1.022 MeV).
        This is roughly estimated from two papers:
        - 10.1016/0029-554X(73)90186-9
        - 10.13182/NT11-A12285
        Three values are eye-estimated and polynomially fitted using
        Wolfram Alpha. This is a crude computation with poorly vetted
        papers working with HPGe (rather than the typical NaI) detectors.
        NOTE: This breaks down for E>~4 MeV, the ratio will grow > 1.
            For E<~1.3MeV, the polynomial starts to increase again, but at
            a very low intensity for such low energy gammas.
        TODO: find better sources or a better method for intensity estimation.

        Inputs:
        E: float; energy of photopeak
        '''

        return (8.63095e-8*E**2) - (0.000209524*E) + 0.136518

    def nuclear(self, roi, X, escape, binE=3.,
                width=None, counts=None, subtract=False):
        '''
        Inject different nuclear interactions into the spectrum.
        Current functionality allows for the introduction of either escape
        peaks or entirely new photopeaks (ignoring Compton continuum).
        Width and counts relationship for escape and photo-peaks is assumed
        to be linear across the spectrum. However, the user can specify
        width and counts as an input.

        Inputs:
        roi: list; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        escape: bool; False if adding photopeak, True if adding escape peaks.
            if True, roi must include the peak > 1,022 keV to introduce peaks.
        binE: float; Energy/Channel ratio for spectrum. Necessary for computing
            escape peak relationships.
        width: float; width, in channels, of peak to introduce. Technically
            defined as the standard deviation for the distribution
            (see numpy.random.normal documentation).
        counts: int; number of counts to introduce in peak (i.e. intensity).
        '''

        # assumes the center of the normal distribution is the ROI center
        b = np.mean(roi)
        E = b*binE
        # escape peak error to ensure physics
        if escape and E < 1022:
            raise ValueError('Photopeaks below 1,022 keV ',
                             'do not produce escape peaks.')
        # avoid overwriting original data
        nX = X.copy()
        bins = nX.shape[0]

        # find (photo)peaks with heights above baseline of at least 10 counts
        # ignoring low-energy distribution typically residing in first 100 bins
        peaks, properties = find_peaks(X[100:],
                                       prominence=20,
                                       width=4,
                                       rel_height=0.5)
        # find the tallest peak to estimate energy resolution
        # remember to shift the peak found by the 100-bin mask
        fit_peak = peaks[np.argsort(properties['prominences'])[-1]]+100
        # fit ROI to estimate representative peak counts
        w = int(len(nX)*0.1)
        fit_roi = [max(fit_peak-w, 0), min(fit_peak+w, bins-1)]
        # fit the most prominent peak
        # [amp, mu, sigma, m, b]
        coeff = self._fit(fit_roi, nX)
        amp, sigma = coeff[0], coeff[2]
        # assume linear relationship in peak counts and width over spectrum
        # width is approximately a delta fnct. at the beginning of the spectrum
        # counts are approximately zero by the end of the spectrum
        slope_sigma = sigma/fit_peak
        # slope_counts should be negative because fit_peak < bins
        slope_counts = np.sqrt(2*np.pi) * amp / (fit_peak - bins)
        # avoid bad fits from adding an exponential amount of counts
        max_counts = min(-slope_counts * bins,
                         np.sqrt(np.sum(nX)))

        # insert peak at input energy
        if not escape:
            # approximate width and counts from relationship estimated above
            sigma_peak = slope_sigma * b
            # avoid bad fits from adding an exponential amount of counts
            cts_peak = min(np.absolute(max_counts - (slope_counts * b)),
                           np.sqrt(np.sum(nX)))
            # overwrite if user input is given
            if width is not None:
                sigma_peak = width
            if counts is not None:
                cts_peak = counts
            # create another spectrum with only the peak
            new_peak, _ = np.histogram(np.round(
                                        np.random.normal(loc=b,
                                                         scale=sigma_peak,
                                                         size=int(cts_peak))),
                                       bins=bins,
                                       range=(0, bins))
            if subtract:
                nX = nX-new_peak
            else:
                nX = nX+new_peak
        # insert escape peaks if specified or physically realistic
        if escape or (E >= 1022 and not subtract):
            # fit the peak at input energy
            # [amp, mu, sigma, m, b]
            coeff = self._fit(roi, nX)
            # background counts integral
            bckg_width = roi[1] - roi[0]
            background = (coeff[3]/2)*(roi[1]**2
                                       - roi[0]**2) + coeff[4] * (bckg_width)
            # find difference from background
            peak_counts = np.sum(nX[roi[0]:roi[1]]) - background

            # normal distribution parameters for escape peaks
            b_single = int((E-511)/binE)
            sigma_single = slope_sigma * b_single
            b_double = int((E-1022)/binE)
            sigma_double = slope_sigma * b_double
            # escape peak intensity estimated as a function of E
            cts = self._escape_int(E)*peak_counts
            # overwrite if user input is given
            if width is not None:
                sigma_single = sigma_double = width
            if counts is not None:
                cts = counts

            # create a blank spectrum with only the escape peak
            single, _ = np.histogram(np.round(
                                      np.random.normal(loc=b_single,
                                                       scale=sigma_single,
                                                       size=int(cts))),
                                     bins=bins,
                                     range=(0, bins))
            double, _ = np.histogram(np.round(
                                      np.random.normal(loc=b_double,
                                                       scale=sigma_double,
                                                       size=int(cts))),
                                     bins=bins,
                                     range=(0, bins))
            if subtract:
                nX = nX-single-double
            else:
                nX = nX+single+double
        return nX

    def find_res(self, X, width=4, roi_perc=0.03):
        '''
        Automatically find reasonable peaks in a spectrum and return one.
        This can be used to randomly find a peak to perturb via resolution.
        Uses BEADS to identify peaks in a spectrum.
        Note that both BEADS and scipy.signals.find_peaks can be very unstable
        and thus this is not always reliable. It is recommended to check for
        reasonable peaks or fits/augmentations after using this method.

        Inputs:
        X: array-like; 1D spectrum array of count-rates
        width: int; minimum channel width for an identified peak
            (see scipy.signals.find_peaks for more information)
        roi_perc: float; percent of total channels in X to have on each
            shoulder of an ROI.
        '''

        beads_results = beads(X, **self.BEADS_PARAMS)
        # np.clip(min=0) ensures no negative counts when finding peaks
        peaks, _ = find_peaks(beads_results[0].clip(min=0),
                              width=width,
                              rel_height=0.5)
        choice = np.random.choice(peaks, 1)
        w = int(len(X)*roi_perc)
        roi = [int(max(choice-w, 0)), int(min(choice+w, len(X)-1))]
        return roi

    def resolution(self, roi, X, multiplier=1.5, conserve=True):
        '''
        Manipulate the resolution, or width, of a photopeak as measured by
        the full-width at half-maximum (FWHM).
        In terms of reasonable values for multiplier, be cautious for
        values >> 1. Wider peaks will overwrite a wider area of the spectrum.
        Note that sometimes the interplay between a tighter or wider ROI
        (which determines the region to fit) and the size of the multiplier
        can affect the shape of the resulting peak.

        Inputs:
        roi: list; (min, max) bin/index values for region of interest - used
            to index from data, X
        X: array-like; 1D spectrum array of count-rates
        multiplier: float; scaler to manipulate FWHM by. Greater than 1
            widens the peak and vice versa.
        conserve: bool; if True, peak counts will be conserved after
            augmentation, meaning a taller peak for multipler<1 & vice versa
        '''

        if multiplier <= 0:
            raise ValueError('{} must be positive.'.format(multiplier))

        # avoid overwriting original data
        X = X.copy()
        if multiplier < 0:
            multiplier = 1/abs(multiplier)

        # [amp, mu, sigma, m, b]
        coeff = self._fit(roi, X)
        # amp = coeff[0]
        fwhm = 2*np.sqrt(2*np.log(2))*coeff[2]
        new_sigma = multiplier * fwhm / (2*np.sqrt(2*np.log(2)))
        coeff[2] = new_sigma

        # there's no need to refind background/baseline
        # because it was fit in coeff above
        # but this could be used to isolate background
        # y, m, b = self._crude_bckg(roi, X)

        # expanding ROI if new peak is too wide
        # 6-sigma ensures the entire Gaussian distribution is captured
        # NOTE: this is unstable, new peaks (and the background/baseline)
        # can overwrite other spectral features, should it be removed?
        if 2*new_sigma >= roi[1]-roi[0]:
            # maximum expansion cannot be more than length of spectrum
            roi[0] = max(0, roi[0]-int(new_sigma))
            roi[1] = min(X.shape[0]-1, roi[1]+int(new_sigma))

        ch = np.arange(roi[0], roi[1])
        peak = self._lingauss(ch,
                              amp=coeff[0],
                              mu=coeff[1],
                              sigma=new_sigma,
                              m=coeff[3],
                              b=coeff[4])
        if conserve:
            # only counts from background
            background = (coeff[3]*ch + coeff[4]).clip(min=0)
            # only counts from old peak
            old_cts = (X[ch] - background).clip(min=0)
            # only counts from new peak
            new_cts = (peak - background).clip(min=0)
            # scale new peak to conserve original counts
            if np.sum(new_cts) > 0:
                peak = (new_cts*(np.sum(old_cts)/np.sum(new_cts))) + background

        # normalize to conserve relative count-rate
        # NOTE: this is realistic physically, but is it necesary?
        # peak = peak * (np.sum(X[roi[0]:roi[1]]) / np.sum(peak))

        # add noise to the otherwise smooth transformation
        # .clip() necessary so counts are not negative
        # .astype(float) avoids ValueError: lam value too large
        peak = self.resample(peak.clip(min=0).astype(np.float64))
        X[roi[0]:roi[1]] = peak
        return X

    def mask(self, X, mode='random', interval=5, block=(0, 100)):
        '''
        Mask specific regions of a spectrum to force feature importance.
        This may or may not be physically realistic, depending on the masking
        scenario (e.g. pileup) but it represents a common image augmentation.
        NOTE: the default values for interval and block are not used, but
        recommended sizes or degrees for reasonable augmentations.

        Inputs:
        X: array-like; should be 1D, i.e. one spectrum to be augmented
        mode: str; three modes are supported:
            'interval': mask every interval's channel
            'block': mask everything within a block range
            'both': mask every interval's channel within a block range
            'random': randomly pick one of the above
        interval: int; mask every [this int] channel in the spectrum
        block: tuple; spectral range to mask (assumed spectral length is
            1000 channels)
        '''

        # avoid overwriting original data
        X = X.copy()

        modes = ['random', 'interval', 'block', 'both']
        if mode not in modes:
            raise ValueError('Input mode not supported.')
        if mode == 'random':
            mode = np.random.choice(modes)
            if mode == 'interval':
                # high => exclusive: 10+1
                interval = np.random.randint(1, 11)
            elif mode == 'block':
                # default spectral length is 1,000 channels
                # TODO: abstract spectral length
                low = np.random.randint(0, 999)
                # default block width is low+10 to max length
                # TODO: abstract block width
                high = np.random.randint(low+10, 1000)
                block = (low, high)

        # mask spectrum (i.e. set values to 0)
        if mode == 'interval':
            X[::interval] = 0
        elif mode == 'block':
            X[block[0]:block[1]] = 0
        elif mode == 'both':
            X[block[0]:block[1]:interval] = 0

        return X

    def _ResampleLinear1D(self, original, targetLen):
        '''
        Originally from StackOverflow:
        https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
        Upsamples or downsamples an array by interpolating
        the value in each bin to a given length.

        Inputs:
        original: array-like; spectrum or array to be resampled
        targetLen: int; target length to resize/resample array
        '''

        original = np.array(original, dtype=float)
        index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=float)
        # find the floor (round-down) for each bin (cutting off with int)
        index_floor = np.array(index_arr, dtype=int)
        # find the ceiling (max/round-up) for each bin
        index_ceil = index_floor + 1
        # compute the difference/remainder
        index_rem = index_arr - index_floor

        val1 = original[index_floor]
        val2 = original[index_ceil % len(original)]
        # interpolate the new value for each new bin
        interp = val1 * (1.0-index_rem) + val2 * index_rem
        assert (len(interp) == targetLen)
        return interp

    def _Poisson1D(self, X, lam):
        '''
        Apply positive gain shift by randomly distributing counts in each bin
        according to a Poisson distribution with parameter lam.
        The random Poisson distribution results in a spectrum that can have a
        slightly different distribution of counts rather than the uniform
        deformation of _ResampleLinear1D.
        The drift is energy dependent (i.e. more drift for higher energies).
        This mode only supports positive gain shift.

        Inputs:
        X: array-like; 1D spectrum, with count-rate for each channel
        lam: float; Poisson parameter for gain drift. Determines the severity
            of gain drift in spectrum.
        '''

        new_ct = X.copy()
        for i, c in enumerate(X):
            # randomly sample a new assigned index for every count in bin
            # using np.unique, summarize which index each count goes to
            idx, nc = np.unique(np.round(
                                    np.random.poisson(lam=lam*(i/X.shape[0]),
                                                      size=int(c))),
                                return_counts=True)
            # check to see if any indices are greater than the spectral length
            missing_idx = np.count_nonzero(i+idx >= new_ct.shape[0])
            if missing_idx > 0:
                # add blank bins if so
                new_ct = np.append(new_ct,
                                   np.repeat(0,
                                             np.max(idx)+i-new_ct.shape[0]+1))
            # distribute all counts according to their poisson index
            new_ct[(i+idx).astype(int)] += nc
            # adjust for double-counting
            new_ct[i] -= np.sum(nc)

        return new_ct

    def gain_shift(self, X, bins=None, lam=np.random.uniform(-5, 5),
                   k=0, mode='resample'):
        '''
        Modulate the gain-shift underlying a spectrum.
        This simulates a change in the voltage to channel mapping, which
        will affect how the spectral shape appears in channel vs. energy space.
        If a positive gain shift occurs (multiplier increases), e.g. 1V=1ch
        becomes 0.9V=1ch, spectral features will stretch out and widen across
        the spectrum. Vice versa for a negative gain shift.
        Qualitatively, a positive drift manifests in a smeared or stretched
        spectrum with wider peaks whereas a negative drift manifests in a
        squeezed or tightened spectrum with narrower peaks.
        Both a positive and negative gain drift are supported, however only
        mode='resample' supports negative drift.

        Inputs:
        X: array-like; 1D spectrum, with count-rate for each channel
        bins: array-like; 1D vector (with length len(counts)+1) of either
            bin edges in energy space or channel numbers.
        lam: float; Poisson parameter for gain drift. Determines the severity
            of gain drift in spectrum.
        k: int; number of bins to shift the entire spectrum by
        mode: str; two possible gain shift algorithms can be used
            'resample': linearly resample the spectrum according to a new
            length (lam), evenly redistributing the counts.
            'poisson': statistically/randomly resample the counts in each bin
            according to a poisson distribution of parameter lam.
            NOTE: 'poisson' only works in the positive direction.
            TODO: Future feature implementation should probably focus
                just on the rebinning algorithm, since it is simpler
                and can work in both directions.
        '''

        modes = ['resample', 'poisson']
        if mode not in modes:
            raise ValueError('{} is not a supported algorithm.'.format(mode))
        if len(X.shape) > 1:
            raise ValueError(f'gain_shift expects only 1 spectrum (i.e. 1D \
                               vector) but {X.shape[0]} were passed')

        # gain-shift algorithm
        # add blank bins before or after the spectrum
        if k < 0:
            X = np.append(X, np.repeat(0., np.absolute(k)))
            X[0] = np.sum(X[:np.absolute(k)])
            X = np.delete(X, np.arange(1, np.absolute(k)))
            # fix the length of the spectrum to be the same as before
            if bins is not None:
                bins = np.linspace(bins[0], bins[-1], X.shape[0]+1)
        elif k > 0:
            X = np.insert(X, 0, np.repeat(0., k))
            # fix the length of the spectrum to be the same as before
            if bins is not None:
                width = bins[1] - bins[0]
                bins = np.arange(bins[0], bins[-1]+(k*width), width)

        # only a direct bin shift is desired
        if lam == 0.:
            return X, bins
        # gain-drift algorithm(s)
        elif mode == 'resample' or (mode == 'poisson' and lam < 0):
            # second condition needed since 'poisson' does not support
            # negative gain drift (lam < 0)
            new_ct = self._ResampleLinear1D(X, int(X.shape[0]+lam))
        elif mode == 'poisson':
            # recalculate binning if passed
            new_ct = self._Poisson1D(X, abs(lam))

        # enforce the same count-rate
        new_ct *= np.sum(X)/np.sum(new_ct)

        # compute bins if passed
        if bins is not None:
            width = bins[1] - bins[0]
            new_b = np.arange(bins[0],
                              bins[0]+((len(new_ct)+1)*width),
                              width)
            return new_ct, new_b
        else:
            return new_ct, bins
