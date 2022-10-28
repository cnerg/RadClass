import numpy as np
import h5py
import time
import logging
import progressbar

import RadClass.DataSet as ds


class RadClass:
    '''
    Bulk handler class. Contains most functions needed for processing data
        stream files and pass along to an analysis object.

    Inputs:
    stride: The number of seconds (which equals the number of indexes
        to advance after analyzing an integration interval. e.g. if
        stride = 3, then the next integration period will setp 3 seconds ahead.
        If stride = integration, all observations will be analyzed once.
    integration: The number of seconds (which equals the number of indexes)
        indexes) to integrate the spectrum over. e.g. if integration = 5, then
        the next 5 rows (corresponding to the next 5 seconds) of data will be
        pulled and summed/integrated to analyze.
    datapath: HDF5 group/subgroup path to dataset, including prefix,
        node ID, and analysis element. e.g. '/path/to/dataset'
    filename: The filename for the data to be analyzed.
    TODO: Make node and filename -lists- so that multiple nodes and files
        can be processed by the same object.
    start_time: Unix epoch timestamp, in units of seconds. All data in filename
        at and after this point will be analyzed. Useful for processing only
        a portion of a data file. Default: None, ignored.
    stop_time: Unix epoch timestamp, in units of seconds. All data in filename
        earlier and up to stop_time will be analyzed. Useful for processing
        only a portion of a data file. Default: None, ignored.
    NOTE: To convert from string to epoch, try:
    time.mktime(datetime.datetime.strptime(x, "%m/%d/%Y %H:%M:%S").timetuple())
    Where "%m/%d/%Y %H:%M:%S" is the string format (see datetime docs).
    Requires time and datetime modules
    store_data: boolean; if true, save the resulting data to a numpy array.
        This array can then be written using RadClass.write(). This must be
        True for a post_analysis object to run.
    analysis: AnalysisParameters object (e.g. H0 or BackgroundEstimator) that
        will run iteratively as RadClass processes data. That is, for each
        sample, that processed data slice will be passed to the analysis object
        using its analysis.run() method.
    post_analysis: AnalysisParameters object (e.g. DiffSpectra) that runs with
        all data after it has been analyzed by RadClass. That is, all data
        rows/samples will be analyzed simultaneously using this object's
        post_analysis.run() method.
    cache_size: Optional parameter to reduce file I/O and therefore
        increase performance. Indexes a larger selection of rows to analyze.
        cache_size -must- be greater than integration above.
        If not provided (None), cache_size is ignored (equals integration).
    labels: list of dataset name labels in this order:
        [ live_label: live dataset name in HDF5 file,
          timestamps_label: timestamps dataset name in HDF5 file,
          spectra_label: spectra dataset name in HDF5 file ]
    '''

    def __init__(self, stride, integration, datapath, filename, start_time=None,
                 stop_time=None, analysis=None, post_analysis=None,
                 store_data=True, cache_size=None,
                 labels={'live': '2x4x16LiveTimes',
                         'timestamps': '2x4x16Times',
                         'spectra': '2x4x16Spectra'}):
        self.stride = stride
        self.integration = integration
        self.datapath = datapath
        self.filename = filename
        self.store_data = store_data

        if cache_size is None:
            self.cache_size = self.integration
        else:
            self.cache_size = cache_size

        self.start_time = start_time
        self.stop_time = stop_time
        self.labels = labels

        # analysis object that will manipulate data iteratively
        self.analysis = analysis
        # analysis object that will manipulate analyzed data
        self.post_analysis = post_analysis

    def queue_file(self):
        '''
        Initialize a file for analysis using the load_data.py scripts.
        Collects all information and assumes it exists in the file.
        NOTE: queue_file will include all timestamps within the range of the
        user's start & stop, but not exceed it. e.g. if the user asks for a
        start timestamp of 2.5, and timestamps 2 and 3 are present, then
        3 will be chosen as the start-time so as not to include data prior
        to the user's 2.5. Likewise, RadClass will only include data less than
        the user's stop-time

        Attributes:
        processor: DataSet object responsible for indexing data from file.
        current_i/start_i: Used in indexing rows for analysis. May be different
            than the first idx=0 if start_time is specified.
        stop_i: Last index of timestamps. May be different if stop_time
            is specified.
        '''

        self.processor = ds.DataSet(self.labels)
        self.processor.init_database(self.filename, self.datapath)

        # possible invalid user input scenarios
        # user start time is great than the last possible data timestamp
        if self.start_time is not None and self.start_time > self.processor.timestamps[-1]:
            raise ValueError('User start time is larger than all timestamps.')
        # user stop time is less than the first possible timestamp
        elif self.stop_time is not None and self.stop_time < self.processor.timestamps[0]:
                raise ValueError('User stop time is less than the first timestamp.')
        # user start time is greater than user stop time
        elif self.start_time is not None and self.stop_time is not None and self.start_time > self.stop_time:
            raise ValueError('User start time is larger than stop time.')

        # parameters for keeping track of progress through a file
        self.start_i = 0
        self.stop_i = len(self.processor.timestamps) - 1

        if self.start_time is not None:
            self.start_i = max(np.searchsorted(self.processor.timestamps,
                                               self.start_time,
                                               side='right')-1, 0)
        self.start_time = self.processor.timestamps[self.start_i]
        self.current_i = self.start_i

        if self.stop_time is not None:
            self.stop_i = np.searchsorted(self.processor.timestamps,
                                          self.stop_time,
                                          side='right')-1
        self.stop_time = self.processor.timestamps[self.stop_i]

        # this check must be done after RadClass finds dataset compatible
        # timestamps for any user start or stop time, since they may be
        # different than the precise input timestamp
        # if true, no data will be processed but RadClass will continue
        if self.start_time == self.stop_time:
            logging.info("start_time=stop_time=%d", self.start_time)
            raise RuntimeWarning('Start and Stop times used are the same, no data processed.')

    def collapse_data(self, rows_idx):
        '''
        Integrates a subset of data from the total data matrix given some
        integration time. Utilizes data_slice() from DataSet to extract
        the requisite rows of data and then numpy.sum(data,axis=0) to combine
        all rows. Assumes negligible drift and accurate energy calibration.

        NOTE: Returned data is normalized for live times (1s measurement
        period less dead time). i.e. The returned data is a count rate at
        each bin.

        Return: 1D numpy array for integration interval
        '''

        # extract requisite data rows
        # normalize by live times to produce count rate data
        # processor.live can be indexed by appropriate timestamps
        # (i.e. row indices).
        data_matrix = (self.cache[rows_idx-self.cache_idx[0]] /
                       self.processor.live[rows_idx][:, None])

        # utilizes numpy architecture to sum data
        total = np.sum(data_matrix, axis=0)

        return total

    def collect_rows(self):
        '''
        Uses the integration duration to collect all rows needed for analysis.
        Uses numpy.arange() to enumerate row indices from start to end.

        Return: list of rows for data indexing
        '''

        # NOTE: this behavior is currently disabled by the if-statement in
        # march() removing the 'or' portion with allow this to work
        #
        # if the final portion of the file is smaller than a full integration
        # interval, only what is left is collected for this analysis
        end_i = min(self.current_i + self.integration,
                    len(self.processor.timestamps) - 1)

        # enumerate number of rows to integrate exclusive of the endpoint
        rows_idx = np.arange(self.current_i, end_i)

        # check if all rows are stored in cache by checking for last row
        if end_i not in self.cache_idx:
            self.run_cache()

        return rows_idx

    def march(self):
        '''
        Advance a step using stride time to find the new working integration
        interval. Also keeps track of the EOF state but checking for
        out-of-index situations.

        Returns: boolean for EOF
        '''

        # increment the working interval starting index and advance stride
        new_i = self.current_i + self.stride

        # stop analysis if EOF reached
        # NOTE: stops prematurely, for windows of full integration only
        running = True
        if (new_i + self.integration) >= self.stop_i:
            running = False

        if running:
            # update working integration interval timestep
            self.current_i = new_i

        return running

    def run_cache(self):
        '''
        Updates RadClass.cache to contain the requisite rows needed for
        integration. Tracked via indices and cache_size.
        '''
        end_i = min(self.current_i + self.cache_size,
                    len(self.processor.timestamps) - 1)
        # enumerate number of rows to integrate exclusive of the endpoint
        self.cache_idx = np.arange(self.current_i, end_i)
        self.cache = self.processor.data_slice(self.datapath, self.cache_idx)

    def iterate(self):
        '''
        Full iteration over the entirety of the data file. Runs
        until EOF reached. Prints progress over the course of the analysis.
        Only runs for a set node (datapath) with data already queued.
        '''
        pbar = progressbar.ProgressBar(max_value=100, redirect_stdout=True)
        inverse_dt = 1.0 / (self.stop_i - self.start_i)

        # number of samples analyzed between log updates
        log_interval = max(min((self.stop_i - self.start_i)/100, 100), 10)
        running = True  # tracks whether to end analysis
        while running and (self.start_i != self.stop_i):
            # print status at set intervals
            if (self.current_i - self.start_i) % log_interval == 0:
                pbar.update(round((self.current_i - self.start_i) * inverse_dt * 100, 4))

                current_time = self.processor.timestamps[self.current_i]
                readable_time = time.strftime('%m/%d/%Y %H:%M:%S',  time.gmtime(current_time))
                logging.info("--\tCurrently working on timestamps: %d\n", readable_time)

            # execute analysis and advance in stride
            rows_idx = self.collect_rows()
            data = self.collapse_data(rows_idx)

            # pass data to analysis object if available
            if self.analysis is not None:
                self.analysis.run(data,
                                  self.processor.timestamps[self.current_i])
            if self.store_data:
                self.storage[self.processor.timestamps[self.current_i]] = data

            running = self.march()

        # print completion summary
        logging.info("\n...Complete...")
        logging.info("Finished analyzing %s.\n\tNumber of observations analyzed: %d", self.filename, len(self.processor.timestamps))

    def run_all(self):
        '''
        Run all analysis for all nodes in initialization. This serves as an
        easy way to run the whole analysis rather than individually executing
        class methods. Although either way is valid and manual execution
        may be better for debugging and development.
        '''

        self.storage = dict()

        self.queue_file()
        # initialize cache
        self.run_cache()
        self.iterate()

        # do not convert to numpy array if empty
        if bool(self.storage):
            self.storage = np.insert(arr=np.array(list(self.storage.values())),
                                     obj=0,
                                     values=list(self.storage.keys()),
                                     axis=1)

            if self.post_analysis is not None:
                self.post_analysis.run(self.storage)

    def write(self, filename):
        '''
        Write results to file using h5py, similar to MINOS file structure.
        filename should not include the file extension.
        '''
        with h5py.File(filename+'.h5', 'a') as f:
            keys = ['timestamps', 'spectra']
            # build/include header if file is new
            if len(f.keys()) == 0:
                # only chunked datasets can be resized,
                #  so chunks must be initialized
                f.create_dataset(keys[0],
                                 self.storage[:, 0].shape,
                                 data=self.storage[:, 0],
                                 maxshape=(None,),
                                 chunks=self.storage[:, 0].shape,
                                 dtype='float64')
                f.create_dataset(keys[1],
                                 self.storage[:, 1:].shape,
                                 data=self.storage[:, 1:],
                                 maxshape=(None, 1000),
                                 chunks=self.storage[:, 1:].shape,
                                 dtype='float64')
            else:
                dset1 = f[keys[0]]
                dset2 = f[keys[1]]

                dset1.resize(dset1.shape[0]+self.storage[:, 0].shape[0],
                             axis=0)
                dset2.resize(dset2.shape[0]+self.storage[:, 1:].shape[0],
                             axis=0)

                dset1[-self.storage[:, 0].shape[0]:] = self.storage[:, 0]
                dset2[-self.storage[:, 1:].shape[0]:] = self.storage[:, 1:]
