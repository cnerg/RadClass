import numpy as np
import time
import logging
import progressbar

import RadClass.DataSet as ds


class RadClass:
    '''
    Bulk handler class. Contains most functions needed for processing data
        stream files and pass along to an analysis object.

    Attributes:
    running: indicates whether the end of a file has been reached and thus end.
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
    store_data: boolean; if true, save the results to a CSV file.
    cache_size: Optional parameter to reduce file I/O and therefore
        increase performance. Indexes a larger selection of rows to analyze.
        cache_size -must- be greater than integration above.
        If not provided (None), cache_size is ignored (equals integration).
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
    labels: list of dataset name labels in this order:
        [ live_label: live dataset name in HDF5 file,
          timestamps_label: timestamps dataset name in HDF5 file,
          spectra_label: spectra dataset name in HDF5 file ]
    '''

    def __init__(self, stride, integration, datapath, filename, analysis=None,
                 store_data=True, cache_size=None, start_time=None,
                 stop_time=None,
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

        # analysis object that will manipulate data
        self.analysis = analysis

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

        # parameters for keeping track of progress through a file
        self.start_i = 0
        self.stop_i = len(self.processor.timestamps) - 1

        if self.start_time is not None:
            timestamp = self.processor.timestamps[self.processor.timestamps >=
                                                  self.start_time]
            # check if any timestamps were found
            if timestamp.size == 0:
                raise ValueError('User start time is larger than all timestamps.')
            else:
                timestamp = timestamp[0]
            self.start_i = max(np.searchsorted(self.processor.timestamps,
                                               timestamp,
                                               side='right')-1, 0)
        self.start_time = self.processor.timestamps[self.start_i]
        self.current_i = self.start_i

        if self.stop_time is not None:
            if self.stop_time <= self.processor.timestamps[0]:
                raise ValueError('User stop time is no greater than the first timestamp.')
            elif self.start_time >= self.stop_time:
                raise ValueError('User start time is larger than stop time.')

            timestamp = self.processor.timestamps[self.processor.timestamps >=
                                                  self.stop_time]
            # check if any timestamps were found
            if timestamp.size == 0:
                timestamp = self.processor.timestamps[-1]
            else:
                timestamp = timestamp[0]
            self.stop_i = np.searchsorted(self.processor.timestamps,
                                          timestamp,
                                          side='right')-1
        self.stop_time = self.processor.timestamps[self.stop_i]

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
        while running:
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
                                   np.arange(len(self.cache[0])).astype(str))
                header = ', '.join(col for col in header)
            np.savetxt(fname=f,
                       X=self.storage,
                       delimiter=',',
                       header=header)
