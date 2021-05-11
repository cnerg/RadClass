import numpy as np
import pandas as pd
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
    cache_size: (WIP) optional parameter to reduce file I/O and therefore
        increase performance. Indexes a larger selection of rows to analyze.
        If not provided (None), cache_size is ignored (equals integration).
    labels: list of dataset name labels in this order:
        [ live_label: live dataset name in HDF5 file,
          timestamps_label: timestamps dataset name in HDF5 file,
          spectra_label: spectra dataset name in HDF5 file ]
    '''

    def __init__(self, stride, integration, datapath, filename, analysis=None,
                 store_data=False, cache_size=None,
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

        self.labels = labels

        # analysis object that will manipulate data
        self.analysis = analysis

    def queue_file(self):
        '''
        Initialize a file for analysis using the load_data.py scripts.
        Collects all information and assumes it exists in the file.

        Attributes:
        processor: DataSet object responsible for indexing data from file.
        working_time: The current working epoch timestamp at which data is
            being analyzed. Used to keep track of progress within a file.
        '''

        self.processor = ds.DataSet(self.labels)
        self.processor.init_database(self.filename, self.datapath)
        # parameters for keeping track of progress through a file

        self.working_time = self.processor.timestamps[0]

    def collapse_data(self, rows):
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
        data_matrix = self.processor.data_slice(self.datapath, rows)

        # normalize by live times to produce count rate data
        # processor.live can be indexed by appropriate timestamps but
        # data_matrix only has indices for rows.
        for row in rows:
            if row not in self.cache_rows:
                self.run_cache()
            idx = np.where(self.cache_rows == row)[0][0]
            dead_time = self.processor.live[row]
            self.cache[idx] = self.cache[idx] / dead_time

        # old, more inefficient way of summing
        #total = np.zeros_like(data_matrix[0])
        #for row in data_matrix:
        #    total += row

        # utilizes numpy architecture to sum data
        total = np.sum(data_matrix, axis=0)

        return total

    def collect_rows(self):
        '''
        Uses the integration duration to collect all rows needed for analysis.
        Uses numpy.arange() to enumerate row indices from start to end.

        Return: list of rows for data indexing
        '''

        # collect start index from tracked timestamp
        start_i, = np.where(self.processor.timestamps == self.working_time)

        # NOTE: this behavior is currently disabled by the if-statement in
        # march() removing the 'or' portion with allow this to work
        #
        # if the final portion of the file is smaller than a full integration
        # interval, only what is left is collected for this analysis
        end_i = min(start_i + self.integration,
                    len(self.processor.timestamps) - 1)

        # enumerate number of rows to integrate exclusive of the endpoint
        rows = np.arange(start_i, end_i)
        return rows

    def march(self):
        '''
        Advance a step using stride time to find the new working integration
        interval. Also keeps track of the EOF state but checking for
        out-of-index situations.

        Returns: boolean for EOF
        '''

        # find the working interval starting index and advance stride
        start_i, = np.where(self.processor.timestamps == self.working_time)
        new_i = start_i[0] + self.stride

        # stop analysis if EOF reached
        # NOTE: stops prematurely, for windows of full integration only
        running = True
        if ((new_i >= len(self.processor.timestamps)) or
                ((new_i + self.integration) >= len(self.processor.timestamps))):
            running = False

        if running:
            # update working integration interval timestep
            self.working_time = self.processor.timestamps[new_i]

        return running

    def run_cache(self):
        start_i, = np.where(self.processor.timestamps == self.working_time)
        start_i = start_i[0]
        if start_i + self.cache_size >= len(self.processor.timestamps):
            end_i = len(self.processor.timestamps) - 1
        else:
            end_i = start_i + self.cache_size
        # enumerate number of rows to integrate exclusive of the endpoint
        self.cache_rows = np.arange(start_i, end_i)
        self.cache = self.processor.data_slice(self.datapath, self.cache_rows)

    def iterate(self):
        '''
        Full iteration over the entirety of the data file. Runs
        until EOF reached. Prints progress over the course of the analysis.
        Only runs for a set node (datapath) with data already queued.
        '''
        bar = progressbar.ProgressBar(max_value=100, redirect_stdout=True)
        inverse_dt = 1.0 / (self.processor.timestamps[-1] - self.processor.timestamps[0])

        log_interval = 10000  # number of samples analyzed between log updates
        running = True  # tracks whether to end analysis
        while running:
            # print status at set intervals
            if np.where(self.processor.timestamps == self.working_time)[0][0] % log_interval == 0:
                bar.update(round((self.working_time - self.processor.timestamps[0]) * inverse_dt, 4)*100)

                readable_time = time.strftime('%m/%d/%Y %H:%M:%S',  time.gmtime(self.working_time))
                logging.info("--\tCurrently working on timestamps: {}\n".format(readable_time))

            # execute analysis and advance in stride
            rows = self.collect_rows()
            data = self.collapse_data(rows)

            # pass data to analysis object if available
            if self.analysis is not None:
                self.analysis.run(data)

            if self.store_data:
                self.storage = pd.concat([self.storage, pd.DataFrame([data], index=[self.working_time])])

            running = self.march()

        # print completion summary
        logging.info("\n...Complete...")
        logging.info("Finished analyzing {}.\n\tNumber of observations analyzed: {}".format(self.filename, len(self.processor.timestamps)))

    def run_all(self):
        '''
        Run all analysis for all nodes in initialization. This serves as an
        easy way to run the whole analysis rather than individually executing
        class methods. Although either way is valid and manual execution
        may be better for debugging and development.
        '''

        if self.store_data:
            self.storage = pd.DataFrame()

        self.queue_file()
        # initialize cache
        self.run_cache()
        self.iterate()

        if self.store_data:
            self.storage.to_csv('results.csv')
