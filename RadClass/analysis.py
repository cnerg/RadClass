import numpy as np
import pandas as pd
import time

import RadClass.DataSet as ds

class RadClass:
    '''
    Bulk analysis class. Contains most functions needed for conducting an
        analysis of a MINOS MUSE file(s).

    TODO: Add detailed class information here.

    Attributes:
    regions: An iterable list of ROI classes (see ROI.py) that contains info
        on the regions to be analyzed in the spectrum.
        If this list is empty, no analysis will be conducted. Each ROI in this
        list can be analyzed by the class.
    stride: The number of seconds (which approximately equals the number of
        indexes) to advance after analyzing an integration interval. e.g. if
        stride = 3, then the next integration period will start 3 seconds ahead.
    integration: The number of seconds (which approximately equals the number of
        indexes) to integrate the spectrum over. e.g. if integration = 5, then
        the next 5 rows (corresponding to the next 5 seconds) of data will be
        pulled and summed/integrated to analyze.
    datapath: HDF5 group/subgroup path to dataset, including prefix,
                   node ID, and analysis element. e.g. '/path/to/dataset'
    filename: The filename for the MINOS MUSE data to be analyzed.
    TODO: Make node and filename -lists- so that multiple nodes and files
        can be processed by the same object.
    binning: The binning for MUSE data. This is assumed to be 3 keV bins over
        1000 channels, and is currently defaulted, but may need to be changed
        in the future as a user input (TODO)
    live_times: A numpy array of live detection times corrected for dead time.
        These are assumed to be approximately 1 second, but a TODO would be
        to discard readings that do not reach a required live time after
        dead time correction.
    timestamps: A numpy array of the epoch timestamp at which a detection was
        recorded. These are used for indexing and processing data.
    working_time: The current working epoch timestamp at which data is being
        analyzed. This is used to keep track of progress within a file.
    working_node: The MUSE node name currently being analyzed. Used for keeping
        track of progress when multiple nodes are being analyzed.
    '''

    running = True

    def __init__(self, stride, integration, datapath, filename, analysis = None,
                    store_data = False, cache_size = None,
                                        labels = {'live': '2x4x16LiveTimes',
                                                  'timestamps': '2x4x16Times',
                                                  'spectra': '2x4x16Spectra'}):
        '''
        Init for the class, all inputs are required for initialization.
        See class docstring for information on each input.
        '''

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

        # currently defaulted
        self.binning = 3

    def queue_file(self):
        '''
        Initialize a file for analysis using the load_data.py scripts.
        Collects all information and assumes it exists in the file.
        '''

        self.processor = ds.DataSet(self.labels)
        self.processor.init_database(self.filename, self.datapath)
        # parameters for keeping track of progress through a file
        
        self.working_time = self.processor.timestamps[0]

    def collapse_data(self, rows):
        '''
        Integrates a subset of data from the total MUSE data matrix given some
        integration time. Utilizes data_slice() from load_data.py to extract
        the requisite rows of data and then numpy.sum(data,axis=0) to combine
        all rows. Assumes negligible drift and accurate energy calibration.
        
        NOTE: Returned data is normalized for live times (1s measurement
        period less dead time). i.e. The returned data is a count rate at
        each bin.

        Return: 1D numpy array for integration interval
        '''

        # extract requisite data rows
        data_matrix = self.processor.data_slice(self.datapath, rows)
        #try:
        #    self.cache[rows[-1]]
        #except IndexError:
        #    print("Out of Cache!")

        # normalize by live times to produce count rate data
        #for row in range(len(rows)):
        #    data_matrix[row] = data_matrix[row] / self.processor.live[row]
        for idx, row in enumerate(rows):
            dead_time = self.processor.live[row]
            data_matrix[idx] = data_matrix[idx] / dead_time

        # old, more inefficient way of summing
        #total = np.zeros_like(data_matrix[0])
        #for row in data_matrix:
        #    total += row

        # utilizes numpy architecture to sum data
        total = np.sum(data_matrix, axis = 0)

        return total

    def collect_rows(self):
        '''
        Uses the integration duration to collect all rows needed for analysis.
        Uses numpy.arange() to enumerate row indices from start to end.

        Return: list of rows for data indexing
        '''

        # collect start index from tracked timestamp
        start_i, = np.where(self.processor.timestamps == self.working_time)

        # this behavior is currently disabled by the if-statement in march(self)
        # removing the 'or' portion with allow this to work
        #
        # if the final portion of the file is smaller than a full integration
        # interval, only what is left is collected for this analysis
        if start_i + self.integration >= len(self.processor.timestamps):
            end_i = len(self.processor.timestamps) - 1
        else:
            end_i = start_i + self.integration

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

        # find the working integration interval starting index and advance stride
        start_i, = np.where(self.processor.timestamps == self.working_time)
        start_i = start_i[0]
        new_i = start_i + self.stride

        # stop analysis if EOF reached
        # NOTE: stops prematurely, only does analysis for windows of full integration time
        if new_i >= len(self.processor.timestamps) or (new_i + self.integration) >= len(self.processor.timestamps):
            return False
        else:
            # update working integration interval timestep
            self.working_time = self.processor.timestamps[new_i]
            #print("New timestamp: {}".format(self.processor.timestamps[new_i]))
            return True

    def run_cache(self):
        start_i, = np.where(self.processor.timestamps == self.working_time)
        start_i = start_i[0]
        if start_i + self.cache_size >= len(self.processor.timestamps):
            end_i = len(self.processor.timestamps) - 1
        else:
            end_i = start_i + self.cache_size

        # enumerate number of rows to integrate exclusive of the endpoint
        cache_rows = np.arange(start_i, end_i)

        self.cache = self.processor.data_slice(self.datapath, cache_rows)

    def iterate(self):
        '''
        Full iteration over the entirety of the MINOS-MUSE data file. Runs
        until EOF reached. Prints progress over the course of the analysis. Only
        runs for a set node with data already queued into class.
        '''

        while self.running:
            # print status at set intervals
            if np.where(self.processor.timestamps == self.working_time)[0][0] % 100000 == 0:
                readable_time = time.strftime('%m/%d/%Y %H:%M:%S',  time.gmtime(self.working_time))
                print("=========================================================")
                print("Currently working on timestamps: {}\n".format(readable_time))

            #self.run_cache()
            
            # execute analysis and advance in stride
            rows = self.collect_rows()
            data = self.collapse_data(rows)

            if self.analysis is not None:
                self.analysis.run(data)

            if self.store_data:
                self.storage = pd.concat([self.storage,pd.DataFrame([data], index = [self.working_time])])

            self.running = self.march()
        
        # print completion summary
        print("\n...Complete...")
        print("Finished analyzing {}.\n\tNumber of observations analyzed: {}".format(self.filename,len(self.processor.timestamps)))

    def run_all(self):
        '''
        Run all analysis for all nodes in initialization. This serves as an
        easy way to run the whole analysis rather than individually executing
        class methods. Although either way is valuable and manual execution
        may be better for debugging and development.
        '''

        if self.store_data:
            self.storage = pd.DataFrame()
        
        self.queue_file()
        self.iterate()

        if self.store_data:
            self.storage.to_csv('results_'+self.filename)
