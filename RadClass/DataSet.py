import h5py

class DataSet:
    '''
    Conducts data I/O for HDF5 files.
    
    The docstrings for methods below include examples for processing MUSE data,
    but could be used on any HDF5 dataset with 3 data groups (detector live-time,
    a vector of timestamps, and an array of spectrum data).
    
    Attributes:
        live: A numpy array with dead-time corrected instrument recording time.
        timestamps: epoch timestamps for the start(?) of a measurement recording.
        labels: list of dataset name labels in this order:
            [ live_label: live dataset name in HDF5 file,
              timestamps_label: timestamps dataset name in HDF5 file,
              spectra_label: spectra dataset name in HDF5 file ]
    '''

    def __init__(self, labels = {'live': '2x4x16LiveTimes',
                                 'timestamps': '2x4x16Times',
                                 'spectra': '2x4x16Spectra'}):
        '''
        Initializes DataSet class.
        No vectors/data will be stored until methods are called.
        '''

        self.live = None
        self.timestamps = None
        self.file = None

        self.live_label = labels['live']
        self.timestamps_label = labels['timestamps']
        self.spectra_label = labels['spectra']

    def close(self):
        '''
        Remove HDF5 file from DataSet memory. This method will only attempt to
        close a file if it has been loaded into memory already (i.e. self.file
        has not been already closed or was never loaded).
        NOTE: This method will have to be called by parent classes using DataSet.
        That is, RadClass. Since self.file should not be called by any other
        class or method other than DataSet, it should be destructed once a script
        completes, regardless of whether DataSet.close() is called. This method
        is primarily for parent classes (RadClass) that will run through multiple
        files and thus require DataSet to be reset.

        Return: None; self.file should now be closed/empty.
        '''

        if self.file is not None:
            self.file.close()
            # reset file instance
            self.file = None

    def init_database(self, filename, datapath):
        '''
        Use h5py to load HDF5 MUSE files. Each file typically contains at least
        one MUSE node/detector recording data for 1 month periods.
        Assumes that the HDF5 directory has 3 elements required for analysis:
        LiveTimes - Each iterations measurement time in units of [sec]
            corrected for dead time.
        Times - The epoch timestamp for when a measurement occurred.
        Spectra - Measurement data. Recorded as a matrix with each
            row representing one measurement period with spectrum details
            subdivided into 1000 bins of 3 keV each.
        NOTE: This function only returns the live time and timestamps. The spectra
            data is too large to manipulate en masse. Instead, see the data_slice
            function for returning certian rows of data.
        
        Input:
        filename: string path to file, including directory.
        datapath: HDF5 group/subgroup path to dataset, including prefix,
                   node ID, and analysis element. e.g. '/path/to/dataset'

        Attributes:
        live_times: A numpy array of live detection times corrected for dead time.
            These are assumed to be approximately 1 second, but a possible TODO
            would be to discard readings that do not reach a required live time
            after dead time correction.
        timestamps: A numpy array of the epoch timestamp at which a detection was
            recorded. These are used for indexing and processing data.

        Return: None; vectors saved in class attributes.
        '''

        self.file = h5py.File(filename, 'r')
        
        # [:] returns a numpy array via h5py
        self.live = self.file[datapath + self.live_label][:]
        self.timestamps = self.file[datapath + self.timestamps_label][:]

    def data_slice(self, datapath, rows):
        '''
        Use h5py to read a MINOS MUSE spectra data file and return a specified
        number of rows of the data. The spectral data is a matrix of size
        appx. 2678142x1000 (for a 1 month data file) and thus too large to keep
        in memory during execution. Instead, using h5py, specific rows of the
        data can be returned for manipulation. Performance can be saved if cached.

        Input:
        filename: string path to file, including directory, of data
        datapath: HDF5 group/subgroup path to dataset, including prefix,
                node ID, and analysis element. e.g. '/path/to/dataset'
        rows: LIST of rows to be sliced and returned. These rows correspond to
            the row indices of the timestamp for the corresponding data.

        Return: numpy matrix array of rows of data (###x1000 dimensions)
        '''

        # data is a numpy array (as returned by h5py)
        data = self.file[datapath + self.spectra_label][rows]

        return data
