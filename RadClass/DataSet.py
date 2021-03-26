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
    '''

    def __init__(self):
        '''
        Initializes DataSet class.
        No vectors/data will be stored until methods are called.
        '''

        self.live = None
        self.timestamps = None

    def init_database(self, filename, datapath):
        '''
        Use h5py to load HDF5 MUSE files. Each file typically contains at least
        one MUSE node/detector recording data for 1 month periods.
        Assumes that the HDF5 directory has 3 elements required for analysis:
        2x4x16LiveTimes - Each iterations measurement time in units of [sec]
            corrected for dead time.
        2x4x16Times - The epoch timestamp for when a measurement occurred.
        2x4x16Spectra - Measurement data. Recorded as a matrix with each
            row representing one measurement period with spectrum details
            subdivided into 1000 bins of 3 keV each.
        NOTE: This function only returns the live time and timestamps. The spectra
            data is too large to manipulate en masse. Instead, see the data_slice
            function for returning certian rows of data.
        
        Input:
        filename: string path to file, including directory.
        datapath: HDF5 group/subgroup path to dataset, including prefix,
                   node ID, and analysis element. e.g. '/path/to/dataset'

        Return: live times and timestamps as numpy arrays
        '''

        file = h5py.File(filename, 'r')
        
        # [:] returns a numpy array via h5py
        self.live = file[datapath][:]
        self.timestamps = file[datapath][:]

        file.close()

        return self.live, self.timestamps

    def data_slice(self, filename, datapath, rows):
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

        file = h5py.File(filename, 'r')

        # data is a numpy array (as returned by h5py)
        data = file[datapath][rows]
        file.close()

        return data
