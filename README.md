# RadClass

![GitHub issues](https://img.shields.io/github/issues/cnerg/RadClass)
![GitHub pull requests](https://img.shields.io/github/issues-pr/cnerg/RadClass)
[![Coverage Status](https://coveralls.io/repos/github/CNERG/RadClass/badge.svg?branch=main)](https://coveralls.io/github/CNERG/RadClass?branch=main)

A collection of tools (including data analysis and machine learning applications) that can be used to explore radiation data (temporal and spectral) for external identification of radiation signatures.

## Table of Contents

1. [Installation](#installation)

2. [Usage](#usage)

3. [Data Format](#data-format)

## Installation

### Dependencies

* Python 3

Versions 3.6-3.9 are currently supported by tests. The following Python packages are required to run (found in `requirements.txt`):

* h5py
* numpy
* progressbar2
* matplotlib
* seaborn
* scipy
* sklearn
* hyperopt
* torch
* shadow-ssml

Modules can be imported from the repository directory (e.g. `from RadClass.H0 import H0`) or `RadClass` can be installed using pip:

`python -m pip install /path/to/RadClass/. --user`

## Usage

RadClass is designed to be modular, allowing a user to use analysis modules as needed:

![RadClass Workflow](/images/RadClass_workflow.png)

An HDF5 data file is specified as input, which is processed by `DataSet`. The user can specify a type of `AnalysisParameters`. For example, `H0` for hypothesis testing, `BackgroundEstimator` for background estimation, etc.
`Processor` then uses `DataSet` and the user specified `AnalysisParameters` to run, storing the results for use by the user.
To see examples of how a user can initialize and run `Processor`, review /tests/.

## Data Format

`RadClass.Processor` expects a data structure as follows:

![File Structure](/images/file_structure.png)

The HDF5 file must have three groups:

1. A vector of epoch timestamps at which the measurement was taken.
2. If applicable, the live recording time of the detector to correct for dead time. If not applicable, a vector of 1's.
3. A data matrix of `n` measurements given for `m` bins.

Each group's name must be specified in an input dictionary: `labels`.

Integration occurs over the course of the data matrix.

![Integration Algorithm](/images/integration_algorithm.png)

Data rows are corrected for dead time and summed for the specified integration input length (then averaged over the integration period).
If stride is specified, the working timestamp will advance forward by the specified amount. While a required input, setting `stride = integration` will ignore this behavior.
A `cache_size` can be given, which will pre-slice a specified number of rows into `RadClass` to reduce file I/O. Otherwise, each set of integration rows will be sliced separately.
A `start_time` and `stop_time` can also be specified for periods of data processing smaller than the total length of the input file. These two variables must be epoch timestamps in
units of seconds, which can generally be given using the `time` and `datetime` Python modules:

`time.mktime(datetime.datetime.strptime(x, "%m/%d/%Y %H:%M:%S").timetuple())`

Where `x` is a date string. See [module](https://docs.python.org/3/library/time.html) [docs](https://docs.python.org/3/library/datetime.html) for more details.
