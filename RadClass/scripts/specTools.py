import numpy as np
import pandas as pd
import h5py as h
from typing import List, Optional, Type


def integrate_spectral_matrix(
		S: np.ndarray,
		integration_time: int,
		stride: int
) -> List[np.ndarray]:
	"""
	:param S: matrix of 1-sec spectra
	:param integration_time: desired integration, length of each spectral block
	:param stride: shift between spectral blocks
	:return: list of integrated spectra, each as a np.ndarray (1,n) vector for n channels
	"""
	# set limits for loop
	last_row = S.shape[0]
	current_row = 0
	spectra = []
	while (current_row + integration_time) <= last_row:
		spectra.append(
			np.atleast_2d(np.sum(S[current_row:current_row+integration_time, :], axis=0)).reshape(1, -1)
		)
		current_row += stride
	return spectra

"""
def remove_event_counter(df):
	# removes the trailing counter from the event label
	def relabel_row(r):
		return '_'.join(r['event'].split('_')[:-1])

	df['event'] = df.apply(relabel_row, axis=1)

	return df
"""

def separate_event_counter(df):
	"""make event instance/counter a separate column for tracking/parsing"""
	def _helper(r):
		split_event = r['event'].split('_')
		r['event'] = '-'.join(split_event[:-1])
		r['instance'] = split_event[-1]
		return r

	df = df.apply(_helper, axis=1)
	return df


def resample_spectra(
        df: pd.DataFrame,
        n: int,
        n_channels=1000
) -> pd.DataFrame:
    """
    :param df: dataframe containing m spectra as rows and labels
    :param n: number of resamples for each spectrum
    :return: list of m * (n + 1) spectra
    """
    def _resample(spec):
        """performs single resample"""
        return np.array([np.random.poisson(lam=channel) for channel in spec])

    # combine labels to make repeating easier
    unsplit_columns = df.columns
    print("Before combine_label():\n")
    print(df.columns)
    df = combine_label(df)
    print("\n\nAfter combine_label()\n")
    print(df.columns)

    spectra = np.array(df.iloc[:, :n_channels])
    # note we assume our label is in one columns
    labels = np.array(df.iloc[:, n_channels])

    # note np.repeat() repeats each element rather than repeating the whole array
    new_spectra = [_resample(spectrum) for spectrum in spectra for _ in range(n) ]
    new_labels = np.concatenate([labels.reshape(-1, 1), np.repeat(labels, n).reshape(-1, 1)], axis=0)
    combined_data = np.concatenate(
        [np.concatenate([spectra, new_spectra], axis=0), new_labels], axis=1
    )

    # undo label combine to allow separate tracking of event, event counter, and detector/station
    # I might be able to skip the next line
    df_ = pd.DataFrame(data=combined_data, columns=df.columns)
    df_ = split_labels(df_)
    #print("After split_labels()\n")
    #print(df.columns)
    #print("Size of combined data...")

    return df_


def combine_label(df):
    """combines event and detector to make resampling easier"""
    def _combine_helper(r):
        return '_'.join([r['event'], r['detector'], r['instance']])

    df['label'] = df.apply(_combine_helper, axis=1)
    df = df.drop(['event', 'detector', 'instance'], axis=1)
    return df


def split_labels(df):
	"""opposite of combine labels to do after resampling"""
	def _split_helper(r):
		r['event'] = r['label'].split('_')[0]
		r['detector'] = r['label'].split('_')[1]
		r['instance'] = r['label'].split('_')[2]
		return r

	df = df.apply(_split_helper, axis=1)
	df = df.drop('label', axis=1)
	
	return df


def read_h_file(
		file: str,
		integration_time: int,
		stride: int,
		resample: bool=False,
		n: int=None
) -> pd.DataFrame:
	"""
	extract time-integrated spectra for multiple events and detectors from hdf5 file
	:param file: hdf5 file as string
	:param integration_time: desired integration for spectral processing
	:param stride: stride for moving-window time integration
	:param resample: choose to resample spectra to generate additional 
	:return: flattened pd.dataFrame of spectra and associated information/labels
	"""
	df_list = []

	cols = [f'channel {i}' for i in range(1, 1001)] # number for channels ugly hardcoded

	f = h.File(file, 'r')
	events = list(f.keys())
	for event in events:
		print(f'Processing {event} events')
		current_event = f[event]
		nodes = list(current_event.keys())
		for node in nodes:
			spectral_matrix = np.array(current_event[node]['spectra'])
			spectra_list = integrate_spectral_matrix(spectral_matrix, integration_time, stride)
			for s in spectra_list:
				df_ = pd.DataFrame(data=s, columns=cols)
				df_['event'] = event
				df_['detector'] = node
				df_list.append(df_)
			#return [np.array(spectra_list[0]), event, node]

	df = pd.concat(df_list)
	df = separate_event_counter(df)

	return df

