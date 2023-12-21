'''
Author: Ken Dayman

Ken shared these scripts with me for processing spectral data.
I left these here because configs.py uses them, but they are probably
uninteresting to someone who does not use the same data. -Jordan Stomps
'''

import numpy as np
import pandas as pd
import h5py as h
from typing import List


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

