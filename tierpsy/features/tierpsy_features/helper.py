#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:36:52 2017

@author: ajaver
"""
import pandas as pd
import numpy as np
import numba
import math
import os
from scipy.ndimage import generic_filter

extras_dir = os.path.join(os.path.dirname(__file__), 'extras')
def load_OW_eigen_projections():
    eigen_projection_file =  os.path.join(extras_dir, 'master_eigen_worms_N2.mat')
    assert os.path.exists(eigen_projection_file)
    with tables.File(EIGEN_PROJECTION_FILE) as fid:
        eigen_worms = fid.get_node('/eigenWorms')[:]
        eigen_worms = eigen_worms.T
    return eigen_worms

def load_eigen_projections(n_projections = 7):
    eigen_projection_file = os.path.join(extras_dir, 'pca_components.npy')
    if not os.path.exists(eigen_projection_file):
        raise FileNotFoundError('The file {} does not exists. I cannot start tierpsy features.') 
    eigen_worms = np.load(eigen_projection_file)[:n_projections]
    return eigen_worms


@numba.jit
def fillfnan(arr):
    '''
    fill foward nan values (iterate using the last valid nan)
    I define this function so I do not have to call pandas DataFrame
    '''
    out = arr.copy()
    for idx in range(1, out.shape[0]):
        if np.isnan(out[idx]):
            out[idx] = out[idx - 1]
    return out

@numba.jit
def fillbnan(arr):
    '''
    fill foward nan values (iterate using the last valid nan)
    I define this function so I do not have to call pandas DataFrame
    '''
    out = arr.copy()
    for idx in range(out.shape[0]-2, -1, -1):
        if np.isnan(out[idx]):
            out[idx] = out[idx + 1]
    return out

def nanunwrap(x):
    '''correct for phase change for a vector with nan values     '''
    x = x.astype(np.float)

    bad = np.isnan(x)
    x = fillfnan(x)
    x = fillbnan(x)
    x = np.unwrap(x)
    x[bad] = np.nan
    return x

def get_n_worms_estimate(frame_numbers, percentile = 99):
    '''
    Get an estimate of the number of worms using the table frame_numbers vector
    '''
    
    n_per_frame = frame_numbers.value_counts()
    n_per_frame = n_per_frame.values
    if len(n_per_frame) > 0:
        n_worms_estimate = np.percentile(n_per_frame, percentile)
    else:
        n_worms_estimate = 0
    return n_worms_estimate


def get_delta_in_frames(delta_time, fps):
    '''Get the conversion of delta time in frames. Make sure it is more than one.'''
    return max(1, int(round(fps*delta_time)))

def add_derivatives(feats, cols2deriv, delta_frames, fps):
    '''
    Calculate the derivatives of timeseries features, and add the columns to the original dataframe.
    '''
    #%%
    val_cols = [x for x in cols2deriv if x in feats]
    
    feats = feats.sort_values(by='timestamp')
    
    df_ts = feats[val_cols].copy()
    df_ts.columns = ['d_' + x for x in df_ts.columns]
    
    m_o, m_f = math.floor(delta_frames/2), math.ceil(delta_frames/2)
    
    
    vf = df_ts.iloc[delta_frames:].values
    vo = df_ts.iloc[:-delta_frames].values
    vv = (vf - vo)/(delta_frames/fps)
    
    #the series was too small to calculate the derivative
    if vv.size > 0:
        df_ts.loc[:] =  np.nan
        df_ts.iloc[m_o:-m_f] = vv
        
    feats = pd.concat([feats, df_ts], axis=1)
    #%%
    return feats

def nanmedian_filter(time_series: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving median filter to a 1D time series, ignoring NaN values.
    The filter is applied along the time axis only, preserving NaNs in the output.
    
    Parameters:
    - time_series: np.ndarray, input 1D array with possible NaN values
    - window_size: int, the size of the moving window (must be an odd number)
    
    Returns:
    - np.ndarray, the smoothed time series with NaNs preserved
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    def nanmedian(window: np.ndarray) -> float:
        "Calculate the median of a window, ignoring NaNs."
        valid_values = window[~np.isnan(window)]
        if valid_values.size > 0:
            return np.median(valid_values)
        else:
            return np.nan

    # Apply the custom nanmedian function using generic_filter
    smoothed = generic_filter(time_series, nanmedian, size=window_size, mode='nearest')
    
    # Preserve NaNs in the original time series
    smoothed[np.isnan(time_series)] = np.nan

    return smoothed

def fill_nans_1D(arr):
    """
    Linearly interpolate over NaN values in a 1D NumPy array.

    This function identifies all NaN entries in the input array and replaces
    each NaN with a linearly interpolated value based on the nearest non-NaN
    neighbors. If NaNs appear at the very beginning or end of the array, they
    are filled by carrying forward/backward the nearest non-NaN value. If the
    entire array is NaN, the function returns a copy of the original array.

    Parameters
    ----------
    arr : array-like, shape (N,)
        One-dimensional array containing numeric values and possible NaNs.

    Returns
    -------
    arr_filled : numpy.ndarray, shape (N,)
        A new array where NaN entries have been replaced by interpolated
        values. Non-NaN entries remain unchanged.

    """
    arr = np.asarray(arr, dtype=float)

    # If there are no NaNs or all NaNs, return a copy immediately
    if (not np.isnan(arr).any()) or (np.isnan(arr).all()):
        return arr.copy()

    # Indices of valid (non-NaN) entries
    good_idx = np.where(~np.isnan(arr))[0]
    # Indices of NaN entries
    nan_idx = np.where(np.isnan(arr))[0]

    # Create a copy to hold filled values
    arr_filled = arr.copy()

    # Use numpy.interp for linear interpolation over NaNs
    arr_filled[nan_idx] = np.interp(
        nan_idx,       # x-coordinates to fill
        good_idx,      # x-coordinates of known (non-NaN) values
        arr[good_idx]  # known y-values
    )

    return arr_filled

class DataPartition():
    def __init__(self, partitions=None, n_segments=49):

        #the upper limits are one more than the real limit so I can do A[ini:fin]
        partitions_dflt = {'head': (0, 8),
                            'neck': (8, 16),
                            'midbody': (16, 33),
                            'hips': (33, 41),
                            'tail': (41, 49),
                            'head_tip': (0, 3),
                            'head_base': (5, 8),
                            'tail_base': (41, 44),
                            'tail_tip': (46, 49),
                            'all': (0, 49),
                            #'hh' : (0, 16),
                            #'tt' : (33, 49),
                            'body': (8, 41),
                            }
        
        if partitions is None:
            partitions = partitions_dflt
        else:
            partitions = {p:partitions_dflt[p] for p in partitions}
            
        
        if n_segments != 49:
            r_fun = lambda x : int(round(x/49*n_segments))
            for key in partitions:
                partitions[key] = tuple(map(r_fun, partitions[key]))
        
        self.n_segments = n_segments
        self.partitions =  partitions

    def apply(self, data, partition, func, segment_axis=1):
        assert self.n_segments == data.shape[segment_axis]
        assert partition in self.partitions
        
        ini, fin = self.partitions[partition]
        sub_data = np.take(data, np.arange(ini, fin), axis=segment_axis)
        d_transform = func(sub_data, axis=segment_axis)
        
        return d_transform
   
    def apply_partitions(self, data, func, segment_axis=1):
        return {p:self.apply(data, p, func, segment_axis=segment_axis) for p in self.partitions}
