#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:49:14 2020

@author: em812
"""
import os
import pandas as pd
import numpy as np
from tierpsy.helper.misc import print_flush
from tierpsy import AUX_FILES_DIR
import pdb

FEAT_SET_DIR = os.path.join(AUX_FILES_DIR,'feat_sets')
feature_sets_filenames = {
    'tierpsy': {
        'all' : 'tierpsy_features_all_names.csv',
        'tierpsy_level_0' : 'tierpsy_level_0.csv',
        'tierpsy_level_1' : 'tierpsy_level_1.csv',
        'tierpsy_8' : 'tierpsy_8.csv',
        'tierpsy_16' : 'tierpsy_16.csv',
        'tierpsy_256' : 'tierpsy_256.csv',
        'tierpsy_2k' : 'top2k_tierpsy_no_blob_no_eigen_only_abs_no_norm.csv'
        },
    'openworm': {
        'all' : 'openworm_features_all_names.csv',
        }
    }

valid_time_windows_connector = ':'
valid_time_windows_extend = '+'
valid_time_windows_separator = ','
time_windows_format_explain = \
    'Each time window must be defined by the start time and the end time ' + \
    'connected by \'{}\' '.format(valid_time_windows_connector) + \
    '(start_time:end_time). Different windows must be ' + \
    'separated by \'{}\'. '.format(valid_time_windows_separator) + \
    'A sequence of equally sized windows can be defined with the format ' + \
    'start_time{0}end_time{0}step.'.format(valid_time_windows_connector) + \
    'A combination of time intervals can define a SINGLE window with the format ' + \
    'start_time1{0}end_time1 {1} start_time2{0}end_time2.'.format(
        valid_time_windows_connector, valid_time_windows_extend)

def time_windows_parser(time_windows):
    """
    EM : Converts the string input from the GUI to a list object of integers.
    Asserts that for each time window start_time<=end_time
    """

    if not time_windows.replace(' ',''):
        return [[0,-1]]
    if valid_time_windows_connector not in time_windows:
        raise ValueError(
            'Invalid format of time windows: ' + time_windows_format_explain
            )
        return

    # Remove spaces and replace end with -1
    windows = time_windows.replace(' ','').replace('end','-1')
    # Split at ',' to separate time windows
    windows = windows.split(valid_time_windows_separator)

    # Split each non-empty time window at '+'
    windows = [x.split(valid_time_windows_extend) for x in windows if x]

    # then split each non-empty time
    # window at ':'
    windows = [[x.split(valid_time_windows_connector) for x in win if x] for win in windows]

    # Convert to integers
    try:
        windows = [[[int(x) for x in wdw] for wdw in win] for win in windows]
    except ValueError:
        print_flush(
            'Time windows input could not be converted to list of numbers.'+
            time_windows_format_explain
            )
        raise

    fin_windows = []
    for iwin,win in enumerate(windows):
        # If there was no + in the window definition (one interval):
        if len(win)==1:
            window=win[0]
            if len(window)==3:
                if window[1]==-1:
                    raise ValueError(
                        'Invalid format of time windows: When the format ' +
                        'start_time:end_time:step is used, the end_time ' +
                        'has to be defined explicitly in seconds or frames.' +
                        ' It cannot be \'end\' or \'-1\'.')
                else:
                    assert window[0]<=window[1], \
                        "Invalid format of time windows: The end time of " + \
                        "time window {}/{} ".format(iwin+1,len(windows) + \
                        "cannot be smaller than the start time.")
                    assert window[2]<=window[1]-window[0], \
                        "Invalid format of time windows: The step size in " + \
                        "time window {}/{} ".format(iwin+1,len(windows)) + \
                        "cannot be larger than the (end_time-start_time)."
                start,end,step = window
                step_wins = [
                    [i,j] for i,j in zip(
                    list(range(*window)),
                    list(range(start+step,end,step))+[end])
                    ]
                for add in step_wins:
                    fin_windows.append([add])
            elif len(window)==2:
                if window[1]!=-1:
                    assert window[0]<=window[1], \
                        "Invalid format of time windows: The end time of " +\
                        "time window {}/{} ".format(iwin+1,len(windows)) + \
                        "cannot be smaller than the start time."
                fin_windows.append([window])
            else:
                ValueError(
                    'Invalid format of time windows: ' +
                    time_windows_format_explain
                    )
        # If there are + in the window definition (more than one intervals):
        else:
            extended_win = []
            for window in win:
                if len(window)==3:
                    raise ValueError(
                        'Invalid format of time windows: The step-based format ' +
                        'start_time:end_time:step cannot be combined with ' +
                        'the format start_time1:end_time1 + start_time2:end_time2.')
                if len(window)!=2:
                    raise ValueError(
                        'Invalid format of time windows: ' +
                        time_windows_format_explain
                        )
                if window[1]!=-1:
                    assert window[0]<=window[1], \
                        "Invalid format of time windows: The end time of " +\
                        "time window {}/{} ".format(iwin+1,len(windows)) + \
                        "cannot be smaller than the start time."
                extended_win.append(window)
            fin_windows.append(extended_win)
    return fin_windows

def drop_ventrally_signed(feat_names):
    """
    EM: drops the ventrally signed features
    Param:
        features_names = list of features names
    Return:
        filtered_names = list of features names without ventrally signed
    """

    absft = [ft for ft in feat_names if '_abs' in ft]
    ventr = [ft.replace('_abs', '') for ft in absft]

    filtered_names = list(set(feat_names).difference(set(ventr)))

    return filtered_names

def feat_set_parser(feature_type, select_feat):
    """
    EM : gets the full path of the file containing the selected feature set.
    """
    if select_feat in feature_sets_filenames[feature_type].keys():
        feat_set_file = os.path.join(
            FEAT_SET_DIR, feature_sets_filenames[feature_type][select_feat])
        selected_feat = pd.read_csv(feat_set_file, header=None, index_col=None)
        selected_feat = selected_feat.values.flatten().tolist()
    else:
        raise ValueError
    return selected_feat

def select_parser(
        feature_type, keywords_include, keywords_exclude, select_feat,
        dorsal_side_known):
    """
    EM: collects feature-selection related variables from the GUI, parses them
    to lists of strings and returns the list of finally selected features.
    """
    # EM : get full path to feature set file
    selected_feat = feat_set_parser(feature_type, select_feat)

    if feature_type=='openworm':
        return selected_feat

    # EM : get list of keywords to include and to exclude
    keywords_in = keywords_parser(keywords_include)
    keywords_ex = keywords_parser(keywords_exclude)

    # EM : catch conflicts
    if (keywords_in is not None) and (keywords_ex is not None):
        if len(list(set(keywords_in) & set(keywords_ex))) > 0:
            raise ValueError('Cannot accept the same keyword in both ' +
                             'keywords_include and keywords_exclude.\n' +
                             'Keyword(s) {} found in both lists.'.format(
                                 list(set(keywords_in) & set(keywords_ex))))

    # if keywords_in is None and keywords_ex is None \
    #     and feat_set is None and dorsal_side_known:

    #     return None

    if not dorsal_side_known:
        selected_feat = drop_ventrally_signed(selected_feat)

    if keywords_in is not None:
        selected_feat = [
            ft for ft in selected_feat
            if np.any([x in ft for x in keywords_in])
            ]
    if keywords_ex is not None:
        selected_feat = [
            ft for ft in selected_feat
            if np.all([x not in ft for x in keywords_ex])
            ]

    return selected_feat

def _get_threshold(text):
    text = text.replace(' ','')

    if not text:
        return

    try:
        threshold = float(text)
    except ValueError:
        print_flush('The threshold for trajectories filtering must be a number.')

    return threshold

def filter_args_parser(filter_args):

    if not bool(filter_args):
        return

    filter_params = {
        'min_traj_length': _get_threshold(filter_args['filter_time_min']),
        'min_distance_traveled': _get_threshold(filter_args['filter_travel_min']),
        'time_units': filter_args['filter_time_units'],
        'distance_units': filter_args['filter_distance_units']
        }

    filter_params['timeseries_names'] = ['length', 'width_midbody']
    filter_params['min_thresholds'] = [
        _get_threshold(filter_args['filter_length_min']),
        _get_threshold(filter_args['filter_width_min'])]
    filter_params['max_thresholds'] = [
        _get_threshold(filter_args['filter_length_max']),
        _get_threshold(filter_args['filter_width_max'])]
    filter_params['units'] = [
        filter_args['filter_distance_units'],
        filter_args['filter_distance_units']]

    return filter_params

def keywords_parser(keywords):
    """
    EM : Converts the string input from the GUI to a list object of strings.
    """
    # Remove spaces
    kwrds = keywords.replace(' ','')
    # Split at ',' to separate time windows, then keep non-empty words
    kwrds = [x for x in kwrds.split(',') if x]

    if kwrds:
        return kwrds
    else:
        return None

