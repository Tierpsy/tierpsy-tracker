#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: avelinojaver
"""
# %%
import os
import glob
import datetime
import tables
import pandas as pd
import numpy as np

from functools import partial
from multiprocessing import Pool, cpu_count

from tierpsy.helper.misc import TimeCounter, print_flush
from tierpsy.summary.process_ow import ow_plate_summary, \
    ow_trajectories_summary, ow_plate_summary_augmented
from tierpsy.summary.process_tierpsy import tierpsy_plate_summary, \
    tierpsy_trajectories_summary, tierpsy_plate_summary_augmented
from tierpsy.summary.helper import \
    get_featsum_headers, get_fnamesum_headers, shorten_feature_names
from tierpsy.summary.parsers import \
    time_windows_parser, filter_args_parser, select_parser

feature_files_ext = {'openworm' : ('_features.hdf5', '_feat_manual.hdf5'),
                     'tierpsy' : ('_featuresN.hdf5', '_featuresN.hdf5')
                     }

valid_feature_types = list(feature_files_ext.keys())
valid_summary_types = ['plate', 'trajectory', 'plate_augmented']

feat_df_id_cols = \
    ['file_id', 'i_fold', 'worm_index', 'n_skeletons', 'well_name', 'is_good_well']


def check_in_list(x, list_of_x, x_name):
    if not x in list_of_x:
        raise ValueError(
            '{} invalid {}. Valid options {}.'.format(x, x_name, list_of_x)
            )


def check_n_parallel(n_par):
    """
    check_n_parallel clips the value of n_par between 1 and the
    number of available cores - 1.
    If tierpsy detects it's beinng run in an HPC (PBS or SLURM), no upper bound
    is applied (we assume the user knows what they're doing)

    Parameters
    ----------
    n_par : int
        desired number of parallel processes

    Returns
    -------
    int
        Actual number of parallel processes that can be achieved on the system
    """
    # check if we're running in hpc
    if any(x.startswith(('PBS_', 'SLURM_')) for x in os.environ):
        return max(1, n_par)

    try:
        max_n_procs = len(os.sched_getaffinity(0)) - 1
    except:
        from multiprocessing import cpu_count
        max_n_procs = cpu_count() - 1
    n_par = min(n_par, max_n_procs)
    n_par = max(1, n_par)
    return n_par


def get_summary_func(
        feature_type, summary_type,
        time_windows_ints, time_units,
        selected_feat,
        dorsal_side_known, filter_params,
        is_manual_index, **fold_args
        ):
    """
    Chooses the function used for the extraction of feature summaries based on
    the input from the GUI
    """
    if feature_type == 'tierpsy':
        if summary_type == 'plate':
            func = partial(
                tierpsy_plate_summary,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params
                )
        elif summary_type == 'trajectory':
            func = partial(
                tierpsy_trajectories_summary,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params
                )
        elif summary_type == 'plate_augmented':
            func = partial(
                tierpsy_plate_summary_augmented,
                time_windows=time_windows_ints, time_units=time_units,
                only_abs_ventral = not dorsal_side_known,
                selected_feat = selected_feat,
                is_manual_index=is_manual_index,
                filter_params = filter_params,
                **fold_args
                )

    elif feature_type == 'openworm':
        if summary_type == 'plate':
            func = ow_plate_summary
        elif summary_type == 'trajectory':
            func = ow_trajectories_summary
        elif summary_type == 'plate_augmented':
            func = partial(ow_plate_summary_augmented, **fold_args)
    return func


def sort_columns(df, selected_feat):
    """
    Sorts the columns of the feat summaries dataframe to make sure that each
    line written in the features summaries file contains the same features with
    the same order. If a feature has not been calculated for a specific file,
    then a nan column is added in its place.
    """

    not_existing_cols = [col for col in selected_feat if col not in df.columns]

    if len(not_existing_cols) > 0:
        for col in not_existing_cols:
            df[col] = np.nan

    df = df[[x for x in feat_df_id_cols if x in df.columns] + selected_feat]

    return df

def make_df_filenames(fnames):
    """
    EM : Create dataframe with filename summaries and time window info for
    every time window
    """
    dd = tuple(zip(*enumerate(sorted(fnames))))
    df_files = pd.DataFrame({'file_id' : dd[0], 'filename' : dd[1]})
    df_files['is_good'] = False
    return df_files


def calculate_summaries(
        root_dir, feature_type, summary_type, is_manual_index,
        abbreviate_features, dorsal_side_known,
        time_windows='0:end', time_units=None,
        select_feat='all', keywords_include='', keywords_exclude='',
        _is_debug=False, n_parallel=-1, **kwargs
        ):
    """
    Gets input from the GUI, calls the function that chooses the type of
    summary and runs the summary calculation for each file in the root_dir.
    """
    filter_args = {k:kwargs[k] for k in kwargs.keys() if 'filter' in k}
    fold_args = {k:kwargs[k] for k in kwargs.keys() if 'filter' not in k}

    #check the options are valid
    check_in_list(feature_type, valid_feature_types, 'feature_type')
    check_in_list(summary_type, valid_summary_types, 'summary_type')
    n_parallel = check_n_parallel(n_parallel)


    # EM : convert time windows to list of integers in frame number units
    time_windows_ints = time_windows_parser(time_windows)
    filter_params = filter_args_parser(filter_args)
    # EM: get lists of strings (in a tuple) defining the feature selection
    # from keywords_in,
    # keywords_ex and select_feat.
    selected_feat = select_parser(
        feature_type, keywords_include, keywords_exclude, select_feat, dorsal_side_known)

    #get summary function
    # INPUT time windows time units here
    summary_func = get_summary_func(
        feature_type, summary_type,
        time_windows_ints, time_units,
        selected_feat,
        dorsal_side_known, filter_params,
        is_manual_index, **fold_args)

    # get basenames of summary files
    save_base_name = 'summary_{}_{}'.format(feature_type, summary_type)
    if is_manual_index:
        save_base_name += '_manual'
    save_base_name += '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    #get extension of results file
    possible_ext = feature_files_ext[feature_type]
    ext = possible_ext[1] if is_manual_index else possible_ext[0]

    fnames = glob.glob(os.path.join(root_dir, '**', '*' + ext), recursive=True)
    if not fnames:
        print_flush('No valid files found. Nothing to do here.')
        return None, None

    # EM :Make df_files dataframe with filenames and file ids
    df_files = make_df_filenames(fnames)

    # EM : Create features_summaries and filenames_summaries files
    #      and write headers
    fnames_files = []
    featsum_files = []
    for iwin in range(len(time_windows_ints)):
        # EM : Create features_summaries and filenames_summaries files
        if select_feat != 'all':
            win_save_base_name = save_base_name.replace(
                'tierpsy', select_feat+'_tierpsy')
        else:
            win_save_base_name = save_base_name

        if not (len(time_windows_ints)==1 and time_windows_ints[0][0]==[0,-1]):
            win_save_base_name = win_save_base_name+'_window_{}'.format(iwin)

        f1 = os.path.join(
            root_dir, 'filenames_{}.csv'.format(win_save_base_name))
        f2 = os.path.join(
            root_dir,'features_{}.csv'.format(win_save_base_name))

        fnamesum_headers = get_fnamesum_headers(
            f2, feature_type, summary_type, iwin, time_windows_ints[iwin],
            time_units, len(time_windows_ints), select_feat, filter_params,
            fold_args, df_files.columns.to_list())
        featsum_headers = get_featsum_headers(f1)

        with open(f1, 'w') as fid:
            fid.write(fnamesum_headers)

        with open(f2, 'w') as fid:
            fid.write(featsum_headers)

        fnames_files.append(f1)
        featsum_files.append(f2)

    progress_timer = TimeCounter('')

    def _displayProgress(n):
        args = (n + 1, len(df_files), progress_timer.get_time_str())
        dd = "Extracting features summary. "
        dd += "File {} of {} done. Total time: {}".format(*args)
        print_flush(dd)

    _displayProgress(-1)

    # EM : Extract feature summaries from all the files for all time windows.

    # make a partial function that only needs the one input
    _partial_calculate_summaries_one_video = partial(
        _calculate_summaries_one_video,
        summary_func=summary_func,
        fnames_files=fnames_files,
        featsum_files=featsum_files,
        abbreviate_features=abbreviate_features,
        selected_feat=selected_feat,
        )

    if n_parallel < 2:
        # just do one at a time, no Pool call to avoid overhead
        for ifile, row in df_files.iterrows():
            _partial_calculate_summaries_one_video(row)
            _displayProgress(ifile)
    else:

        # iterrows returning a line counter breaks the partial/parallel, so
        # create silly generator that just discards the counter
        row_looper = (row for _, row in df_files.iterrows())
        with Pool(n_parallel) as p:
            outs = p.imap_unordered(
                _partial_calculate_summaries_one_video, row_looper)

            # loop throug outs consumes the iterator and does the maths
            ifile = 0
            for _ in outs:
                _displayProgress(ifile)
                ifile += 1

    out = '****************************'
    out += '\nFINISHED. Created Files:'
    for f1, f2 in zip(fnames_files, featsum_files):
        out += '\n-> {}\n-> {}'.format(f1, f2)

    print_flush(out)

    return df_files


def _calculate_summaries_one_video(
        row, summary_func, fnames_files, featsum_files,
        abbreviate_features=False, selected_feat='all'):

    # name of the results file to process, and its incremental id
    fname = row['filename']
    file_id = row['file_id']

    # summary_func is a partial function with all parameters already passed
    # this is the bit that actually does the calculations
    summaries_per_win = summary_func(fname)

    # loop on windows to write to output
    for iwin, df in enumerate(summaries_per_win):
        # get the name of filenames_summary and features_summary for this window
        f1 = fnames_files[iwin]
        f2 = featsum_files[iwin]
        print(f1, f2)

        try:
            # add the file_id column, and sort the columns.
            # a few important columns first, then all the features
            # Important otherwise if some files have all nan in a feature
            # that other files had values for, it misaligns the feat matrix
            df.insert(0, 'file_id', file_id)
            df = sort_columns(df, selected_feat)
        except (
                AttributeError, IOError, KeyError,
                tables.exceptions.HDF5ExtError,
                tables.exceptions.NoSuchNodeError):
            continue
        else:
            # if nothing fails, write to the correct file
            # Get the filename summary line
            filenames = row.copy()
            if not df.empty:
                filenames['is_good'] = True
            # Store the filename summary line
            with open(f1, 'a') as fid:
                fid.write(','.join([str(x) for x in filenames.values])+"\n")

            # if no features were calculated, move on to next iteration
            if df.empty:
                continue

            # Abbreviate names
            if abbreviate_features:
                df = shorten_feature_names(df)

            # Store line(s) of features summaries for the given file
            # and given window
            # if we haven't written any data in a file, i.e. it only contains
            # comments or it is empty, we need to write the features names too
            is_write_header = _has_only_comments(f2)
            with open(f2, 'a') as fid:
                df.to_csv(fid, header=is_write_header, index=False)


def _has_only_comments(filepath, comment_character='#'):
    """
    _has_only_comments Read a file line by line, return False as soon as
    a line does not begin with comment_character.
    Return True if it loops to the end and all lines
    begin with comment_character.
    Return True on an empty file


    Parameters
    ----------
    filepath : [type]
        [description]
    comment_character: [str]
        character that identifies a comment
    """
    with open(filepath, 'r') as fid:
        for line in fid:
            if not line.startswith(comment_character):
                return False
    # got here without finding a non-comment line:
    return True

# %%
if __name__ == '__main__':

    import re
    import time
    from pathlib import Path
    from pandas.testing import assert_frame_equal

    # root_dir = \
    #     # '/Users/em812/Data/Tierpsy_GUI/test_results_multiwell/Syngenta/Results'
    #     # '/Users/em812/Data/Tierpsy_GUI/test_results_2'
    #     #'/Users/em812/Data/Tierpsy_GUI/test_results_multiwell/20190808_subset'
    # is_manual_index = False
    # feature_type = 'tierpsy'
    # # feature_type = 'openworm'
    # # summary_type = 'plate_augmented'
    # summary_type = 'plate'
    # #summary_type = 'trajectory'

    # Luigi
    root_dir = \
       '/Users/lferiani/Hackathon/multiwell_tierpsy/summaries/Results'

    # delete existing csvs
    for fname in Path(root_dir).rglob('*csv'):
        if fname.name.startswith('feat') or fname.name.startswith('filenames'):
            if 'groundtruth' not in fname.name:
                fname.unlink()

    is_manual_index = False
    feature_type = 'tierpsy'
    # feature_type = 'openworm'
    # summary_type = 'plate_augmented'
    # summary_type = 'plate'
    summary_type = 'trajectory'

    # fold_args = dict(
    #              n_folds = 2,
    #              frac_worms_to_keep = 0.8,
    #              time_sample_seconds = 10*60
    #              )
    kwargs = {
        'filter_time_min': '25',
        'filter_travel_min': '124',
        'filter_time_units': 'frame_numbers',
        'filter_distance_units': 'microns',
        'filter_length_min': '200',
        'filter_length_max': '2000',
        'filter_width_min': '20',
        'filter_width_max': '500'
        }

    time_windows = '0:end' #'0:100+200:300+350:400, 150:200' #'0:end:1000' #
    time_units = 'seconds'
    select_feat = 'all' #'tierpsy_2k'
    keywords_include = ''
    keywords_exclude = '' #'blob' #'curvature,velocity,norm,abs'
    abbreviate_features = False
    dorsal_side_known = False

    tic = time.time()
    df_files = calculate_summaries(
        root_dir, feature_type, summary_type, is_manual_index,
        abbreviate_features, dorsal_side_known,
        time_windows=time_windows, time_units=time_units,
        select_feat=select_feat, keywords_include=keywords_include,
        keywords_exclude=keywords_exclude,
        _is_debug=False, n_parallel=8, **kwargs)
        # **fold_args)

    print(f'Time elapsed: {time.time() - tic}s')
# %%
    # Luigi
#    df_files, all_summaries = calculate_summaries(
#         root_dir, feature_type, summary_type, is_manual_index,
#         time_windows, time_units, **fold_args)

    # check results are same as ground truth:
    other_csvs = []
    truth_csvs = []
    for csv in Path(root_dir).rglob('*.csv'):
        if csv.name.startswith('feat') or csv.name.startswith('filenames'):
            if 'groundtruth' in csv.name:
                continue
            dt_str = re.findall(r'\d{8}_\d{6}', csv.name)[0]
            other_csvs.append(csv)
            truth_csvs.append(
                csv.parent / csv.name.replace(dt_str, 'groundtruth')
            )

    def _read(fname):
        df = pd.read_csv(fname, comment='#', index_col=None)
        sorting_by = ['file_id', 'well_name', 'worm_index', 'i_fold']
        sorting_by = [c for c in sorting_by if c in df]
        df = df.sort_values(by=sorting_by).reset_index(drop=True)
        return df

    for truth_fname, candidate_fname in zip(truth_csvs, other_csvs):
        truth_df = _read(truth_fname)
        cand_df = _read(candidate_fname)
        assert_frame_equal(truth_df, cand_df, check_like=True)
    print('all good')
