#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:31:24 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables

from tierpsy.features.tierpsy_features import get_timeseries_features, timeseries_all_columns
from tierpsy.features.tierpsy_features.summary_stats import get_summary_stats

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, TABLE_FILTERS
from tierpsy.helper.params import read_fps, read_ventral_side

from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import was_fov_split

def save_timeseries_feats_table(
        features_file, derivate_delta_time, splitfov_json_path=''):
    timeseries_features = []
    fps = read_fps(features_file)

    # initialise class for splitting fov
    if splitfov_json_path:
        is_fov_tosplit = True
    else:
        is_fov_tosplit = False
    print('is fov to split?',is_fov_tosplit)

    if is_fov_tosplit:
        # split fov in wells
        masked_image_file = features_file.replace(
            'Results', 'MaskedVideos').replace('_featuresN.hdf5', '.hdf5')

        fovsplitter = FOVMultiWellsSplitter(
            masked_image_file, splitfov_json_path)
        # store wells data in the features file
        fovsplitter.write_fov_wells_to_file(features_file)


    with pd.HDFStore(features_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        blob_features = fid['/blob_features'] if '/blob_features' in fid else None
    # add worm indices and timestamp columns from trajectories data to blob features to use in grouping
    if blob_features is not None:
        blob_features_with_idx = pd.concat([blob_features,
                                            trajectories_data[['worm_index_joined', 'timestamp_raw']]], axis=1)
        
    blob_features_g = blob_features_with_idx.groupby('worm_index_joined')
    trajectories_data_g = trajectories_data.groupby('worm_index_joined')
    progress_timer = TimeCounter('')
    base_name = get_base_name(features_file)
    tot_worms = len(trajectories_data_g)
    def _display_progress(n):
            # display progress
        dd = " Calculating tierpsy features. Worm %i of %i done." % (n+1, tot_worms)
        print_flush(
            base_name +
            dd +
            ' Total time:' +
            progress_timer.get_time_str())

    _display_progress(0)
    with tables.File(features_file, 'r+') as fid:

        for gg in ['/timeseries_data', '/event_durations', '/timeseries_features']:
            if gg in fid:
                fid.remove_node(gg)


        feat_dtypes = [(x, np.float32) for x in timeseries_all_columns]

        feat_dtypes = [('worm_index', np.int32),
                       ('timestamp', np.int32),
                       ('well_name', 'S3')] + feat_dtypes

        timeseries_features = fid.create_table(
                '/',
                'timeseries_data',
                obj = np.recarray(0, feat_dtypes),
                filters = TABLE_FILTERS)

        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]
        else:
            food_cnt = None

        #If i find the ventral side in the multiworm case this has to change
        ventral_side = read_ventral_side(features_file)

        for ind_n, (worm_index, worm_data) in enumerate(trajectories_data_g):
            # extract the corresponding blob features
            if blob_features is not None:
                worm_blob_data = blob_features_g.get_group(worm_index)
            else:
                worm_blob_data = None

            skel_id = worm_data['skeleton_id'].values

            #deal with any nan in the skeletons
            good_id = skel_id>=0
            skel_id_val = skel_id[good_id]
            traj_size = skel_id.size

            args = []
            for p in ('skeletons', 'widths', 'dorsal_contours', 'ventral_contours'):

                node_str = '/coordinates/' + p
                if node_str in fid:
                    node = fid.get_node(node_str)
                    dat = np.full((traj_size, *node.shape[1:]), np.nan)
                    if skel_id_val.size > 0:
                        if len(node.shape) == 3:
                            dd = node[skel_id_val, :, :]
                        else:
                            dd = node[skel_id_val, :]
                        dat[good_id] = dd
                else:
                    dat = None

                args.append(dat)

            timestamp = worm_data['timestamp_raw'].values.astype(np.int32)

            feats = get_timeseries_features(*args,
                                           timestamp = timestamp,
                                           food_cnt = food_cnt,
                                           fps = fps,
                                           ventral_side = ventral_side,
                                           derivate_delta_time = derivate_delta_time,
                                           worm_blob_data = worm_blob_data
                                           )
            #save timeseries features data
            feats = feats.astype(np.float32)
            feats['worm_index'] = worm_index
            if is_fov_tosplit:
                feats['well_name'] = fovsplitter.find_well_from_trajectories_data(worm_data)
            else:
                feats['well_name'] = 'n/a'
            # cast well_name to the correct type
            # (before shuffling columns, so it remains the last entry)
            # needed because for some reason this does not work:
            # feats['well_name'] = feats['well_name'].astype('S3')
            feats['_well_name'] = feats['well_name'].astype('S3')
            feats.drop(columns='well_name', inplace=True)
            feats.rename(columns={'_well_name':'well_name'}, inplace=True)

            #move the last fields to the first columns
            cols = feats.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            cols[1], cols[2] = cols[2], cols[1]

            feats = feats[cols]

            feats['worm_index'] = feats['worm_index'].astype(np.int32)
            feats['timestamp'] = feats['timestamp'].astype(np.int32)
            feats = feats.to_records(index=False)

            timeseries_features.append(feats)
            _display_progress(ind_n)


def save_feats_stats(features_file, derivate_delta_time):
    with pd.HDFStore(features_file, 'r') as fid:
        fps = fid.get_storer('/trajectories_data').attrs['fps']
        timeseries_data = fid['/timeseries_data']
        blob_features = fid['/blob_features'] if '/blob_features' in fid else None
        is_fov_tosplit = was_fov_split(features_file) # do we need split-FOV sumaries?

    # check
    if is_fov_tosplit:
        assert 'well_name' in timeseries_data.columns, (
            'fov_wells in features file but no well_name in timeseries_data')

    #Now I want to calculate the stats of the video
    if is_fov_tosplit:
        # get summary stats per well and then concatenate them all
        well_name_list = list(set(timeseries_data['well_name']) - set(['n/a']))
        exp_feats = []
        for wc, well in enumerate(well_name_list):
            print('Processing well {} out of {}'.format(wc, len(well_name_list)))
            idx = timeseries_data['well_name'] == well
            # calculate stats per well
            tmp = get_summary_stats(timeseries_data[idx].reset_index(),
                                    fps,
                                    blob_features[idx].reset_index(),
                                    derivate_delta_time)
            tmp = pd.DataFrame(zip(tmp.index, tmp), columns=['name','value'])
            tmp['well_name'] = well
            exp_feats.append(tmp)

        # now concat all
        exp_feats = pd.concat(exp_feats, ignore_index=True)

    else: # we don't need to split the FOV

        exp_feats = get_summary_stats(timeseries_data,
                                      fps,
                                      blob_features,
                                      derivate_delta_time)

    # save on disk
    # now if is_fov_tosplit exp_feats is a dataframe, otherwise a series
    if len(exp_feats)>0:

        # different syntax according to df or series
        if is_fov_tosplit:
            tot = max(len(x) for x in exp_feats['name'])
            dtypes = {'name':'S{}'.format(tot), 'value':np.float32, 'well_name':'S3'}
            exp_feats_rec = exp_feats.to_records(index=False, column_dtypes=dtypes)
        else:
            tot = max(len(x) for x in exp_feats.index)
            dtypes = [('name', 'S{}'.format(tot)), ('value', np.float32)]
            exp_feats_rec = np.array(list(zip(exp_feats.index, exp_feats)), dtype = dtypes)

        # write on hdf5 file
        with tables.File(features_file, 'r+') as fid:
            for gg in ['/features_stats']:
                if gg in fid:
                    fid.remove_node(gg)
            fid.create_table(
                    '/',
                    'features_stats',
                    obj = exp_feats_rec,
                    filters = TABLE_FILTERS)



def get_tierpsy_features(features_file, derivate_delta_time = 1/3, splitfov_json_path=''):
    #I am adding this so if I add the parameters to calculate the features i can pass it to this function
    save_timeseries_feats_table(features_file, derivate_delta_time, splitfov_json_path)
    save_feats_stats(features_file, derivate_delta_time)



if __name__ == '__main__':

    base_file = '/Users/lferiani/Desktop/Data_FOVsplitter/24WP/Results/test_24well_run1_prestim_20220128_165153.22594548/metadata'
    features_file = base_file + '_featuresN.hdf5'

    # restore features after previous step before testing
    import os
    import shutil
    from tierpsy import DFLT_SPLITFOV_PARAMS_PATH
    shutil.copy(features_file.replace('.hdf5','.bk'), features_file)
    splitfov_json = os.path.join(
        DFLT_SPLITFOV_PARAMS_PATH, 'HYDRA_94WP_UPRIGHT.json')

    get_tierpsy_features(features_file,
                         derivate_delta_time = 1/3,
                         splitfov_json_path=splitfov_json)



#    #%%
#    from tierpsy.features.tierpsy_features.velocities import _h_get_velocity
#    from tierpsy.features.tierpsy_features.helper import get_delta_in_frames
#
#    fname = '/Users/avelinojaver/Desktop/small_worms/Results/20191121_featuresN.hdf5'
#
#    delta_time = 0.3
#
#
#    with pd.HDFStore(fname, 'r') as fid:
#        fps = fid.get_storer('/trajectories_data').attrs['fps']
#        blob_features = fid['/blob_features']
#        trajectories_data = fid['/trajectories_data']
#        trajectories_data_g = trajectories_data.groupby('worm_index_joined').groups
#        derivate_delta_frames = get_delta_in_frames(derivate_delta_time, fps)
#        for ind_n, (worm_index, indexes) in enumerate(trajectories_data_g.items()):
#            coords = blob_features.loc[indexes, ['coord_x', 'coord_y']].values
#            velocity = _h_get_velocity(coords, derivate_delta_frames, fps)
#
#            print(velocity.shape)
