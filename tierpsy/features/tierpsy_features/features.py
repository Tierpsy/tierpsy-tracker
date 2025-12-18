#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:01:03 2017

@author: ajaver
"""
import pandas as pd
import numpy as np
import warnings

from .helper import get_delta_in_frames, add_derivatives

from .velocities import get_velocity_features, velocities_columns
from .postures import get_morphology_features, morphology_columns, \
get_posture_features, posture_columns, posture_aux

from .curvatures import get_curvature_features, curvature_columns
from .food import get_cnt_feats, food_columns
from .path import get_path_curvatures, path_curvature_columns, path_curvature_columns_aux, path_straightness_columns

from .events import get_events, event_columns

#all time series features
timeseries_feats_no_dev_columns = velocities_columns + morphology_columns + posture_columns + \
                curvature_columns + food_columns + path_curvature_columns + path_straightness_columns

#add derivative columns
timeseries_feats_columns = timeseries_feats_no_dev_columns + ['d_' + x for x in timeseries_feats_no_dev_columns]

#add all the axiliary columns
aux_columns =  posture_aux + path_curvature_columns_aux
timeseries_all_columns = (timeseries_feats_columns + event_columns + aux_columns)


#add ventral features
ventral_signed_columns = ['relative_to_body_speed_midbody']
ventral_signed_columns += path_curvature_columns + curvature_columns
ventral_signed_columns += [x for x in velocities_columns if 'angular_velocity' in x]
ventral_signed_columns += [x for x in posture_columns if 'eigen_projection' in x]
ventral_signed_columns = ventral_signed_columns + ['d_' + x for x in ventral_signed_columns]

#all the ventral_signed_columns must be in timeseries_feats_columns
assert len(set(ventral_signed_columns) - set(timeseries_feats_columns))  == 0

valid_ventral_side = ('', 'clockwise','anticlockwise', 'unknown')

def get_timeseries_features(skeletons, 
                            widths = None, 
                            dorsal_contours = None, 
                            ventral_contours = None,
                            fps = 1,
                            derivate_delta_time = 1/3, 
                            ventral_side = '',
                            timestamp = None,
                            food_cnt = None,
                            is_smooth_food_cnt = False,
                            worm_blob_data = None
                            ):
    
    '''
    skeletons -> n_frames x n_segments x 2
    widths -> n_frames x n_segments
    dorsal_contours -> n_frames x n_segments x 2
    ventral_contours -> n_frames x n_segments x 2
    derivate_delta_time -> delta time in seconds used to calculate derivatives (including velocity)
    
    '''
    
    assert ventral_side in valid_ventral_side

    derivate_delta_frames = get_delta_in_frames(derivate_delta_time, fps)

    feat_morph = get_morphology_features(skeletons, widths, dorsal_contours, ventral_contours)
    feat_posture = get_posture_features(skeletons)
    
    #I am still missing the velocity and path features but it should look like this
    cols_to_use = [x for x in feat_posture.columns if x not in feat_morph] #avoid duplicate length
    
    features_df = feat_morph.join(feat_posture[cols_to_use])
    
    curvatures = get_curvature_features(skeletons)
    features_df = features_df.join(curvatures)
    
    velocities = get_velocity_features(skeletons, derivate_delta_frames, fps)
    if velocities is not None:
        features_df = features_df.join(velocities)
    
    if food_cnt is not None:
        food = get_cnt_feats(skeletons, 
                             food_cnt,
                             is_smooth_food_cnt
                             )
        features_df = features_df.join(food)
    
    
    path_curvatures, path_coords, path_straightness = get_path_curvatures(skeletons,fps)
    features_df = features_df.join(path_curvatures)
    features_df = features_df.join(path_coords)
    features_df = features_df.join(path_straightness)
    
    if timestamp is None:
        timestamp = np.arange(features_df.shape[0], np.int32)
        warnings.warn('`timestamp` was not given. I will assign an arbritary one.')
    
    features_df['timestamp'] = timestamp
    
    events_df = get_events(features_df, fps, worm_blob_data=worm_blob_data, skeletons=skeletons)
    
    dd = [x for x in events_df if x in event_columns]
    features_df = features_df.join(events_df[dd])
    
    #add the derivatives
    features_df = add_derivatives(features_df, 
                                  timeseries_feats_no_dev_columns, 
                                  derivate_delta_frames, 
                                  fps)
    
    #correct ventral side sign
    if ventral_side == 'clockwise':
        features_df[ventral_signed_columns] *= -1
    

    #add any missing column
    all_columns = ['timestamp'] + timeseries_all_columns
    df = pd.DataFrame([], columns = timeseries_all_columns)
    features_df = pd.concat((df, features_df), ignore_index=True, sort=False)

    features_df = features_df[all_columns]
            
    return features_df