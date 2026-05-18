# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:33:34 2015

@author: ajaver
"""

import os

import numpy as np
import pandas as pd
import tables
from scipy.spatial.distance import cdist

from tierpsy.helper.misc import TimeCounter, print_flush, TABLE_FILTERS

def assignBlobTraj(trajectories_file, max_allowed_dist=20, area_ratio_lim=(0.5, 2)):
    #loop, save data and display progress
    base_name = os.path.basename(trajectories_file).replace('_trajectories.hdf5', '').replace('_skeletons.hdf5', '')
    
    with pd.HDFStore(trajectories_file, 'r') as fid:
        plate_worms = fid['/plate_worms']
    
    traj_ind = assignBlobTrajDF(plate_worms, max_allowed_dist, area_ratio_lim, base_name=base_name)

    if traj_ind is not None:
        with tables.File(trajectories_file, 'r+') as fid:
            tbl = fid.get_node('/', 'plate_worms')
            tbl.modify_column(column=traj_ind, colname='worm_index_blob')

        #print_flush(progress_time.get_str(frame))    

def assignBlobTrajDF(traj_df, max_allowed_dist, area_ratio_lim, base_name=''):
    
    def _get_cost_matrix(frame_data, frame_data_prev):
        coord = frame_data[['coord_x', 'coord_y']].values
        coord_prev = frame_data_prev[['coord_x', 'coord_y']].values
        costMatrix = cdist(coord_prev, coord)  # calculate the cost matrix
        
        # assign a large value to non-valid combinations by area
        area = frame_data['area'].values
        area_prev = frame_data_prev['area'].values
        area_ratio = area_prev[:, None]/area[None,:]
        area_ratio[np.isnan(area_ratio)] = 1e20
        
        bad_ratio = (area_ratio<area_ratio_lim[0]) | \
        (area_ratio>area_ratio_lim[1]) | \
        np.isnan(costMatrix)
        
        costMatrix[bad_ratio] = 1e20
        return costMatrix
    
    def _get_prev_ind_match(costMatrix):
        def _label_bad_ind(indexes, dist, max_allowed_dist):
            #label as bad the pairs that have a distance larger than max_allowed_dist
            indexes[dist>max_allowed_dist] = -1
            #remove indexes that where assigned twice (either a merge or a split event)
            uind, counts = np.unique(indexes, return_counts=True)
            duplicated_ind = uind[counts>1]
            bad_ind = np.in1d(indexes, duplicated_ind) 
            indexes[bad_ind] = -1
            return indexes
        
        #I get the corresponding index in the previous data_frame
        #I remove pairs located at positions larger than max_allowed_dist
        #And indexes that where assigned twice or more (split events)
        map_to_prev = np.argmin(costMatrix, axis=0) #must have dimensions of frame_data
        min_dist_pp = costMatrix[map_to_prev, np.arange(costMatrix.shape[1])]
        _label_bad_ind(map_to_prev, min_dist_pp, max_allowed_dist)
        
        #here i am looking at in the prev indexes that would have been 
        #assigned twice or more to the next indexes (merge events)
        map_to_next = np.argmin(costMatrix, axis=1) #must have dimensions of frame_data_prev
        min_dist_pp = costMatrix[np.arange(costMatrix.shape[0]), map_to_next]
        _label_bad_ind(map_to_next, min_dist_pp, max_allowed_dist)
        
        
        bad_prev_ind =  np.where(map_to_next==-1)[0] #techincally either index too far away or duplicated
        possible_merges = np.in1d(map_to_prev, bad_prev_ind) 
        map_to_prev[possible_merges] = -1
        return map_to_prev
     
    
    frame_data_prev = None
    tot_worms = 0
    all_indexes = []
    frames_grouped = traj_df.groupby('frame_number')
    
    #if isinstance(area_ratio_lim, (float, int)):
    #    area_ratio_lim = (1/area_ratio_lim, area_ratio_lim)
    
    progress_time = TimeCounter(base_name + ' Assigning trajectories.', len(frames_grouped))  
    for frame, frame_data in frames_grouped:
        #what happens if the frames are not continous?
        if frame_data_prev is not None:    
            _, prev_traj_ind = all_indexes[-1]
            costMatrix = _get_cost_matrix(frame_data, frame_data_prev)
            map_to_prev = _get_prev_ind_match(costMatrix)
            
            traj_indexes = np.zeros_like(map_to_prev)
            unmatched = map_to_prev == -1
            matched = ~unmatched
            
            #assign matched index from the previous indexes
            traj_indexes[matched] = prev_traj_ind[map_to_prev[matched]]
            
            vv = np.arange(1, np.sum(unmatched) + 1) + tot_worms
            if vv.size > 0:
                tot_worms = vv[-1]
                traj_indexes[unmatched] = vv
                
        else:
            # initialize worm indexes
            traj_indexes = tot_worms + np.arange(1, len(frame_data) + 1)
            tot_worms = traj_indexes[-1]
        
        all_indexes.append((frame_data.index, traj_indexes))
            
        
        frame_data_prev = frame_data
        if frame % 500 == 0:
            # calculate the progress and put it in a string
            print_flush(progress_time.get_str(frame))
    
    if all_indexes:
        row_ind, traj_ind = map(np.concatenate, zip(*all_indexes))
        traj_ind = traj_ind[np.argsort(row_ind)]
        return traj_ind
    
#%%
def _validRowsByArea(plate_worms, use_center = True):
    #%%
    # here I am assuming that most of the time the largest area in the frame is a worm. Therefore a very large area is likely to be
    # noise
    groupsbyframe = plate_worms.groupby('frame_number')
    maxAreaPerFrame = groupsbyframe.agg({'area': 'max'})
    
    med_area = np.median(maxAreaPerFrame)
    mad_area = np.median(np.abs(maxAreaPerFrame - med_area))
    min_area = med_area - mad_area * 6
    max_area = med_area + mad_area * 6

    groupByIndex = plate_worms.groupby('worm_index_blob')

    median_area_by_index = groupByIndex.agg({'area': np.median})

    good = ((median_area_by_index > min_area) & (
        median_area_by_index < max_area)).values
    valid_ind = median_area_by_index[good].index

    plate_worms_f = plate_worms[plate_worms['worm_index_blob'].isin(valid_ind)]
    
   
    
    if use_center:
         # median location, it is likely the worm spend more time here since the
         # stage movements tries to get it in the centre of the frame
         # This assuption could case errors if the videos were not taken using WT2
        CMx_med, CMy_med = plate_worms_f[['coord_x', 'coord_y']].median()
    else:
        area_med = plate_worms_f['area'].median() #let's use the median area to select what particle to use as seed
    
    # let's use a threshold of movement of at most a quarter of the worm size,
    # otherwise we discard frame.
    L_med = plate_worms_f['box_length'].median()
    L_th = L_med / 4

    # now if there are still a lot of valid blobs we decide by choising the
    # closest blob
    valid_rows = []
    tot_frames = plate_worms['frame_number'].max() + 1
    
    def get_valid_indexes(frame_number, prev_row):
        
        try:
            current_group_f = groupbyframe_f.get_group(frame_number)
        except KeyError:
            # there are not valid index in the current group
            prev_row = None
            return prev_row
        
        
        th_cost = L_th
        
        # pick the closest blob if there are more than one blob to pick
        if prev_row is not None:
            #the closest to the previous one if there are prev_rows
            delX = current_group_f['coord_x'] - prev_row['coord_x']
            delY = current_group_f['coord_y'] - prev_row['coord_y']
            _cost = np.sqrt(delX * delX + delY * delY)
       
        else:
            #we need a seed trajectory...
            if use_center:
                #here we use the trajectory that is closest to the centre (this is a good assumption in WT2 since the worm is moving around)
                delX = current_group_f['coord_x'] - CMx_med
                delY = current_group_f['coord_y'] - CMy_med
                _cost = np.sqrt(delX * delX + delY * delY)
            else:
                _cost = (current_group_f['area'] - area_med).abs()/area_med
                th_cost = 0.5
        
        #get the minimum cost
        best_ind = _cost.idxmin()
        if _cost[best_ind] < th_cost:
            prev_row = current_group_f.loc[best_ind]
            valid_rows.append(best_ind)
        else:
            prev_row = None

        return prev_row
    
    # group by frame
    groupbyframe_f = plate_worms_f.groupby('frame_number')

    prev_row = None
    first_frame = tot_frames
    for frame_number in range(tot_frames):
        prev_row = get_valid_indexes(frame_number, prev_row)
        if not isinstance(prev_row, int) and first_frame > frame_number:
            first_frame = frame_number
    
    # if the first_frame is larger than zero it means that it might have lost some data in from the begining
    # let's try to search again from opposite direction
    if frame_number > 0 and len(valid_rows) > 0:
        prev_row = plate_worms_f.loc[np.min(valid_rows)]
        for frame_number in range(frame_number, -1, -1):
            prev_row = get_valid_indexes(frame_number, prev_row)

    
    return valid_rows
#%%

def correctSingleWormCase(trajectories_file, is_WT2):
    '''
    Only keep the object with the largest area when cosider the case of individual worms.
    '''
    with pd.HDFStore(trajectories_file, 'r') as traj_fid:
        plate_worms = traj_fid['/plate_worms']

    # emtpy table nothing to do here
    if len(plate_worms) == 0:
        return
    
    valid_rows = _validRowsByArea(plate_worms, use_center = is_WT2)

    # np.array(1, dtype=np.int32)
    worm_index_joined = np.full(len(plate_worms), -1, dtype=np.int32)
    worm_index_joined[valid_rows] = 1



    with tables.open_file(trajectories_file, mode='r+') as fid:
        plate_worms = fid.get_node('/plate_worms')
        # add the result the column worm_index_joined
        plate_worms.modify_column(
            colname='worm_index_joined',
            column=worm_index_joined)
        fid.flush()
        

def joinGapsTrajectories(trajectories_file, 
                         min_track_size=50,
                         max_frames_gap=100, 
                         area_ratio_lim=(0.67, 1.5)):
    #% get the first and last rows for each trajectory. Pandas is easier of manipulate than tables.
    with pd.HDFStore(trajectories_file, 'r') as fid:
        df = fid['plate_worms'][['worm_index_blob', 'frame_number',
                                 'coord_x', 'coord_y', 'area', 'box_length']]

    worm_index_joined = joinGapsTrajectoriesDF(df, 
                           min_track_size=min_track_size,
                           max_frames_gap=max_frames_gap, 
                           area_ratio_lim=area_ratio_lim
                           )

    # update worm_index_joined field
    with tables.open_file(trajectories_file, mode='r+') as fid:
        plate_worms = fid.get_node('/plate_worms')
        # add the result the column worm_index_joined
        plate_worms.modify_column(
            colname='worm_index_joined',
            column=worm_index_joined)
        fid.flush()

def joinGapsTrajectoriesDF(plate_worms, 
                           min_track_size=50,
                           max_frames_gap=100, 
                           area_ratio_lim=(0.67, 1.5),
                           worm_index_type='worm_index_blob'):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames
    min_track_size -- minimum tracksize accepted
    max_frames_gap -- time gap between joined trajectories
    '''



    def _findNextTraj(df, 
                      area_ratio_lim, 
                      min_track_size, 
                      max_frames_gap):
        '''
        area_ratio_lim -- allowed range between the area ratio of consecutive frames
        min_track_size -- minimum tracksize accepted
        max_frames_gap -- time gap between joined trajectories
        '''
    
        df = df[[worm_index_type, 'frame_number',
                 'coord_x', 'coord_y', 'area', 'box_length']].dropna()
        # select the first and last frame_number for each separate trajectory
        tracks_data = df[[worm_index_type, 'frame_number']]
        tracks_data = tracks_data.groupby(worm_index_type)
        tracks_data = tracks_data.aggregate(
            {'frame_number': ['idxmin', 'idxmax', 'count']})
    
        # filter data only to include trajectories larger than min_track_size
        tracks_data = tracks_data[
            tracks_data['frame_number']['count'] >= min_track_size]
        valid_indexes = tracks_data.index
    
        # select the corresponding first and last rows of each trajectory
        first_rows = df.loc[tracks_data['frame_number']['idxmin'].values]
        last_rows = df.loc[tracks_data['frame_number']['idxmax'].values]
        # let's use the particle id as index instead of the row number
        last_rows.index = tracks_data['frame_number'].index
        first_rows.index = tracks_data['frame_number'].index
    
        #% look for trajectories that could be join together in a small time gap
        join_frames = []
        for curr_index in valid_indexes:
            # the possible connected trajectories must have started after the end of the current trajectories,
            # within a timegap given by max_frames_gap
            possible_rows = first_rows[
                (first_rows['frame_number'] > last_rows['frame_number'][curr_index]) & (
                    first_rows['frame_number'] <= last_rows['frame_number'][curr_index] +
                    max_frames_gap)]
    
            # the area change must be smaller than the one given by area_ratio_lim
            # it is better to use the last point change of area because we are
            # considered changes near that occur near time
            areaR = last_rows['area'][curr_index] / possible_rows['area']
            possible_rows = possible_rows[
                (areaR > area_ratio_lim[0]) & (
                    areaR < area_ratio_lim[1])]
    
            # not valid rows left
            if len(possible_rows) == 0:
                continue
    
            R = np.sqrt((possible_rows['coord_x'] -
                         last_rows['coord_x'][curr_index]) ** 2 +
                        (possible_rows['coord_y'] -
                         last_rows['coord_y'][curr_index]) ** 2)
    
            indmin = int(np.argmin(R.values))
            next_index = R.index[indmin]
            # only join trajectories that move at most one worm body
            if R.iloc[indmin] <= last_rows.loc[curr_index, 'box_length']:
                #print(curr_index, next_index)
                join_frames.append((next_index, curr_index))
    
        relations_dict = dict(join_frames)
    
        return relations_dict, valid_indexes
    
    
    def _joinDict2Index(worm_index, relations_dict, valid_indexes):
        worm_index_new = np.full_like(worm_index, -1)
    
        for ind in valid_indexes:
            # seach in the dictionary for the first index in the joined trajectory
            # group
            ind_joined = ind
            while ind_joined in relations_dict:
                ind_joined = relations_dict[ind_joined]
    
            # replace the previous index for the root index
            worm_index_new[worm_index == ind] = ind_joined
    
        return worm_index_new
    
    
    
    relations_dict, valid_indexes = _findNextTraj(plate_worms, area_ratio_lim, min_track_size, max_frames_gap)
    # read the worm_index_blob column, this is the index order that have to
    # be conserved in the worm_index_joined column
    worm_index_blob = plate_worms[worm_index_type].values
    worm_index_joined = _joinDict2Index(worm_index_blob, relations_dict, valid_indexes)

    return worm_index_joined



def joinBlobsTrajectories(trajectories_file, 
                          is_one_worm, 
                          is_WT2,
                          max_allowed_dist, 
                          area_ratio_lim, 
                          min_track_size,
                          max_frames_gap):
    
    #allow to recieve int/float values
    if not isinstance(area_ratio_lim, (tuple,list)):
        area_ratio_lim = (1/area_ratio_lim, area_ratio_lim)

    
    assignBlobTraj(trajectories_file, max_allowed_dist, area_ratio_lim)
    if is_one_worm:
        correctSingleWormCase(trajectories_file, is_WT2)
    else:
        joinGapsTrajectories(trajectories_file, min_track_size, max_frames_gap, area_ratio_lim)

    with tables.File(trajectories_file, "r+") as traj_fid:
        traj_fid.get_node('/plate_worms')._v_attrs['has_finished'] = 2
        traj_fid.flush()