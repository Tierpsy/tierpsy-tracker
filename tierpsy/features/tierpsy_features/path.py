#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:54:00 2017

@author: ajaver
"""
import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d

from .curvatures import curvature_grad
from .postures import get_length
from .helper import DataPartition, get_n_worms_estimate, fill_nans_1D

path_curvature_columns = ['path_curvature_body', 
                          'path_curvature_tail', 
                          'path_curvature_midbody', 
                          'path_curvature_head'
                          ]

path_curvature_columns_aux = ['coord_x_body', 'coord_y_body', 
                              'coord_x_tail', 'coord_y_tail',
                              'coord_x_midbody', 'coord_y_midbody', 
                              'coord_x_head', 'coord_y_head'
                              ]

path_straightness_columns = ['path_straightness_midbody',
                             'path_straightness_tail',
                             'path_straightness_body',
                             'path_straightness_head']

DFLT_ARGS = dict(
        path_step = 11,
        path_grad_window = 5,
        clip_val_body_lengths = 20,
        bin_size_microns = 250,
        bin_size_body_lengths = 0.25
        )

#%%
def _h_path_curvature(skeletons, 
                      body_length = None,
                      partition_str = 'body', 
                      path_step = DFLT_ARGS['path_step'], 
                      path_grad_window = DFLT_ARGS['path_grad_window'],
                      _is_debug = False):
    
    if body_length is None:
        #caculate the length if it is not given
        body_length = np.nanmedian(get_length(skeletons))
    #clip_val = clip_val_body_lengths/body_length

    p_obj = DataPartition(n_segments=skeletons.shape[1])
    body_coords = p_obj.apply(skeletons, partition_str, func=np.mean)
    

    xx = body_coords[:,0]
    yy = body_coords[:,1]
    tt = np.arange(body_coords.shape[0])
    
    #empty array return
    if body_coords.size == 0 or np.all(np.isnan(body_coords)):
        return np.full_like(tt, np.nan), body_coords


    #interpolate nan values
    good = ~np.isnan(xx)
    
    x_i = xx[good] 
    y_i = yy[good] 
    t_i = tt[good]
    
    t_i = np.hstack([-1, t_i, body_coords.shape[0]]) 
    x_i = np.hstack([x_i[0], x_i, x_i[-1]]) 
    y_i = np.hstack([y_i[0], y_i, y_i[-1]]) 
    
    fx = interp1d(t_i, x_i)
    fy = interp1d(t_i, y_i)
    
    xx_i = fx(tt)
    yy_i = fy(tt)
    
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(xx_i)
    dy = np.diff(yy_i)
    dr = np.sqrt(dx * dx + dy * dy)
    
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))
    
    fx = interp1d(lengths, xx_i)
    fy = interp1d(lengths, yy_i)
    ft = interp1d(lengths, tt)
    
    sub_lengths = np.arange(lengths[0], lengths[-1], path_step)
    
    #there is not enough data to calculate the curvature
    if len(sub_lengths) <= 4*path_grad_window:
        return np.full(skeletons.shape[0], np.nan), body_coords
    
    xs = fx(sub_lengths)
    ys = fy(sub_lengths)
    ts = ft(sub_lengths)
    
    curve = np.vstack((xs, ys)).T
    curvature_r = curvature_grad(curve, 
                                    points_window = path_grad_window, 
                                    axis=0,
                                    is_nan_border=False)
    
    #clip values to remove regions with extremely large curvatures (typically short reversars)
    #curvature_r = np.clip(curvature_r, -clip_val,clip_val)
    
    ts_i = np.hstack((-1, ts, tt[-1] + 1))
    c_i = np.hstack((curvature_r[0], curvature_r, curvature_r[-1]))
    curvature_t = interp1d(ts_i, c_i)(tt)
    
    if _is_debug:
        import matplotlib.pylab as plt
        from matplotlib.collections import LineCollection
        #path_curvature[np.isnan(worm_features['speed'])] = np.nan
        #path_curvature = np.clip(curvature_t, -0.02, 0.02)
        path_curvature = curvature_t
        
        curv_range = (np.nanmin(path_curvature), np.nanmax(path_curvature))
        
        points = body_coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, 
                            cmap = plt.get_cmap('plasma'),
                            norm = plt.Normalize(*curv_range))
        lc.set_array(path_curvature)
        lc.set_linewidth(2)
    
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1,2,1)
        plt.gca().add_collection(lc)
    
        plt.xlim(3000, 11000)
        plt.ylim(3000, 11000)
        plt.axis('equal')
        
        plt.subplot(1,2,2)
        plt.plot(path_curvature)
    
    return curvature_t, body_coords

def get_path_curvatures(skeletons, **argkws):
    path_curvatures = []    
    path_coords = []
    path_straightness_list = []
    body_length = np.nanmedian(get_length(skeletons))
    
    for partition_str in ['body', 'tail', 'midbody', 'head']:
        
        
        path_curv, coords = \
            _h_path_curvature(skeletons, 
                              body_length,
                              partition_str = partition_str, 
                              **argkws
                              )
        
        path_curvatures.append(('path_curvature_' + partition_str, path_curv))
        
        path_coords.append(('coord_x_' + partition_str, coords[...,0]))
        path_coords.append(('coord_y_' + partition_str, coords[...,1]))

        # Calculate straightness
        x = coords[:, 0]
        y = coords[:, 1]
        path_straightness_window = 125 # 5 seconds at 25 fps
        part_straightness = straightness(x, y, path_straightness_window)
        path_straightness_list.append(('path_straightness_' + partition_str, part_straightness))
        
    cols, dat = zip(*path_straightness_list)
    path_straightness_df = pd.DataFrame(np.array(dat).T, columns=cols)

    cols, dat = zip(*path_curvatures)
    
    path_curvatures_df = pd.DataFrame(np.array(dat).T, columns=cols)
    
    cols, dat = zip(*path_coords)
    path_coords_df = pd.DataFrame(np.array(dat).T, columns=cols)
    return path_curvatures_df, path_coords_df, path_straightness_df

def _test_plot_cnts_maps(ventral_contour, dorsal_contour):
    import matplotlib.pylab as plt
    pix2microns = 10
    
    x_min = np.nanmin(ventral_contour[:, :, 0])
    x_max = np.nanmax(ventral_contour[:, :, 0])
    
    y_min = np.nanmin(dorsal_contour[:, :, 1])
    y_max = np.nanmax(dorsal_contour[:, :, 1])
    
    
    rx = int(round((x_max - x_min)/pix2microns))
    ry = int(round((y_max - y_min)/pix2microns))
    
    size_counts = (rx + 1, ry + 1)
    
    partitions_dflt = {'head': (0, 8),
                            'neck': (8, 16),
                            'midbody': (16, 33),
                            'hips': (33, 41),
                            'tail': (41, 49),
                            'all': (0, 49),
                            'body': (8, 41)
                            }
    
    all_cnts = {}
    for part, rr in partitions_dflt.items():
        
        p_vc = ventral_contour[:, rr[0]:rr[1], :].astype(np.float32)
        p_dc = dorsal_contour[:, rr[0]:rr[1], :].astype(np.float32)
        h = np.hstack((p_vc[:, ], p_dc[:, ::-1, :], p_vc[:, 0, :][:, None, :]))
        
        
        cnts = [np.round((x-np.array((x_min, y_min))[None, :])/pix2microns) for x in h]
        
        
        counts = np.zeros(size_counts, np.float32)
        
        for ii, cnt in enumerate(cnts):
            if np.any(np.isnan(cnt)):
                continue
            cc = np.zeros(size_counts, np.float32)
            cc = cv2.drawContours(cc, [cnt[:, None, :].astype(np.int)], contourIdx=-1, thickness=-1, color=1)
            counts += cc
        
        plt.figure()
        plt.imshow(counts, interpolation='none')
        plt.title(part)
        
        all_cnts[part] = counts
    
        print(part)
      
def straightness(x, y, window):
    """
    Compute a *windowed straightness index* for an (x, y) trajectory.

    Straightness S_i for window starting at frame i is
    
        S_i = net_displacement / path_length
            = ||(x[i+w] – x[i], y[i+w] – y[i])|| /
              Σ_{k=i}^{i+w-1} ||(x[k+1] – x[k], y[k+1] – y[k])||

    Parameters
    ----------
    x, y : 1-D array-like
        Coordinates of the trajectory to measure (same length N).
    window : int
        Window size in frames (w ≥ 1).  A window spans frames
        i … i + w, inclusive, so the last valid start index is N – w – 1.

    Returns
    -------
    S : 1-D `numpy.ndarray`, shape (N,)
        Straightness time-series.  The final `window+1` entries are `np.nan`
        because a full window cannot be formed there.

    Notes
    -----
    * Uses **overlapping windows** that advance one frame at a time.

    """
    
    #  basic checks 
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of the same length")
    if window < 1 or not float(window).is_integer():
        raise ValueError("window must be a positive integer")

    N = x.size
    w = int(window)
    if N <= w:
        # Not enough data for even one window
        return np.full(N, np.nan)
    #  step-wise Euclidean distances 
    # d[k] = distance between frames k and k+1  (length N-1)
    dx = np.diff(x)
    dy = np.diff(y)
    d  = np.hypot(dx, dy)

    # cumulative path length so we can query any window in O(1)
    cum_d = np.concatenate(([0.0], np.cumsum(d)))
    
    #  path lengths for each window
    path_len = cum_d[w:] - cum_d[:-w]
    
    #  net displacements for window ends 
    net_dx = x[w:] - x[:-w]          # length N-w
    net_dy = y[w:] - y[:-w]
    net_disp = np.hypot(net_dx, net_dy)

    #  straightness for valid windows 
    S_valid = net_disp / path_len          # length N-w

    #  pad with NaNs to match original length 
    S = np.empty(N)
    S[:N-w]  = S_valid
    S[N-w:]  = np.nan                      # last (w) frames have no window

    return S

#%%
def _get_path_coverage_feats(timeseries_data, bin_size_microns):
    
    #find the columns that correspond to curvature_coords
    cols = [x for x in timeseries_data if x in path_curvature_columns_aux]
    path_coords_df = timeseries_data[cols]
    
    

    bin_vals = ((path_coords_df - path_coords_df.mean())/bin_size_microns).round()
    try:
        bin_vals = bin_vals.fillna(method='ffill').fillna(method='bfill').astype(np.int)
    except ValueError:
        #likely full of nan's return empty
        return {}

    path_coverage_feats = {}
    # loop over worm body parts
    for b_part in set(x.rpartition('_')[-1] for x in bin_vals.columns):
        # get the binned coordinates of the given body part 
        # (the coordinates defining in which square of the grid the worm body part is)
        dat = bin_vals[['coord_x_' + b_part,'coord_y_' + b_part]]
        dat.columns = ['X', 'Y']
        # groupby individual grid squares
        gg = dat.groupby(["X", "Y"])
        
        #here i am counting the number of times any worm enter to a given grid
        # count the number of worm occurancies in each grid square throughout the video
        grid_counts = gg.size().reset_index(name="Counts")
        #cc = pd.crosstab(dat['X'], dat['Y'])
        
        #now i want to assign a label to each grid each (worm_index, timestamp)
        ind_bins = np.full(dat.shape[0], -1)
        for ii, (k, vals) in enumerate(gg):
            ind_bins[vals.index] = ii
        df = timeseries_data[['worm_index']].copy()
        df['ind_bins'] = ind_bins
        
        #now i want to see the duration a given worm spend in each grid
        grid_durations = []
        for w, vec in df.groupby('worm_index'):
            xx = vec['ind_bins'].values
            xr = np.insert(xx[1:], xx.size-1, -1)
            
            b_flags = xr!=xx
            #b_id = xx[b_flags]
            b_s = np.diff(np.insert(np.where(b_flags)[0], 0, -1))
            grid_durations.append(b_s)

        if grid_durations:
            grid_durations = np.concatenate(grid_durations)
        else:
            grid_durations = np.zeros(0)

        path_coverage_feats[b_part] = (grid_counts, grid_durations)
        
        
    return path_coverage_feats
        
def get_path_extent_stats(timeseries_data, fps, is_normalized = False):
    
    if is_normalized:
        body_length = timeseries_data['length'].median()
        bin_size_microns = DFLT_ARGS['bin_size_body_lengths']*body_length
        area_per_grid = 1
        is_norm_str = '_norm'
    else:
        bin_size_microns = DFLT_ARGS['bin_size_microns']
        is_norm_str = ''
        area_per_grid = bin_size_microns**2
    
    path_coverage_feats = _get_path_coverage_feats(timeseries_data, bin_size_microns)
    
    Q = [50, 95]
    
    grid_stats = []
    for b_part, (grid_counts, grid_durations) in path_coverage_feats.items():
        if grid_durations.size > 0:
            grid_transit_time = np.percentile(grid_durations, Q)/fps
        else:
            grid_transit_time = (np.nan, np.nan)

        if grid_counts['Counts'].size > 0:
            path_coverage = grid_counts['Counts'].size*area_per_grid
            path_density = np.percentile(grid_counts['Counts'], Q)/grid_counts['Counts'].sum()
        else:
            path_coverage = np.nan 
            path_density = (np.nan, np.nan)
        
        posfix = b_part + is_norm_str
        grid_stats += [
                (path_coverage, 'path_coverage_' + posfix),
                (path_density[0], 'path_density_{}_{}th'.format(posfix, Q[0])),
                (path_density[1], 'path_density_{}_{}th'.format(posfix, Q[1])),
                (grid_transit_time[0], 'path_transit_time_{}_{}th'.format(posfix, Q[0])),
                (grid_transit_time[1], 'path_transit_time_{}_{}th'.format(posfix, Q[1])),
                ]
    
    grid_stats_s = pd.Series(*list(zip(*grid_stats)))
    return grid_stats_s

#%%
if __name__ == '__main__':
    import os
    import tables
    
    
    #%%
    #_test_plot_cnts_maps(ventral_contour, dorsal_contour)
    base_dir = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm'
    skeletons_file = os.path.join(base_dir, 'MY23_worms5_food1-10_Set4_Pos5_Ch4_29062017_140148_skeletons.hdf5')
    features_file = skeletons_file.replace('_skeletons.hdf5', '_featuresN.hdf5')
    
    #features_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/MY16_worms5_food1-10_Set5_Pos4_Ch1_02062017_131004_featuresN.hdf5'
    features_file = '/Users/ajaver/OneDrive - Imperial College London/tierpsy_features/test_data/multiworm/170817_matdeve_exp7co1_12_Set0_Pos0_Ch1_17082017_140001_featuresN.hdf5'
    
    with pd.HDFStore(features_file, 'r') as fid:
        blob_features = fid['/blob_features']
        trajectories_data = fid['/trajectories_data']
        timeseries_data = fid['/timeseries_data']
        
        fps = fid.get_storer('/trajectories_data').attrs['fps']
        good = trajectories_data['skeleton_id']>=0
        trajectories_data = trajectories_data[good]
        blob_features = blob_features[good]
        
    if False:
        
        trajectories_data_g = trajectories_data.groupby('worm_index_joined')
        
        
        for worm_index in trajectories_data_g.groups.keys():
            worm_index = 4#695
            worm_data = trajectories_data_g.get_group(worm_index)
            skel_id = worm_data['skeleton_id'].values 
            with tables.File(features_file, 'r') as fid:
                skeletons = fid.get_node('/coordinates/skeletons')[skel_id, :, :]
            worm_features = timeseries_data.loc[skel_id]
            
            path_curvatures_df, path_coords_df = get_path_curvatures(skeletons, _is_debug=True)
            break
    #%%
    get_path_extent_stats(timeseries_data)
    
    