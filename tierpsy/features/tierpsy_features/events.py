#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ajaver
"""

import numpy as np
import pandas as pd
from tierpsy.features.tierpsy_features.helper import nanmedian_filter, fill_nans_1D
event_columns = ['motion_mode','behavioural_states', 'food_region', 'turn']
durations_columns = ['event_type', 'region',
                     'duration', 'timestamp_initial',
                     'timestamp_final', 'edge_flag']
event_region_labels = {
            'motion_mode': {-1:'backward', 1:'forward', 0:'paused',},
            'behavioural_states': {0: 'quiescence', 1: 'dwelling', 2: 'roaming', 3: 'sprinting'},
            'food_region': {-1:'outside', 1:'inside', 0:'edge'},
            'turn': {1:'intra', 0:'inter'}
            }

assert set(event_region_labels.keys()).issubset(event_columns)

#%%
def _get_pulses_indexes(light_on, min_window_size=0, is_pad = True):
    '''
    Get the start and end of a given pulse.
    '''

    if is_pad:

        light_on = np.pad(light_on, (1,1), 'constant', constant_values = False)

    switches = np.diff(light_on.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)

    if is_pad:
        turn_on -= 1
        turn_off -= 1
        turn_on = np.clip(turn_on, 0, light_on.size-3)
        turn_off = np.clip(turn_off, 0, light_on.size-3)


    assert turn_on.size == turn_off.size

    delP = turn_off - turn_on

    good = delP > min_window_size

    return turn_on[good], turn_off[good]

#%%
def _find_turns(worm_data, fps, ang_vel_thresh = 0.85, smooth_window_sec = 1.2):
    """
    This function detects turns based on the angular velocity of the worm's midbody.
    It smooths the angular velocity and flags frames where the absolute value exceeds a threshold.

    Parameters:
    - worm_data : DataFrame with skeleton or centroid positions.
    - fps : Frames per second of the video.
    - ang_vel_thresh : Angular velocity threshold to detect turns.
    - smooth_window_sec : Smoothing window size in seconds.

    Returns:
    - turn_vector : Boolean array, True at frames where a turn is detected.
    """
    import numpy as np
    import pandas as pd

    # ********************************
    # temporarily import parameters
    import yaml

    # find the absolute path of the extras directory in the tierpsy-tracker package based on the location of this file
    import os
    events_path = os.path.abspath(__file__)
    
    # find parameters_combo.yaml in the extras directory
    extras_dir = events_path.replace('/features/tierpsy_features/events.py', '/extras')
    param_combo_file_name = 'parameters_combo.yaml'
    PARAM_COMBO_FILE = os.path.join(extras_dir, param_combo_file_name)
    with open(PARAM_COMBO_FILE, 'r') as f:
        params = yaml.safe_load(f)

    ang_vel_thresh = params['angular_velocity_threshold']
    # smooth_window_sec = params['smoothing_window_sec']
    delta_frames = params['delta_frames']
    # End of temporary import
    # ********************************

    # Ensure required columns are present
    if 'angular_velocity' not in worm_data:
        raise ValueError("worm_data must contain 'angular_velocity' column.")

    # smooth_window_frames = int(smooth_window_sec * fps)
    # if smooth_window_frames % 2 == 0:
    #     smooth_window_frames += 1  # Ensure odd window size for centered smoothing
    
    # calculate angular velocity from skeletons
    if skeletons is not None:
        from tierpsy.features.tierpsy_features.velocities import get_velocity
        partition = 'body'
        signed_speed, angular_velocity, centered_skeleton = get_velocity(skeletons, partition, delta_frames, fps)

    # interpolate over NaNs and use absolute value of angular velocity
    interpolated_angular_velocity = fill_nans_1D(angular_velocity,max_gap=10*fps)
    ang_velocity = np.abs(interpolated_angular_velocity)

    # Detect potential turns based on threshold
    potential_turn_vector = ang_velocity > ang_vel_thresh
    
    # apply a filter on blob movement based on coordinates change (coord_x, coord_y) a few frames before and after the turn
    # if worm_blob_data is not None and 'coord_x' in worm_blob_data and 'coord_y' in worm_blob_data:
    #     coord_x = worm_blob_data['coord_x'].values
    #     coord_y = worm_blob_data['coord_y'].values
        
    #     # Calculate rolling window size (e.g., 1 seconds before and after the turn)
    #     rolling_window_size = int(fps * 1)
    #     if rolling_window_size % 2 == 0:
    #         rolling_window_size += 1
        
    #     # Calculate cumulative distance traveled (sum of frame-to-frame distances)
    #     # dx = np.diff(coord_x, prepend=coord_x[0])
    #     # dy = np.diff(coord_y, prepend=coord_y[0])
    #     # frame_distances = np.sqrt(dx**2 + dy**2)
    #     # cumulative_distance = pd.Series(frame_distances).rolling(
    #     #     window=rolling_window_size, center=True, min_periods=1).sum().values
        
    #     # Calculate net displacement (straight-line distance from start to end of window)
    #     net_displacement = pd.Series(np.sqrt(
    #         (coord_x - np.roll(coord_x, rolling_window_size//2))**2 + 
    #         (coord_y - np.roll(coord_y, rolling_window_size//2))**2
    #     )).rolling(window=rolling_window_size, center=True, min_periods=1).max().values
        
    #     # Movement is valid only if net displacement is equal or larger than a fraction of worm length
    #     worm_length = worm_data['length'].median()
    #     movement_ratio_threshold = 0.2  # e.g., net displacement should be at least 30% of worm length
    #     movement_mask = net_displacement >= (movement_ratio_threshold * worm_length)
        
    #     potential_turn_vector = potential_turn_vector & movement_mask


    
    # apply a minimum "blob_compactness" (from worm_blob_data) filter making sure there's at least one frame with high compactness in the surrounding frames
    blob_compactness = worm_blob_data['compactness'] if worm_blob_data is not None else None
    if blob_compactness is not None:
        min_compactness = 0.4
        compactness_mask = blob_compactness > min_compactness
        # create a rolling window to ensure at least one frame in the surrounding frames has high compactness
        rolling_window_size = int(fps * 2)
        if rolling_window_size % 2 == 0:
            rolling_window_size += 1  # Ensure odd window size
        compactness_mask_rolled = pd.Series(compactness_mask).rolling(
            window=rolling_window_size, center=True, min_periods=1).max().values.astype(bool)
        # Update potential_turn_vector to only keep turns where compactness condition is met
        turn_vector = potential_turn_vector & compactness_mask_rolled
    
    # blob_compactness = worm_blob_data['compactness'] if worm_blob_data is not None and 'compactness' in worm_blob_data else None
    # if blob_compactness is not None:
    #     # detect local peaks and sudden changes within 1-second windows
    #     window_size = int(fps * 1)  # 1 second window
    #     if window_size % 2 == 0:
    #         window_size += 1
        
    #     bc = pd.Series(blob_compactness).fillna(method='ffill').fillna(method='bfill')

    #     # 1) detect local peaks within each window
    #     peak_mask = np.zeros(len(bc), dtype=bool)
    #     for i in range(len(bc)):
    #         start = max(0, i - window_size // 2)
    #         end = min(len(bc), i + window_size // 2 + 1)
    #         window_vals = bc.iloc[start:end].values
    #         # if current frame is max in window, mark as peak
    #         if bc.iloc[i] == window_vals.max() and window_vals.max() > 0:
    #             peak_mask[i] = True
        
    #     # 2) detect sudden changes (>20% difference between consecutive frames)
    #     sudden_change_mask = np.zeros(len(bc), dtype=bool)
    #     for i in range(len(bc) - int(fps)):
    #         # compare frame i to frame i+1sec
    #         future_idx = min(i + int(fps), len(bc) - 1)
    #         if bc.iloc[i] > 0:
    #             pct_change = np.abs(bc.iloc[future_idx] - bc.iloc[i]) / bc.iloc[i]
    #             if pct_change >= 0.25:
    #                 sudden_change_mask[i] = True
    #     # remove the peaks lower than a minimum compactness threshold
    #     min_compactness = 0.35
    #     peak_mask = peak_mask & (bc.values >= min_compactness)

    #     # combine peaks and sudden changes
    #     compactness_mask = peak_mask & sudden_change_mask
        
    #     # create a rolling window to ensure at least one qualifying frame nearby
    #     rolling_window_size = int(fps * 1.5)  # 1.5 seconds
    #     if rolling_window_size % 2 == 0:
    #         rolling_window_size += 1
    #     compactness_mask_rolled = pd.Series(compactness_mask).rolling(
    #         window=rolling_window_size, center=True, min_periods=1).max().values.astype(bool)
        
    #     # Update potential_turn_vector to only keep turns where compactness condition is met
    #     turn_vector = potential_turn_vector & compactness_mask_rolled

    # Temporarily export blob compactness, angular velocity, turn_vector, net displacement, movement_mask for each worm-timestamp where there's a potential turn
    # tabale structure: timestamp (2*fps before and after the turn), worm_index-turn_timestamp, angular_velocity, blob_compactness, net_displacement, movement_mask, turn_vector
    # create per-worm turn table (may be empty)
    # turn_frames = np.flatnonzero(turn_vector)

    # # for adjuscent turn frames, keep only the central one
    # if turn_frames.size > 0:
    #     min_separation = int(fps * 1)  # 1 second separation
    #     clusters = []
    #     current_cluster = [turn_frames[0]]
    #     for prev, cur in zip(turn_frames[:-1], turn_frames[1:]):
    #         if cur - prev < min_separation:
    #             current_cluster.append(cur)
    #         else:
    #             clusters.append(current_cluster)
    #             current_cluster = [cur]
    #     clusters.append(current_cluster)

    #     # pick central (median) frame for each cluster
    #     filtered_turn_frames = [int(np.median(cluster)) for cluster in clusters]
    #     turn_frames = np.array(filtered_turn_frames, dtype=int)

    # df_turn_table = None
    # if turn_frames.size > 0:
    #     df_turn_table = export_turn_features_table(
    #         worm_data=worm_data,
    #         worm_blob_data=worm_blob_data,
    #         fps=fps,
    #         turn_frames=turn_frames,
    #         window_s=2,
    #         pad=True,
    #         require_complete=False,
    #         worm_index=None
    #     )
    # export to df_turn_table_{worm_index} to a csv file for debugging
    # find worm_index from worm_blob_data
    # worm_index = worm_blob_data['worm_index_joined'].iloc[0]
    # if df_turn_table is not None:
    #     # export all in the /Users/hkhabbaz/Documents/Research/projects/tierpsy/Development/features folder
    #     parent_folder = "/Users/hkhabbaz/Documents/Research/projects/tierpsy/Development/features/samples/SyngentaStrainScreening/turn_sample_ubuntu/Results/turn_data"
    #     debug_csv_file = os.path.join(parent_folder, f"df_turn_table_{worm_index}.csv")
    #     df_turn_table.to_csv(debug_csv_file, index=False)
        # print(f"Turn features table exported to {debug_csv_file} for worm_index {worm_index}")

    
    return turn_vector

#%%
def _range_vec(vec, th):
    '''
    flag a vector depending on the threshold, th
    -1 if the value is below -th
    1 if the value is above th
    0 if it is between -th and th
    '''
    flags = np.zeros(vec.size)
    _out = vec < -th
    _in = vec > th
    flags[_out] = -1
    flags[_in] = 1
    return flags

def _flag_regions(vec, central_th, extrema_th, smooth_window, min_frame_range):
    '''
    Flag a frames into lower (-1), central (0) and higher (1) regions.
    If the quantity used to flag the frame is NaN, and the frame i smore than
    smooth_window away from the last non-NaN frame, return a NaN

    The strategy is
        1) Smooth the timeseries by smoothed window
        2) Find frames that are certainly lower or higher using extrema_th
        3) Find regions that are between (-central_th, central_th) and
            and last more than min_frame_range. This regions are certainly
            central regions.
        4) If a region was not identified as central, but contains
            frames labeled with a given extrema, label the whole region
            with the corresponding extrema.
    '''
    # vv = pd.Series(vec).fillna(method='ffill').fillna(method='bfill')
    try:
        vv = pd.Series(vec).interpolate(method='nearest')
    except:
        # interpolate can fail if only one value is not nan.
        # just use ffill/bfill
        vv = pd.Series(vec).fillna(method='ffill').fillna(method='bfill')
    smoothed_vec = vv.rolling(window=smooth_window, center=True).mean()

    paused_f = (smoothed_vec > -central_th) & (smoothed_vec < central_th)
    turn_on, turn_off = _get_pulses_indexes(paused_f, min_frame_range)
    inter_pulses = zip([0] + list(turn_off), list(turn_on) + [paused_f.size-1])

    flag_modes = _range_vec(smoothed_vec, extrema_th)

    for ini, fin in inter_pulses:
        dd = np.unique(flag_modes[ini:fin+1])
        dd = [x for x in dd if x != 0]
        if len(dd) == 1:
            flag_modes[ini:fin+1] = dd[0]
        elif len(dd) > 1:
            kk = flag_modes[ini:fin+1]
            kk[kk==0] = np.nan
            kk = pd.Series(kk).fillna(method='ffill').fillna(method='bfill')
            flag_modes[ini:fin+1] = kk

    # the region is ill-defined if the frame was a NaN
    is_nan = pd.Series(vec).fillna(
        method='ffill', limit=smooth_window
        ).fillna(
            method='bfill', limit=smooth_window
            ).isna()
    flag_modes[is_nan] = np.nan

    return flag_modes

def classify_worm_states(smoothed_speeds: np.ndarray, thresholds: np.ndarray = np.array([5, 25, 100])) -> np.ndarray:
    """
    Classify the worm's behaviour into one of four states:
    - 0: Quiescence
    - 1: Dwelling
    - 2: Roaming
    - 3: Sprinting

    Parameters:
    - smoothed_speeds: np.ndarray of shape (n_frames, 3)
      Contains the smoothed speed data for head, midbody, and tail.
    - thresholds: np.ndarray of length 3
      Thresholds defining speed categories:
        - [quiescence_max, dwelling_max, roaming_max]
        - dwelling: 0 <= MIDBODY speed <= dwelling_max
        - roaming: dwelling_max < MIDBODY speed <= roaming_max
        - sprinting: MIDBODY speed > roaming_max
        - quiescence: ALL speeds <= quiescence_max

    Returns:
    - np.ndarray of shape (n_frames,), where each value is an integer
      representing the worm's state for that frame.
    """
    # ********************************
    # temporarily import parameters

    import yaml

    # load paramerters combination
    import os
    events_path = os.path.abspath(__file__)
    
    # find parameters_combo.yaml in the extras directory
    extras_dir = events_path.replace('/features/tierpsy_features/events.py', '/extras')
    param_combo_file_name = 'parameters_combo.yaml'
    PARAM_COMBO_FILE = os.path.join(extras_dir, param_combo_file_name)

    with open(PARAM_COMBO_FILE, 'r') as f:
        params = yaml.safe_load(f)

    dwelling_threshold = params['dwelling_threshold']
    roaming_threshold = params['roaming_threshold']
    sprinting_threshold = params['sprinting_threshold']
    thresholds = np.array([dwelling_threshold, roaming_threshold, sprinting_threshold])

    # end of temporary import
    # ********************************


    
    # Validate input thresholds
    if len(thresholds) != 3:
        raise ValueError("Thresholds array must contain exactly 3 elements.")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("Threshold values must be in ascending order.")

    n_frames = smoothed_speeds.shape[0]
    worm_states = np.full(n_frames, np.nan)  # Initialize with NaNs

    speed_head_base, speed_midbody, speed_tail_base = smoothed_speeds.T
    abs_midbody_speed = np.abs(speed_midbody)

    # Unpack thresholds for readability
    quiescence_max, dwelling_max, roaming_max = thresholds

    # Classify states based on speed thresholds
    # 1: Dwelling (0 to dwelling_max)
    dwelling_mask = (abs_midbody_speed >= 0) & (abs_midbody_speed <= dwelling_max)
    worm_states[dwelling_mask] = 1

    # 2: Cruising (dwelling_max to roaming_max)
    roaming_mask = (abs_midbody_speed > dwelling_max) & (abs_midbody_speed <= roaming_max)
    worm_states[roaming_mask] = 2

    # 3: Sprinting (> roaming_max)
    sprinting_mask = abs_midbody_speed > roaming_max
    worm_states[sprinting_mask] = 3

    # 0: Quiescence (all speeds <= quiescence_max)
    quiescence_mask = (
        (np.abs(speed_head_base) <= quiescence_max) &
        (np.abs(speed_midbody) <= quiescence_max) &
        (np.abs(speed_tail_base) <= quiescence_max)
    )
    worm_states[quiescence_mask] = 0

    return worm_states

def _get_vec_durations(event_vec):
    durations_list = []
    for e_id in np.unique(event_vec):
        if e_id != e_id:
            # skip nans
            continue
        ini_e, fin_e = _get_pulses_indexes(event_vec == e_id, is_pad = True)
        event_durations = fin_e - ini_e

        #flag if the event is on the vector edge or not
        edge_flag = np.zeros_like(fin_e)
        edge_flag[ini_e <= 0] = -1
        edge_flag[fin_e >= event_vec.size-1] = 1

        event_ids = np.full(event_durations.shape, e_id)
        durations_list.append(np.stack((event_ids, event_durations, ini_e, fin_e, edge_flag)).T)

    cols = ['region', 'duration', 'timestamp_initial', 'timestamp_final', 'edge_flag']
    if len(durations_list) == 0:
        event_durations_df = pd.DataFrame(columns = cols)
    else:
        event_durations_df = pd.DataFrame(np.concatenate(durations_list), columns = cols)

    return event_durations_df

def get_event_durations_w(events_df, fps):
    event_durations_list = []
    for col in events_df:
        if not col in ['timestamp', 'worm_index']:
            dd = _get_vec_durations(events_df[col].values)
            dd.insert(0, 'event_type', col)
            event_durations_list.append(dd)

    if len(event_durations_list) == 0:
        event_durations_df = pd.DataFrame()
    else:

        event_durations_df = pd.concat(event_durations_list, ignore_index=True)
        event_durations_df['duration'] /= fps
        #shift timestamps to match the real initial time
        first_t = events_df['timestamp'].min()
        event_durations_df['timestamp_initial'] += first_t
        event_durations_df['timestamp_final'] += first_t


    return event_durations_df


def get_events(df, fps, worm_length = None, _is_debug=False, worm_blob_data = None, skeletons = None):

    #initialize data
    smooth_window_s = 0.5
    min_paused_win_speed_s = 1/3

    if worm_length is None:
        assert 'length' in df
        worm_length = df['length'].median()


    df = df.sort_values(by='timestamp')

    w_size = int(round(fps*smooth_window_s))
    smooth_window = w_size if w_size % 2 == 1 else w_size + 1

    #WORM MOTION EVENTS
    dd = [x for x in ['worm_index', 'timestamp'] if x in df]
    events_df = pd.DataFrame(df[dd])
    if 'speed' in df:
        speed = df['speed'].values
        pause_th_lower = worm_length*0.025
        pause_th_higher = worm_length*0.05
        min_paused_win_speed = fps * min_paused_win_speed_s

        motion_mode = _flag_regions(speed,
                                 pause_th_lower,
                                 pause_th_higher,
                                 smooth_window,
                                 min_paused_win_speed
                                 )
        events_df['motion_mode'] = motion_mode

    #FOOD EDGE EVENTS
    if 'dist_from_food_edge' in df:
        dist_from_food_edge = df['dist_from_food_edge'].values
        edge_offset_lower = worm_length/2
        edge_offset_higher = worm_length
        min_paused_win_food_s = 1

        min_paused_win_food = fps * min_paused_win_food_s
        food_region = _flag_regions(dist_from_food_edge,
                                     edge_offset_lower,
                                     edge_offset_higher,
                                     smooth_window,
                                     min_paused_win_food
                                     )
        events_df['food_region'] = food_region

    #TURN EVENT
    if set(('head_tail_distance', 'major_axis', 'angular_velocity')).issubset(set(df.columns)):
        turn_vector = _find_turns(df, fps, worm_blob_data=worm_blob_data, skeletons=skeletons)
        events_df['turn'] = turn_vector.astype(np.float32)

    # BEHAVIOURAL STATES
    speed_names = ['speed_head_base', 'speed_midbody', 'speed_tail_base']
    if all(name in df for name in speed_names):
        speeds = np.column_stack([df[name] for name in speed_names])
        smooth_speeds = np.apply_along_axis(nanmedian_filter, 0, speeds, 31)
        worm_states = classify_worm_states(smooth_speeds)
        events_df['behavioural_states'] = worm_states
    else:
        # Optionally, log a warning or fill with NaN
        events_df['behavioural_states'] = np.full(df.shape[0], np.nan)

    if _is_debug:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(speed)
        plt.plot(motion_mode*pause_th_higher)

        plt.figure()
        plt.plot(dist_from_food_edge)
        plt.plot(food_region*edge_offset_lower)

    return events_df

#%%
def _get_event_stats(event_durations, n_worms_estimate, total_time):
    '''
    Get the event statistics using the event durations table.
    '''
    if event_durations.size == 0:
        return pd.Series()

    all_events_time = event_durations.groupby('event_type').agg({'duration':'sum'})['duration']
    event_g = event_durations.groupby(['event_type', 'region'])
    event_stats = []

    valid_regions = [x for x in event_region_labels.keys() if x in all_events_time]

    for event_type in valid_regions:
        region_dict = event_region_labels[event_type]
        for region_id, region_name in region_dict.items():
            stat_prefix = event_type + '_' + region_name
            try:
                dat = event_g.get_group((event_type, region_id))
                duration = dat['duration'].values
                edge_flag = dat['edge_flag'].values
            except:
                duration = np.zeros(1)
                edge_flag = np.zeros(0)

            stat_name = stat_prefix + '_duration_50th'
            stat_val = np.nanmedian(duration)
            event_stats.append((stat_val, stat_name))

            stat_name = stat_prefix + '_fraction'
            stat_val = np.nansum(duration)/all_events_time[event_type]
            event_stats.append((stat_val, stat_name))

            stat_name = stat_prefix + '_frequency'
            # calculate total events excluding events that started before the beginig of the trajectory
            total_events = (edge_flag != -1).sum()
            stat_val = total_events/n_worms_estimate/total_time
            event_stats.append((stat_val, stat_name))

    event_stats_s = pd.Series(*list(zip(*event_stats)))
    return event_stats_s

#%%
def get_event_durations(timeseries_data, fps):

    dd = ['worm_index', 'timestamp'] + event_columns
    dd = [x for x in dd if x in timeseries_data]
    events_df = timeseries_data[dd]

    event_durations = []
    for worm_index, dat in events_df.groupby('worm_index'):
        dur = get_event_durations_w(dat, fps)
        dur['worm_index'] = worm_index
        event_durations.append(dur)

    if event_durations:
        event_durations = pd.concat(event_durations, ignore_index=True)
        return event_durations
    else:
        return pd.DataFrame()


def get_event_stats(timeseries_data, fps, n_worms_estimate):
    event_durations = get_event_durations(timeseries_data, fps)

    total_time = (timeseries_data['timestamp'].max() - timeseries_data['timestamp'].min())/fps
    event_stats_s = _get_event_stats(event_durations, n_worms_estimate, total_time)
    return event_stats_s
#%%

if __name__ == '__main__':
    from tierpsy.helper.params import read_fps
    import matplotlib.pylab as plt
    import os
    import glob


    dname = '/Volumes/behavgenom_archive$/Solveig/Results/'
    fnames = glob.glob(os.path.join(dname, 'Experiment8', '**', '*_featuresN.hdf5'), recursive = True)

    for ifname, fname in enumerate(fnames):
        print(ifname+1, len(fnames))
        with pd.HDFStore(fname, 'r') as fid:
            if '/provenance_tracking/FEAT_TIERPSY' in fid:
                timeseries_data = fid['/timeseries_features']

                trajectories_data = fid['/trajectories_data']
                good = trajectories_data['skeleton_id']>=0
                trajectories_data = trajectories_data[good]
            else:
                continue
        break

    #%%
    fps = read_fps(fname)
    for worm_index in [2]:#, 69, 431, 437, 608]:
        worm_data = timeseries_data[timeseries_data['worm_index']==worm_index]
        worm_length = worm_data['length'].median()

        events_df = get_events(worm_data, fps, _is_debug=True)
        #get event durations
        event_durations_df = get_event_durations_w(events_df, fps)


    #%%
    from tierpsy_features.helper import get_n_worms_estimate

    n_worms_estimate = get_n_worms_estimate(timeseries_data['timestamp'])
    get_event_stats(events_df, fps, n_worms_estimate)
    #%%
    from tierpsy.analysis.ske_create.helperIterROI import  getROIfromInd
    turns_vec, d_ratio, ang_velocity = _find_turns(worm_data, fps)
    xx = worm_data['timestamp'].values

    plt.figure(figsize=(25,5))
    plt.plot(xx, ang_velocity)
    plt.plot(xx, d_ratio)
    plt.plot(xx, turns_vec)
    #plt.ylim((0.7, 1.1))
    plt.xlim((xx[0], xx[-1]))


    dd = _get_pulses_indexes(turns_vec, min_window_size=fps//2)
    pulse_ranges = list(zip(*dd))


    masked_file = fname.replace('_featuresN', '')
    for p in pulse_ranges:

        dd = worm_data.loc[worm_data.index[p[0]:p[1]+1]]
        timestamps = dd['timestamp'][::12]
        plt.figure(figsize=(50, 5))
        for ii, tt in enumerate(timestamps):
            _, img, _ = getROIfromInd(masked_file, trajectories_data, tt, worm_index)

            plt.subplot(1, timestamps.size, ii+1)
            plt.imshow(img, cmap='gray', interpolation='none')
            plt.axis('off')
