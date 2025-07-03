#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ajaver
"""

import numpy as np
import pandas as pd

event_columns = ['motion_mode','behavioural_states', 'food_region', 'turn']
durations_columns = ['event_type', 'region',
                     'duration', 'timestamp_initial',
                     'timestamp_final', 'edge_flag']
event_region_labels = {
            'motion_mode': {-1:'backward', 1:'forward', 0:'paused',},
            'behavioural_states': {0: 'quiescence', 1: 'dwelling', 2: 'roaming', 3: 'sprinting'},
            'food_region': {-1:'outside', 1:'inside', 0:'edge'},
            'turn': {1:'inter', 0:'intra'}
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
def _find_turns(worm_data,
                fps,
                d_ratio_th = (0.15, 0.075),
                ang_v_th = (0.75, 0.35),
                interp_window_s = 0.5
                ):
    #check the necessary columns are in the dataframe
    assert set(('head_tail_distance', 'major_axis', 'angular_velocity')).issubset(set(worm_data.columns))

    #adjust the interpolation window in frames
    w_interp = int(fps*interp_window_s)
    w_interp = w_interp if w_interp%2 == 1 else w_interp+1

    try:
        #get the ratio of this mesurements
        #the cubic interpolation is important to detect this feature
        d_ratio = 1-(worm_data['head_tail_distance']/worm_data['major_axis'])
        d_ratio = d_ratio.rolling(window = w_interp).min().interpolate(method='cubic')
        with np.errstate(invalid='ignore'):
            ang_velocity = worm_data['angular_velocity'].abs()
        ang_velocity = ang_velocity.rolling(window = w_interp).max().interpolate(method='cubic')
    except ValueError:
        #there was an error in the interpolation
        return [np.full(worm_data.shape[0], np.nan) for _ in range(3)]


    #find candidate turns that satisfy at the same time the higher threshold
    turns_vec_ini = (d_ratio>d_ratio_th[0]) & (ang_velocity>ang_v_th[0])

    #refine the estimates with the lower threshold in each vector independently
    d_ration_candidates = _get_pulses_indexes(d_ratio>d_ratio_th[1])
    d_ration_r = [x for x in zip(*d_ration_candidates) if np.any(turns_vec_ini[x[0]:x[1]+1])]

    ang_v_candidates = _get_pulses_indexes(ang_velocity>ang_v_th[1])
    ang_v_r = [x for x in zip(*ang_v_candidates) if np.any(turns_vec_ini[x[0]:x[1]+1])]

    #combine the results into a final vector
    turns_vec = np.zeros_like(turns_vec_ini)
    for x in d_ration_r + ang_v_r:
        turns_vec[x[0]:x[1]+1] = True

    return turns_vec, d_ratio, ang_velocity

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


def get_events(df, fps, worm_length = None, _is_debug=False):

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
        turn_vector, _, _ = _find_turns(df, fps)
        events_df['turn'] = turn_vector.astype(np.float32)


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
