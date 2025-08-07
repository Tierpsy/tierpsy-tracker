# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib.pylab as plt
from matplotlib import animation, patches

from .helper import DataPartition, nanunwrap

# i am including an excess of subdivisions with the hope to later reduce them
velocities_columns = ['speed', 
                      'angular_velocity', 
                      'relative_to_body_speed_midbody',
                      'relative_to_body_radial_velocity_head_tip',
                       'relative_to_body_angular_velocity_head_tip',
                       'relative_to_body_radial_velocity_neck',
                       'relative_to_body_angular_velocity_neck',
                       'relative_to_body_radial_velocity_hips',
                       'relative_to_body_angular_velocity_hips',
                       'relative_to_body_radial_velocity_tail_tip',
                       'relative_to_body_angular_velocity_tail_tip', 
                       'speed_neck',
                       'angular_velocity_neck', 
                       'relative_to_neck_radial_velocity_head_tip',
                       'relative_to_neck_angular_velocity_head_tip', 
                       'speed_head_base',
                       'angular_velocity_head_base',
                       'relative_to_head_base_radial_velocity_head_tip',
                       'relative_to_head_base_angular_velocity_head_tip', 
                       'speed_hips',
                       'angular_velocity_hips', 
                       'relative_to_hips_radial_velocity_tail_tip',
                       'relative_to_hips_angular_velocity_tail_tip', 
                       'speed_tail_base',
                       'angular_velocity_tail_base',
                       'relative_to_tail_base_radial_velocity_tail_tip',
                       'relative_to_tail_base_angular_velocity_tail_tip', 
                       'speed_midbody',
                       'angular_velocity_midbody', 
                       'speed_head_tip',
                       'angular_velocity_head_tip', 
                       'speed_tail_tip',
                       'angular_velocity_tail_tip',
                       'angular_velocity_head',
                       'angular_velocity_tail',
                       'speed_head',
                       'speed_tail',
                       ]

#%% features that are relative to specific body parts
relative_to_dict = {'body' : ('head_tip', 'neck', 'hips', 'tail_tip'), 
               'neck' : ('head_tip',),
               'head_base' : ('head_tip',),
               'hips' : ('tail_tip',),
               'tail_base' : ('tail_tip',),
               'midbody' : [],
               'head_tip' : [],
               'tail_tip' : [],
               'head' : [],
               'tail' : []
               }

#%%
def _h_orientation_vector(x, axis=None):
    return x[:, 0, :] - x[:, -1, :]

def _h_get_velocity(x, delta_frames, fps):
    if delta_frames < 1:
        raise ValueError('Invalid number of delta frames %i' % delta_frames)
    delta_time = delta_frames/fps
    if x.shape[0] < delta_frames:
        #not enough frames return empty array
        return np.full_like(x, np.nan)

    v = (x[delta_frames:] - x[:-delta_frames])/delta_time

    #pad with nan so the vector match the original vectors
    pad_w = [(int(np.floor(delta_frames/2)), int(np.ceil(delta_frames/2)))]
    
    #explicity add zero path if there are extra dimensions
    if x.ndim > 1:
        pad_w += [(0,0) for _ in range(x.ndim-1)]
    
    v = np.pad(v, 
              pad_w, 
              'constant', 
              constant_values = np.nan)
    
    return v

#%%
def _h_center_skeleton(skeletons, orientation, coords):
    
    Rsin = np.sin(orientation)[:, None]
    Rcos = np.cos(orientation)[:, None]

    skel_c = skeletons - coords[:, None, :]

    skel_ang = np.zeros_like(skel_c)
    skel_ang[:, :, 0] = skel_c[:, :, 0]*Rcos - skel_c[:, :, 1]*Rsin
    skel_ang[:, :, 1] = skel_c[:, :, 0]*Rsin + skel_c[:, :, 1]*Rcos
    
    return skel_ang

def _h_segment_position(skeletons, partition):
    p_obj = DataPartition([partition], n_segments=skeletons.shape[1])
    coords = p_obj.apply(skeletons, partition, func=np.nanmean)
    orientation_v = p_obj.apply(skeletons, partition, func=_h_orientation_vector)
    return coords, orientation_v

#%%
def get_velocity(skeletons, partition, delta_frames, fps):
    coords, orientation_v = _h_segment_position(skeletons, partition = partition)
    
    nan_frames = np.isnan(coords[:, 0])
    
    is_any_nan = np.any(nan_frames)
    
    if is_any_nan:
        x = np.arange(coords.shape[0])
        xp = np.where(~nan_frames)[0]
        if xp.size > 2:
            #I only do this if there are actually some points to interpolate
            for ii in range(coords.shape[1]):
                coords[:, ii] = np.interp(x, xp, coords[xp, ii])
                orientation_v[:, ii] = np.interp(x, xp, orientation_v[xp, ii])
    
    velocity = _h_get_velocity(coords, delta_frames, fps)
    speed = np.linalg.norm(velocity, axis=1)
    #I do not need to normalize the vectors because it will only add a constant factor, 
    #and I am only interested in the sign
    s_sign = np.sign(np.sum(velocity*orientation_v, axis=1))
    signed_speed = speed *s_sign
    
    #let's change the vector to angles
    orientation = np.arctan2(orientation_v[:, 0], orientation_v[:, 1])
    #wrap the angles so the change is continous no jump between np.pi and -np.pi
    orientation = nanunwrap(orientation) 
    angular_velocity = _h_get_velocity(orientation, delta_frames, fps)
    
    centered_skeleton = _h_center_skeleton(skeletons, orientation, coords)

    if is_any_nan:
        signed_speed[nan_frames] = np.nan
        angular_velocity[nan_frames] = np.nan
    
    return signed_speed, angular_velocity, centered_skeleton

#%%
def _h_relative_velocity(segment_coords, delta_frames, fps):
    x = segment_coords[:, 0]
    y = segment_coords[:, 1]
    r = np.sqrt(x**2+y**2)
    theta = nanunwrap(np.arctan2(y,x))
    
    r_radial_velocity = _h_get_velocity(r, delta_frames, fps)
    r_angular_velocity = _h_get_velocity(theta, delta_frames, fps)
    
    return r_radial_velocity, r_angular_velocity



#%%
def get_relative_velocities(centered_skeleton, partitions, delta_frames, fps):
    p_obj = DataPartition(partitions, n_segments=centered_skeleton.shape[1])

    r_radial_velocities = {}
    r_angular_velocities = {}
    
    
    for p in partitions:
        
    
        segment_coords = p_obj.apply(centered_skeleton, p, func=np.nanmean)
        r_radial_velocity, r_angular_velocity = _h_relative_velocity(segment_coords, delta_frames, fps)
        r_radial_velocities[p] = r_radial_velocity
        r_angular_velocities[p] = r_angular_velocity
        
        
    
    return r_radial_velocities, r_angular_velocities


def get_relative_speed_midbody(centered_skeleton, delta_frames, fps):
    '''
    This velocity meassures how the midbody changes in relation to the body central axis.
    I cannot really define this for the othes parts without getting too complicated.
    '''
    p_obj = DataPartition(['midbody'], n_segments=centered_skeleton.shape[1])
    segment_coords = p_obj.apply(centered_skeleton, 'midbody', func=np.nanmean)
    return _h_get_velocity(segment_coords[:, 0], delta_frames, fps)

#%%


def _h_ax_range(skel_a):
    x_range = [np.nanmin(skel_a[...,0]), np.nanmax(skel_a[...,0])]
    y_range = [np.nanmin(skel_a[...,1]), np.nanmax(skel_a[...,1])]
    
    dx, dy = np.diff(x_range), np.diff(y_range)
    if dx > dy:
        y_range[1] = y_range[0] + dx
    else:
        x_range[1] = x_range[0] + dy
    
    return (x_range, y_range)

def animate_velocity(skel_a, ini_arrow, arrow_size, speed_v, ang_v):
    x_range, y_range = _h_ax_range(skel_a)
    fig = plt.figure(figsize = (15, 8))
    ax = plt.subplot(1,2,1)
    ax_speed = plt.subplot(2,2,2)
    ax_ang_speed = plt.subplot(2,2,4)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    
    line, = ax.plot([], [], lw=2)
    head_p, = ax.plot([], [], 'o')
    orient_arrow = patches.Arrow(*ini_arrow[0], *arrow_size[0], fc='k', ec='k')
    
    ax_speed.plot(speed_v)
    ax_ang_speed.plot(ang_v)
    
    speed_p, = ax_speed.plot([], 'o') 
    ang_speed_p, = ax_ang_speed.plot([],  'o') 
    
    # animation function. This is called sequentially
    def _animate(i):
        global orient_arrow
        
        x = skel_a[i, :, 0]
        y = skel_a[i, :, 1]
        line.set_data(x, y)
        head_p.set_data(x[0], y[0])
        if ax.patches:
            ax.patches.remove(orient_arrow) 
        orient_arrow = patches.Arrow(*ini_arrow[i], *arrow_size[i], width=50, fc='k', ec='k')
        ax.add_patch(orient_arrow)
        
        speed_p.set_data(i, speed_v[i])
        ang_speed_p.set_data(i, ang_v[i])
        return (line, head_p, orient_arrow, speed_p, ang_speed_p)
    
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, _animate,
                                   frames=skel_a.shape[0], interval=20, blit=True);
    return anim

#%%
def get_velocity_features(skeletons, delta_frames, fps):
    assert isinstance(delta_frames, int)
    
    if skeletons.shape[0] < delta_frames:
        return
    
    
    def _process_part(part):
        signed_speed, angular_velocity, centered_skeleton = get_velocity(skeletons, part, delta_frames, fps)
        
        if part == 'body':
            #speed without prefix is the body speed
            part_velocities = [('speed', signed_speed),
                               ('angular_velocity', angular_velocity)]
            #this is really a special case. midbody moving like this <- | ->
            relative_speed_midbody = get_relative_speed_midbody(centered_skeleton, delta_frames, fps)
            
            #add the body signed speed and angular velocity. This values are very similar for the other parts
            part_velocities.append(('relative_to_body_speed_midbody', relative_speed_midbody))
        else:
            #at the end i might only calculate this for the body, but i want to do a massive test to feel sure about this
            part_velocities = [('speed_' + part, signed_speed),
                               ('angular_velocity_' + part, angular_velocity)]
            
        if part in relative_to_dict:
            partitions = relative_to_dict[part]
            r_radial_velocities, r_angular_velocities = get_relative_velocities(centered_skeleton, 
                                    partitions, 
                                    delta_frames, 
                                    fps)
            #pack into a dictionary
            for p in partitions:
                k_r = 'relative_to_{}_radial_velocity_{}'.format(part, p)
                part_velocities.append((k_r, r_radial_velocities[p]))
                
                k_a = 'relative_to_{}_angular_velocity_{}'.format(part, p)
                part_velocities.append((k_a, r_angular_velocities[p]))
        
        
        
        
        return part_velocities
        
    #process all the parts
    velocities = map(_process_part, relative_to_dict.keys())
    #flatten list
    velocities = sum(velocities, []) 
    
    #put all the data into a dataframe
    velocities = pd.DataFrame(OrderedDict(velocities))
    
    
    
    assert velocities.shape[0] == skeletons.shape[0]
    
    return velocities


#%%
if __name__ == '__main__':
    #data = np.load('worm_example_small_W1.npz')
    data = np.load('../notebooks/data/worm_example.npz')
    skeletons = data['skeleton']
    
    fps = 25
    delta_time = 1/3 #delta time in seconds to calculate the velocity
    
    delta_frames = max(1, int(round(fps*delta_time)))
    velocities = get_velocity_features(skeletons, delta_frames, fps)
    
