#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:51:07 2017

@author: ajaver

Get the food contour using thresholdings and morphological operations.

"""
import tables
import numpy as np
import cv2

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

from tierpsy.helper.misc import get_base_name



def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def get_patch_mask(img, min_area = None, max_area = None, block_size = None):
    '''
    Modified version of getROIMask optimized to highlight the food patch
    '''
    #%%
    if min_area is None:
        min_area = max(1, int(min(img.shape)/200))**2

    if max_area is None:
        max_area = (img.shape[0]*img.shape[1])/4

    if block_size is None:
        block_size = int(min(img.shape)/8)
        block_size = block_size+1 if block_size%2==0 else block_size
    #%%
    mask_s = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=3)

    mask = cv2.morphologyEx(mask_s, cv2.MORPH_CLOSE, disk(1), iterations=1)
    mask = cv2.erode(mask, disk(1), iterations=1)
    #kernel = np.array([(-1,-1,-1), (-1, 1, -1), (-1, -1,-1)])
    #ss = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, kernel)
    #mask = mask-ss
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(3), iterations=3)
    # plt.imshow(mask)
    #%%
    #IM_LIMX = img.shape[0] - 2
    #IM_LIMY = img.shape[1] - 2
    # find the contour of the connected objects (much faster than labeled
    # images)

    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]


    # typically there are more bad contours therefore it is cheaper to draw
    # only the valid contours
    mask = np.zeros(img.shape, dtype=img.dtype)
    for ii, contour in enumerate(contours):
        # eliminate blobs that touch a border
        #keep = not np.any(contour == 1) and \
        #    not np.any(contour[:, :, 0] ==  IM_LIMY)\
        #    and not np.any(contour[:, :, 1] == IM_LIMX)
        #if keep:
        area = cv2.contourArea(contour)
        if (area >= min_area):
            '''
            If there is a blob with a very high area it is likely
            the mask captured the whole food patch. Do not fill the contour of those areas,
            otherwise the skeletonization produce weird results.
            '''
            if area >= max_area:
                cv2.drawContours(mask, contours, ii, 1)
            else:
                cv2.drawContours(mask, contours, ii, 1, cv2.FILLED)


    #%%
    return mask

def get_best_circles(mask, resize_factor = 8):
    '''
    Get the best the best fit to a circle using the hough transform.
    '''
    #%%
    #resize image to increase speed. I don't want
    #%%
    min_size = min(mask.shape)
    resize_factor = min_size/max(128, min_size/resize_factor)
    dsize = tuple(int(x/resize_factor) for x in mask.shape[::-1])

    mask_s = cv2.dilate(mask, disk(resize_factor/2))
    mask_s = cv2.resize(mask_s,dsize)
    #%%
    r_min = min(mask_s.shape)
    r_max = max(mask_s.shape)

    #use the first peak of the circle hough transform to initialize the food shape
    hough_radii = np.arange(r_min/4, r_max/2, 2)
    hough_res = hough_circle(mask_s, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                      total_num_peaks=9)
    #%%

    cx, cy, radii = [np.round(x*resize_factor).astype(np.int) for x in (cx, cy, radii)]
    #%%

    return list(zip(accums, cx, cy, radii))

def mask_to_food_contour(mask, n_bins = 90, frac_lowess=0.05, _is_debug=False):
    '''
    Estimate the food contour from a binary mask.
    1) Get the best the best fit to a circle using the hough transform.
    2) Transform the mask into polar coordinates centered in the fitted circle.
    3) Get the closest point to the circle using 2*n_bins angles.
    4) Smooth using lowess fitting (good to ignore outliers) to estimate the food contour.
    5) Transform back into cartesian coordinates.
    '''
    #%%
    h_res = get_best_circles(mask.copy())
    _, cy0, cx0, r0 =  h_res[0]


    (px, py) = np.where(skeletonize(mask))



    xx = px-cx0
    yy = py-cy0

    del_r = np.sqrt(xx*xx + yy*yy) - r0
    theta = np.arctan2(xx,yy)

    theta_d = np.round((theta/np.pi)*n_bins).astype(np.int)

    #g = {k:[] for k in range(-n_bins, n_bins+1)}
    g = {k:[] for k in np.unique(theta_d)}
    for k,dr in zip(theta_d, del_r):
        g[k].append(dr)

    _get_min = lambda g : min(g, key = lambda x:abs(x)) if g else np.nan
    tg, min_dr = zip(*[(k, _get_min(g)) for k,g in g.items()])

    theta_s = np.array(tg)*np.pi/n_bins

    #increase the range for the interpolation
    theta_i = np.hstack((theta_s-2*np.pi, theta_s, theta_s+2*np.pi))

    r_s = np.hstack((min_dr, min_dr, min_dr))

    out = lowess(r_s, theta_i, frac=frac_lowess)

    #win_frac = 1/3
    #filt_w = round(n_bins*win_frac)
    #filt_w = filt_w+1 if filt_w % 2 == 0 else filt_w
    #r_s = medfilt(r_s, filt_w)
    f = interp1d(out[:, 0], out[:, 1])

    theta_new = np.linspace(-np.pi, np.pi, 480)
    r_new = f(theta_new) + r0

    circy = r_new*np.cos(theta_new) + cy0
    circx = r_new*np.sin(theta_new) + cx0
    #%%
    if _is_debug:
        from skimage.draw import circle_perimeter
        import matplotlib.pylab as plt

        plt.figure(figsize=(5,5))
        for ii, (acc, cx, cy, cr) in enumerate(h_res[0:1]):
            #plt.subplot(3,3,ii+1)
            plt.imshow(mask)
            cpy, cpx = circle_perimeter(cy, cx, cr)
            plt.plot(cpx,cpy, '.r')


        plt.figure()
        plt.plot(theta_d, del_r, '.')
        plt.plot(tg, min_dr)


        plt.figure()
        plt.plot(theta_i, r_s,'.')
        plt.plot(out[:, 0], out[:, 1], '.')

    #%%
    return circx, circy, h_res[0]


def get_dark_mask(full_data):
    #get darker objects that are unlikely to be worm
    if full_data.shape[0] < 2:
        #nothing to do here returning
        return np.zeros((full_data.shape[1], full_data.shape[2]), np.uint8)

    #this mask shoulnd't contain many worms
    img_h = cv2.medianBlur(np.max(full_data, axis=0), 5)
    #this mask is likely to contain a lot of worms
    img_l = cv2.medianBlur(np.min(full_data, axis=0), 5)

    #this is the difference (the tagged pixels should be mostly worms)
    img_del = img_h-img_l
    th_d = threshold_otsu(img_del)

    #this is the maximum of the minimum pixels of the worms...
    th = np.max(img_l[img_del>th_d])
    #this is what a darkish mask should look like
    dark_mask = cv2.dilate((img_h<th).astype(np.uint8), disk(11))

    return dark_mask

def get_food_contour_morph(mask_video,
                     min_area = None,
                     n_bins = 180,
                     frac_lowess=0.1,
                     _is_debug=False):
    '''
    Identify the contour of a food patch. I tested this for the worm rig.
    It assumes the food has a semi-circular shape.
    The food lawn is very thin so the challenge was to estimate the contour of a very dim area.
    '''
    #%%

    try:
        with tables.File(mask_video, 'r') as fid:
            full_data = fid.get_node('/full_data')[:5] # I am using the first two images to calculate this info
    except tables.exceptions.NoSuchNodeError:
        return None, None

    img = np.max(full_data[:2], axis=0)
    #dark_mask = get_dark_mask(full_data)

    mask = get_patch_mask(img, min_area = min_area)
    circx, circy, best_fit = mask_to_food_contour(mask,
                                        n_bins = n_bins,
                                        frac_lowess=frac_lowess)


    if _is_debug:
        from skimage.draw import circle_perimeter
        import matplotlib.pylab as plt

        cpx, cpy = circle_perimeter(*best_fit[1:])

        plt.figure(figsize=(5,5))
        plt.gca().xaxis.set_ticklabels([])
        plt.gca().yaxis.set_ticklabels([])

        (px, py) = np.where(skeletonize(mask))
        plt.imshow(img, cmap='gray')
        plt.plot(py, px, '.')
        plt.plot(cpx, cpy, '.r')
        plt.plot(circy, circx, '.')
        plt.grid('off')

    food_cnt = np.vstack((circy, circx)).T
    return food_cnt

if __name__ == '__main__':
    import glob
    import os
    import fnmatch

    n_bins = 180
    frac_lowess=0.1

    exts = ['']

    exts = ['*'+ext+'.hdf5' for ext in exts]

    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_310517/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_160517/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_020617/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_Food/MaskedVideos/FoodDilution_041116'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/Development_C1_170617/'
    #mask_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Development/MaskedVideos/**/'
    #mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/ATR_210417'
    mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/**/'

    fnames = glob.glob(os.path.join(mask_dir, '*.hdf5'))
    fnames = [x for x in fnames if any(fnmatch.fnmatch(x, ext) for ext in exts)]

    for mask_video in fnames:
        food_cnt = get_food_contour_morph(mask_video,
                                        n_bins = n_bins,
                                        frac_lowess=frac_lowess,
                                        _is_debug=True)
        circx, circy = food_cnt.T

        base_name = get_base_name(mask_video)
        with tables.File(mask_video, 'r') as fid:
            full_data = fid.get_node('/full_data')[:]

        full_min = np.max(full_data, axis=0)
        full_max = np.min(full_data, axis=0)



        import matplotlib.pylab as plt
        for nn in range(full_data.shape[0]):
            plt.figure()
            plt.imshow(full_data[nn], cmap='gray')
            plt.plot(circx, circy)
            break
