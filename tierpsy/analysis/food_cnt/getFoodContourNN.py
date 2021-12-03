#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver

Get food contour using a pre-trained neural network

"""
# %%

import tables
import os
import numpy as np
import cv2
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from keras.models import load_model

from skimage.morphology import disk

DFLT_RESIZING_SIZE = 512  # the network was trained with images of this size 512



def _get_sizes(im_size, d4a_size= 24, n_conv_layers=4):
    ''' Useful to determine the expected inputs and output sizes of a u-net.
    Additionally if the image is larger than the network output the points to
    subdivide the image in tiles are given
    '''

    #assuming 4 layers of convolutions
    def _in_size(d4a_size):
        mm = d4a_size
        for n in range(n_conv_layers):
            mm = mm*2 + 2 + 2
        return mm

    def _out_size(d4a_size):
        mm = d4a_size -2 -2
        for n in range(n_conv_layers):
            mm = mm*2 - 2 - 2
        return mm


    #this is the size of the central reduced layer. I choose this value manually
    input_size = _in_size(d4a_size) #required 444 of input
    output_size = _out_size(d4a_size) #set 260 of outpu
    pad_size = int((input_size-output_size)/2)

    if any(x < output_size for x in im_size):
        msg = 'All the sides of the image ({}) must be larger or equal to ' \
                'the network output {}.'
        raise ValueError(msg.format(im_size, output_size))

    n_tiles_x = int(np.ceil(im_size[0]/output_size))
    n_tiles_y = int(np.ceil(im_size[1]/output_size))


    txs = np.round(np.linspace(0, im_size[0] - output_size, n_tiles_x)).astype(np.int)
    tys = np.round(np.linspace(0, im_size[1] - output_size, n_tiles_y)).astype(np.int)


    tile_corners = [(tx, ty) for tx in txs for ty in tys]

    return input_size, output_size, pad_size, tile_corners

def _preprocess(X,
                 input_size,
                 pad_size,
                 tile_corners
                 ):
    '''
    Pre-process an image to input for the pre-trained u-net model
    '''
    def _get_tile_in(img, x,y):
            return img[np.newaxis, x:x+input_size, y:y+input_size, :]

    def _cast_tf(D):
        D = D.astype(np.float32())
        if D.ndim == 2:
            D = D[..., None]
        return D


    #normalize image
    X = _cast_tf(X)
    X /= 255
    X -= np.median(X)

    pad_size_s =  ((pad_size,pad_size), (pad_size,pad_size), (0,0))
    X = np.lib.pad(X, pad_size_s, 'reflect')

    X = [_get_tile_in(X, x, y) for x,y in tile_corners]

    return X

def get_unet_prediction(Xi,
                  model_t,
                  n_flips = 1,
                  im_size=None,
                  n_conv_layers = 4,
                  d4a_size = 24,
                  _is_debug=False):

    '''
    Predict the food probability for each pixel using a pretrained u-net model (Helper)
    '''

    def _flip_d(img_o, nn):
        if nn == 0:
            img = img_o[::-1, :]
        elif nn == 2:
            img = img_o[:, ::-1]
        elif nn == 3:
            img = img_o[::-1, ::-1]
        else:
            img = img_o

        return img

    if im_size is None:
        im_size = Xi.shape

    input_size, output_size, pad_size, tile_corners = \
    _get_sizes(im_size, d4a_size= d4a_size, n_conv_layers=n_conv_layers)

    Y_pred = np.zeros(im_size)
    for n_t in range(n_flips):

        X = _flip_d(Xi, n_t)

        if im_size is None:
            im_size = X.shape
        x_crop = _preprocess(X, input_size, pad_size, tile_corners)
        x_crop = np.concatenate(x_crop)
        y_pred = model_t.predict(x_crop)


        Y_pred_s = np.zeros(X.shape)
        N_s = np.zeros(X.shape)
        for (i,j), yy,xx in zip(tile_corners, y_pred, x_crop):
            Y_pred_s[i:i+output_size, j:j+output_size] += yy[:,:,1]

            if _is_debug:
                import matplotlib.pylab as plt
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(np.squeeze(xx))
                plt.subplot(1,2,2)
                plt.imshow(yy[:,:,1])

            N_s[i:i+output_size, j:j+output_size] += 1
        Y_pred += _flip_d(Y_pred_s/N_s, n_t)

    return Y_pred

def get_food_prob(mask_file, model, max_bgnd_images = 2, _is_debug = False, resizing_size = DFLT_RESIZING_SIZE):
    '''
    Predict the food probability for each pixel using a pretrained u-net model.
    '''

    with tables.File(mask_file, 'r') as fid:
        if not '/full_data' in fid:
            raise ValueError('The mask file {} does not content the /full_data dataset.'.format(mask_file))
        bgnd_o = fid.get_node('/full_data')[:max_bgnd_images].copy()

    assert bgnd_o.ndim == 3
    if bgnd_o.shape[0] > 1:
        bgnd = [np.max(bgnd_o[i:i+1], axis=0) for i in range(bgnd_o.shape[0]-1)]
    else:
        bgnd = [np.squeeze(bgnd_o)]

    min_size = min(bgnd[0].shape)
    resize_factor = min(resizing_size, min_size)/min_size
    dsize = tuple(int(x*resize_factor) for x in bgnd[0].shape[::-1])

    bgnd_s = [cv2.resize(x, dsize) for x in bgnd]
    for b_img in bgnd_s:
        Y_pred = get_unet_prediction(b_img, model, n_flips=1)

        if _is_debug:
            import matplotlib.pylab as plt
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(b_img, cmap='gray')
            plt.subplot(1, 2,2)
            plt.imshow(Y_pred, interpolation='none')

    original_size = bgnd[0].shape
    return Y_pred, original_size, bgnd_s


def cnt_solidity_func(_cnt):
    _hull = cv2.convexHull(_cnt)
    return cv2.contourArea(_cnt) / cv2.contourArea(_hull)


def avg_incnt_func(_cnt, img):
    mask = np.zeros(img.shape, np.uint8)
    mask = cv2.drawContours(mask, _cnt, 1, color=255).astype(np.uint8)
    return cv2.mean(img, mask)[0]


def eccentricity_func(_cnt):
    moments = cv2.moments(_cnt)
    a1 = (moments['mu20']+moments['mu02'])/2
    a2 = np.sqrt(
        4*moments['mu11']**2 + (moments['mu20']-moments['mu02'])**2
        ) / 2
    minor_axis = a1-a2
    major_axis = a1+a2
    eccentricity = np.sqrt(1-minor_axis/major_axis)
    return eccentricity


def get_best_scoring_cnt(cnts, food_proba, _is_debug=False):

    # print(f'raw n contours {len(cnts)}')

    # calculate patches properties
    solidities = np.array([cnt_solidity_func(c) for c in cnts])
    areas = np.array([cv2.contourArea(c) for c in cnts])
    perimeters = np.array([cv2.arcLength(c, True) for c in cnts])
    areas_over_perimeters = np.array([a/p for a, p in zip(areas, perimeters)])
    avg_probas = np.array([avg_incnt_func(c, food_proba) for c in cnts])
    # eccentricities = [eccentricity_func(c) for c in cnts]

    # normalise area and area over perimeter
    # just divide by the maximum, I don't want to lose the relative values if
    # it's only two regions
    areas_norm = areas / np.max(areas)
    aop_norm = areas_over_perimeters / np.max(areas_over_perimeters)

    # normalise
    # for all these quantities, the highest the more likely it's food
    # and they're all defined positive

    # square sum
    total_score = np.sqrt(
        solidities**2 + areas_norm**2 + aop_norm**2 + avg_probas**2)

    cnt_out = cnts[np.argmax(total_score)]

    if _is_debug:
        print({
            'area': areas,
            'area_norm': areas_norm,
            'aop': areas_over_perimeters,
            'aop_normalised': aop_norm,
            'solidity': solidities,
            'avgprob': avg_probas,
            })

        print(total_score)

    return cnt_out


def get_food_contour_nn(mask_file, model, _is_debug=False):
    '''
    Get the food contour using a pretrained u-net model.
    This function is faster if a preloaded model is given since it is very slow
    to load the model and tensorflow.
    '''

    food_prob, original_size, bgnd_images = get_food_prob(
        mask_file, model, _is_debug=_is_debug)
    # bgnd_images are only used in debug mode

    patch_m = (food_prob > 0.5).astype(np.uint8)

    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(patch_m)
        plt.show()

    cnts, _ = cv2.findContours(
        patch_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    # filter contours first to only keep the ones with a defined hull area
    cnts = [
        c for c in cnts if
        (cv2.contourArea(cv2.convexHull(c)) > 1) and
        (cv2.contourArea(c) > 1)
        ]

    # print(total_rank)
    # print(np.argmin(total_rank))
    # print(cnts[np.argmin(total_rank)])

    # pick the contour with the largest solidity
    # print(f'filtered {len(cnts)}')

    if len(cnts) == 1:
        cnts = cnts[0]
    elif len(cnts) > 1:
        # too many contours select the largest
        # cnts = max(cnts, key=cv2.contourArea)
        cnts = get_best_scoring_cnt(cnts, food_prob, _is_debug=_is_debug)
    else:
        return np.zeros([]), food_prob, 0.

    # print("this should be one and be nice:")
    # print(cnts)
    assert len(cnts == 1)

    if _is_debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(patch_m)
        fig.gca().set_title('first patch_m')
        plt.show()

    # this detects the edge and finds the outer rim of said edge
    # probably to make sure we hit the actual edge
    # rather than being a little inside the food patch
    patch_m = np.zeros(
        patch_m.shape, np.uint8)
    patch_m = cv2.drawContours(
        patch_m, cnts, -1, color=1, thickness=cv2.FILLED)
    patch_m = cv2.morphologyEx(
        patch_m, cv2.MORPH_CLOSE, disk(3), iterations=5)

    if _is_debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(patch_m)
        fig.gca().set_title('second pathc_m')
        plt.show()

    cnts, _ = cv2.findContours(
        patch_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    # print(len(cnts))
    # print(cnts[0])

    if len(cnts) == 1:
        cnts = cnts[0]
    elif len(cnts) > 1:
        # too many contours, select the largest
        cnts = max(cnts, key=cv2.contourArea)
    else:
        return np.zeros([]), food_prob, 0.

    hull = cv2.convexHull(cnts)
    hull_area = cv2.contourArea(hull)
    cnt_solidity = cv2.contourArea(cnts)/hull_area

    food_cnt = np.squeeze(cnts).astype(np.float)
    # rescale contour to be the same dimension as the original images
    food_cnt[:, 0] *= original_size[0]/food_prob.shape[0]
    food_cnt[:, 1] *= original_size[1]/food_prob.shape[1]

    if _is_debug:
        import matplotlib.pylab as plt
        img = bgnd_images[0]

        # np.squeeze(food_cnt)
        patch_n = np.zeros(img.shape, np.uint8)
        patch_n = cv2.drawContours(
            patch_n, [cnts], 0, color=1, thickness=cv2.FILLED)
        top = img.max()
        bot = img.min()
        img_n = (img-bot)/(top-bot)
        img_rgb = np.repeat(img_n[..., None], 3, axis=2)
        # img_rgb = img_rgb.astype(np.uint8)
        img_rgb[..., 0] = ((patch_n == 0)*0.5 + 0.5)*img_rgb[..., 0]

        plt.figure()
        plt.imshow(img_rgb)

        plt.plot(hull[:, :, 0], hull[:, :, 1], 'r')
        plt.title('solidity = {:.3}'.format(cnt_solidity))

    return food_cnt, food_prob, cnt_solidity


# %%

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tierpsy.helper.params.models_path import DFLT_MODEL_FOOD_CONTOUR
    from pathlib import Path
    from tqdm import tqdm
    import sys
    import os
    # mask_file = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/oig8/oig-8_ChR2_control_males_3_Ch1_11052017_161018.hdf5'

    if sys.platform == 'darwin':
        bg_path = Path('/Volumes/behavgenom$')
    elif sys.platform == 'linux':
        bg_path = Path.home() / 'net' / 'behavgenom$'
    else:
        raise Exception('not coded for this platform')

    flist_fname = bg_path / 'Luigi/exchange/all_skels_with_food_archive.txt'

    with open(flist_fname, 'r') as fid:
        skel_files = fid.read().splitlines()

    skel_files = [
        f.replace('/Volumes/behavgenom$', str(bg_path)) for f in skel_files]

    out_dir = bg_path / 'Luigi/food_tests/'
    out_log = out_dir / 'memlog.txt'
    out_data = out_dir / 'IoUs.csv'

    # load model now to prevent memory leak
    food_model = load_model(DFLT_MODEL_FOOD_CONTOUR)

    # loop through all skeletons
    for skel_file in tqdm(skel_files):

        mask_file = Path(
            str(skel_file)
            .replace('Results', 'MaskedVideos')
            .replace('_skeletons.hdf5', '.hdf5')
            )
        if not mask_file.exists():
            continue

        with tables.File(skel_file, 'r') as fid:
            if '/food_cnt_coord' in fid:
                old_food_cnt = fid.get_node('/food_cnt_coord')[:].copy()
                old_circx, old_circy = old_food_cnt.T
            else:
                continue

        food_cnt, food_prob, cnt_solidity = get_food_contour_nn(
            mask_file, food_model, _is_debug=False)
        circx, circy = food_cnt.T

        try:
            with tables.File(mask_file, 'r') as fid:
                img = fid.get_node('/full_data')[0].copy()
        except:
            print(f'cant get full_data from {mask_file}')
            continue

        food_mask = cv2.drawContours(
            np.zeros(img.shape, np.uint8), [food_cnt.astype(int)], -1,
            color=1, thickness=cv2.FILLED
            ).astype(bool)
        old_food_mask = cv2.drawContours(
            np.zeros(img.shape, np.uint8), [old_food_cnt.astype(int)], -1,
            color=1, thickness=cv2.FILLED
            ).astype(bool)

        food_IoU = (
            np.sum(np.logical_and(food_mask, old_food_mask)) /
            np.sum(np.logical_or(food_mask, old_food_mask))
            )

        with open(out_data, 'a') as fid:
            print(f'{mask_file},{food_IoU}', file=fid)

        out_name = out_dir / mask_file.with_suffix('.png').name
        if (food_IoU < 1) and (not out_name.exists()):
            fig = plt.figure()
            plt.imshow(img, cmap='gray')
            plt.plot(circx, circy)
            plt.plot(old_circx, old_circy, 'g', linestyle='--')
            plt.show()
            plt.pause(0.2)

            fig.savefig(out_name, dpi=600)
            plt.pause(0.2)

            plt.close('all')

        if sys.platform == 'linux':
            _, used_m, free_m = os.popen(
                'free -th').readlines()[-1].split()[1:]
            with open(out_log, 'a') as fout:
                print(
                    f'free: {free_m}, used:{used_m}, file:{skel_file}',
                    file=fout)
