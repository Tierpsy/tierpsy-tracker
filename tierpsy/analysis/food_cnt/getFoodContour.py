#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""
import os
import tables
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from keras.models import load_model

from .getFoodContourNN import get_food_contour_nn
from .getFoodContourMorph import get_food_contour_morph

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name



def calculate_food_cnt(mask_file, use_nn_food_cnt, model_path, _is_debug=False, solidity_th=0.98):
    if use_nn_food_cnt:
        if not os.path.exists(model_path):
            warnings.warn('The model to obtain the food contour was not found. Nothing to do here...\n If you dont have a valid model. You could try to set `food_method=MORPH` to use a different algorithm.')
            return

        food_model = load_model(model_path)
        food_cnt, food_prob, cnt_solidity = get_food_contour_nn(
            mask_file, food_model, _is_debug=_is_debug)
        if cnt_solidity < solidity_th:
            food_cnt = np.zeros(0)

    else:
        food_cnt = get_food_contour_morph(mask_file, _is_debug=_is_debug)

    return food_cnt


def getFoodContour(mask_file,
                skeletons_file,
                use_nn_food_cnt,
                model_path,
                solidity_th=0.98,
                _is_debug = False
                ):
    base_name = get_base_name(mask_file)

    progress_timer = TimeCounter('')
    print_flush("{} Calculating food contour {}".format(base_name, progress_timer.get_time_str()))


    food_cnt = calculate_food_cnt(mask_file,
                                  use_nn_food_cnt = use_nn_food_cnt,
                                  model_path = model_path,
                                  solidity_th=  solidity_th,
                                  _is_debug = _is_debug)

    #store contour coordinates into the skeletons file and mask_file the contour file
    for fname in [skeletons_file, mask_file]:
        with tables.File(fname, 'r+') as fid:
            if '/food_cnt_coord' in fid:
                fid.remove_node('/food_cnt_coord')

            #if it is a valid contour save it
            if food_cnt is not None and \
               food_cnt.size >= 2 and \
               food_cnt.ndim == 2 and \
               food_cnt.shape[1] == 2:

                tab = fid.create_array('/',
                                       'food_cnt_coord',
                                       obj=food_cnt)
                tab._v_attrs['use_nn_food_cnt'] = int(use_nn_food_cnt)

