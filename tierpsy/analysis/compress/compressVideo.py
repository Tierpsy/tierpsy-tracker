# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:19:58 2015

@author: ajaver
"""
import os

import cv2
import tables
import numpy as np

from tierpsy.analysis.compress.BackgroundSubtractor import BackgroundSubtractorVideo
from tierpsy.analysis.compress.extractMetaData import store_meta_data, read_and_save_timestamp
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
from tierpsy.helper.params import compress_defaults, set_unit_conversions
from tierpsy.helper.misc import TimeCounter, print_flush, TABLE_FILTERS
from tierpsy.analysis.split_fov.helper import parse_camera_serial
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter


def getROIMask(
        image,
        min_area,
        max_area,
        thresh_block_size,
        thresh_C,
        dilation_size,
        keep_border_data,
        is_light_background,
        wells_mask=None):
    '''
    Calculate a binary mask to mark areas where it is possible to find worms.
    Objects with less than min_area or more than max_area pixels are rejected.
        > min_area -- minimum blob area to be considered in the mask
        > max_area -- max blob area to be considered in the mask
        > thresh_C -- threshold used by openCV adaptiveThreshold
        > thresh_block_size -- block size used by openCV adaptiveThreshold
        > dilation_size -- size of the structure element to dilate the mask
        > keep_border_data -- (bool) if false it will reject any blob that touches the image border
        > is_light_background -- (bool) true if bright field, false if fluorescence
        > wells_mask -- (bool 2D) mask that covers (with False) the edges of wells in a MW plate

    '''
    # Objects that touch the limit of the image are removed. I use -2 because
    # openCV findCountours remove the border pixels
    IM_LIMX = image.shape[0] - 2
    IM_LIMY = image.shape[1] - 2

    #this value must be at least 3 in order to work with the blocks
    thresh_block_size = max(3, thresh_block_size)
    if thresh_block_size % 2 == 0:
        thresh_block_size += 1  # this value must be odd

    #let's add a median filter, this will smooth the image, and eliminate small variations in intensity
    # now done with opencv instead of scipy
    image = cv2.medianBlur(image, 5)

    # adaptative threshold is the best way to find possible worms. The
    # parameters are set manually, they seem to work fine if there is no
    # condensation in the sample
    if not is_light_background:  # invert the threshold (change thresh_C->-thresh_C and cv2.THRESH_BINARY_INV->cv2.THRESH_BINARY) if we are dealing with a fluorescence image
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            thresh_block_size,
            -thresh_C)
    else:
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            thresh_block_size,
            thresh_C)

    # find the contour of the connected objects (much faster than labeled
    # images)

    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # find good contours: between max_area and min_area, and do not touch the
    # image border
    goodIndex = []
    for ii, contour in enumerate(contours):
        if not keep_border_data:
            if wells_mask is None:
                # eliminate blobs that touch a border
                # TODO: double check this next line. I suspect contour is in
                # x,y and not row columns
                keep = not np.any(contour == 1) and \
                    not np.any(contour[:, :, 0] ==  IM_LIMY)\
                    and not np.any(contour[:, :, 1] == IM_LIMX)
            else:
                # keep if no pixel of contour is in the 0 part of the mask
                keep = not np.any(wells_mask[contour[:, :, 1],
                                             contour[:, :, 0]] == 0)
        else:
            keep = True

        if keep:
            area = cv2.contourArea(contour)
            if (area >= min_area) and (area <= max_area):
                goodIndex.append(ii)

    # typically there are more bad contours therefore it is cheaper to draw
    # only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 1, cv2.FILLED)

    # drawContours left an extra line if the blob touches the border. It is
    # necessary to remove it
    mask[0, :] = 0
    mask[:, 0] = 0
    mask[-1, :] = 0
    mask[:, -1] = 0

    # dilate the elements to increase the ROI, in case we are missing
    # something important
    struct_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    mask = cv2.dilate(mask, struct_element, iterations=3)

    return mask

def normalizeImage(img):
    # normalise image intensities if the data type is other
    # than uint8
    image = image.astype(np.double)

    imax = img.max()
    imin = img.min()
    factor = 255/(imax-imin)

    imgN = ne.evaluate('(img-imin)*factor')
    imgN = imgN.astype(np.uint8)

    return imgN, (imin, imax)

def reduceBuffer(Ibuff, is_light_background):
    if is_light_background:
        return np.min(Ibuff, axis=0)
    else:
        return np.max(Ibuff, axis=0)

def createImgGroup(fid, name, tot_frames, im_height, im_width, is_expandable=True):
    parentnode, _, name = name.rpartition('/')
    parentnode += '/'

    if is_expandable:
        img_dataset = fid.create_earray(
                        parentnode,
                        name,
                        atom=tables.UInt8Atom(),
                        shape =(0,
                             im_height,
                             im_width),
                        chunkshape=(1,
                             im_height,
                             im_width),
                        expectedrows=tot_frames,
                        filters=TABLE_FILTERS
                        )
    else:
        img_dataset = fid.create_carray(
                        parentnode,
                        name,
                        atom=tables.UInt8Atom(),
                        shape =(tot_frames,
                             im_height,
                             im_width),
                        filters=TABLE_FILTERS
                        )

    img_dataset._v_attrs["CLASS"] = np.string_("IMAGE")
    img_dataset._v_attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
    img_dataset._v_attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
    img_dataset._v_attrs["DISPLAY_ORIGIN"] = np.string_("UL")  # not rotated
    img_dataset._v_attrs["IMAGE_VERSION"] = np.string_("1.2")

    return img_dataset

def initMasksGroups(fid, expected_frames, im_height, im_width,
    attr_params, save_full_interval, is_expandable=True):

    # open node to store the compressed (masked) data
    mask_dataset = createImgGroup(fid, "/mask", expected_frames, im_height, im_width, is_expandable)


    tot_save_full = (expected_frames // save_full_interval) + 1
    full_dataset = createImgGroup(fid, "/full_data", tot_save_full, im_height, im_width, is_expandable)
    full_dataset._v_attrs['save_interval'] = save_full_interval


    assert all(x in ['expected_fps', 'is_light_background', 'microns_per_pixel'] for x in attr_params)
    set_unit_conversions(mask_dataset, **attr_params)
    set_unit_conversions(full_dataset, **attr_params)

    if is_expandable:
        mean_intensity = fid.create_earray('/',
                                        'mean_intensity',
                                        atom=tables.Float32Atom(),
                                        shape=(0,),
                                        expectedrows=expected_frames,
                                        filters=TABLE_FILTERS)
    else:
        mean_intensity = fid.create_carray('/',
                                        'mean_intensity',
                                        atom=tables.Float32Atom(),
                                        shape=(expected_frames,),
                                        filters=TABLE_FILTERS)

    return mask_dataset, full_dataset, mean_intensity



def compressVideo(video_file, masked_image_file, mask_param,  expected_fps=25,
                  microns_per_pixel=None, bgnd_param ={}, buffer_size=-1,
                  save_full_interval=-1, max_frame=1e32, is_extract_timestamp=False,
                  fovsplitter_param={}):
    '''
    Compresses video by selecting pixels that are likely to have worms on it and making the rest of
    the image zero. By creating a large amount of redundant data, any lossless compression
    algorithm will dramatically increase its efficiency. The masked images are saved as hdf5 with gzip compression.
    The mask is calculated over a minimum projection of an image stack. This projection preserves darker regions
    (or brighter regions, in the case of fluorescent labelling)
    where the worm has more probability to be located. Additionally it has the advantage of reducing
    the processing load by only requiring to calculate the mask once per image stack.
     video_file --  original video file
     masked_image_file --
     buffer_size -- size of the image stack used to calculate the minimal projection and the mask
     save_full_interval -- have often a full image is saved
     max_frame -- last frame saved (default a very large number, so it goes until the end of the video)
     mask_param -- parameters used to calculate the mask
    '''

    #get the default values if there is any bad parameter
    output = compress_defaults(masked_image_file,
                                expected_fps,
                                buffer_size = buffer_size,
                                save_full_interval = save_full_interval)

    buffer_size = output['buffer_size']
    save_full_interval = output['save_full_interval']

    if len(bgnd_param) > 0:
        is_bgnd_subtraction = True
        assert bgnd_param['buff_size']>0 and bgnd_param['frame_gap']>0
    else:
        is_bgnd_subtraction = False

    if len(fovsplitter_param) > 0:
        is_fov_tosplit = True
        assert all(key in fovsplitter_param for key in ['total_n_wells', 'whichsideup', 'well_shape'])
        assert fovsplitter_param['total_n_wells']>0
    else:
        is_fov_tosplit = False

    # processes identifier.
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    # select the video reader class according to the file type.
    vid = selectVideoReader(video_file)

    # delete any previous  if it existed
    with tables.File(masked_image_file, "w") as mask_fid:
        pass

    #Extract metadata
    if is_extract_timestamp:
        # extract and store video metadata using ffprobe
        #NOTE: i cannot calculate /timestamp until i am sure of the total number of frames
        print_flush(base_name + ' Extracting video metadata...')
        expected_frames = store_meta_data(video_file, masked_image_file)

    else:
        expected_frames = 1

    # Initialize background subtraction if required

    if is_bgnd_subtraction:
        print_flush(base_name + ' Initializing background subtraction.')
        bgnd_subtractor = BackgroundSubtractorVideo(video_file, **bgnd_param)

    # intialize some variables
    max_intensity, min_intensity = np.nan, np.nan
    frame_number = 0
    full_frame_number = 0
    image_prev = np.zeros([])

    # Initialise FOV splitting if needed
    if is_bgnd_subtraction:
        img_fov = bgnd_subtractor.bgnd.astype(np.uint8)
    else:
        ret, img_fov = vid.read()
        # close and reopen the video, to restart from the beginning
        vid.release()
        vid = selectVideoReader(video_file)

    if is_fov_tosplit:
        # TODO: change class creator so it only needs the video name? by using
        # Tierpsy's functions such as selectVideoReader it can then read the first image by itself

        camera_serial = parse_camera_serial(masked_image_file)

        fovsplitter = FOVMultiWellsSplitter(img_fov,
                                            camera_serial=camera_serial,
                                            px2um=microns_per_pixel,
                                            **fovsplitter_param)
        wells_mask = fovsplitter.wells_mask
    else:
        wells_mask = None



    # initialize timers
    print_flush(base_name + ' Starting video compression.')


    if expected_frames == 1:
        progressTime = TimeCounter('Compressing video.')
    else:
        #if we know the number of frames display it in the progress
        progressTime = TimeCounter('Compressing video.', expected_frames)


    with tables.File(masked_image_file, "r+") as mask_fid:

        #initialize masks groups
        attr_params = dict(
            expected_fps = expected_fps,
            microns_per_pixel = microns_per_pixel,
            is_light_background = int(mask_param['is_light_background'])
            )
        mask_dataset, full_dataset, mean_intensity = initMasksGroups(mask_fid,
            expected_frames, vid.height, vid.width,
            attr_params, save_full_interval)

        if is_bgnd_subtraction:
            bg_dataset = createImgGroup(mask_fid, "/bgnd", 1, vid.height, vid.width, is_expandable=False)
            # because we only save the one background:
            bg_dataset._v_attrs['save_interval'] = vid.frame_max-vid.first_frame + 1
            bg_dataset[0,:,:] = img_fov

        if vid.dtype != np.uint8:
            # this will worm as flags to be sure that the normalization took place.
            normalization_range = mask_fid.create_earray('/',
                                        'normalization_range',
                                        atom=tables.Float32Atom(),
                                        shape=(0, 2),
                                        expectedrows=expected_frames,
                                        filters=TABLE_FILTERS
                                        )

        while frame_number < max_frame:

            ret, image = vid.read()
            if ret != 0:
                # increase frame number
                frame_number += 1

                # opencv can give an artificial rgb image. Let's get it back to
                # gray scale.
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                if image.dtype != np.uint8:
                    # normalise image intensities if the data type is other
                    # than uint8
                    image, img_norm_range = normalizeImage(image)
                    normalization_range.append(img_norm_range)

                #limit the image range to 1 to 255, 0 is a reserved value for the background
                assert image.dtype == np.uint8
                image = np.clip(image, 1,255)

                # Add a full frame every save_full_interval
                if frame_number % save_full_interval == 1:
                    full_dataset.append(image[np.newaxis, :, :])
                    full_frame_number += 1

                # buffer index
                ind_buff = (frame_number - 1) % buffer_size

                # initialize the buffer when the index correspond to 0
                if ind_buff == 0:
                    Ibuff = np.zeros(
                        (buffer_size, vid.height, vid.width), dtype=np.uint8)

                # add image to the buffer
                Ibuff[ind_buff, :, :] = image.copy()
                mean_int = np.mean(image)
                assert mean_int >= 0
                mean_intensity.append(np.array([mean_int]))

            else:
                # sometimes the last image is all zeros, control for this case
                if np.all(Ibuff[ind_buff] == 0):
                    frame_number -= 1
                    ind_buff -= 1

                # close the buffer
                Ibuff = Ibuff[:ind_buff + 1]

            # mask buffer and save data into the hdf5 file
            if (ind_buff == buffer_size - 1 or ret == 0) and Ibuff.size > 0:
                if is_bgnd_subtraction:
                    Ibuff_b  = bgnd_subtractor.apply(Ibuff, frame_number)
                else:
                    Ibuff_b = Ibuff

                #calculate the max/min in the of the buffer
                img_reduce = reduceBuffer(Ibuff_b, mask_param['is_light_background'])

                mask = getROIMask(img_reduce, wells_mask=wells_mask, **mask_param)

                Ibuff *= mask

                # now apply the well_mask if is MWP
                if is_fov_tosplit:
                    fovsplitter.apply_wells_mask(Ibuff) # Ibuff will be modified after this

                # add buffer to the hdf5 file
                frame_first_buff = frame_number - Ibuff.shape[0]
                mask_dataset.append(Ibuff)

            if frame_number % 500 == 0:
                # calculate the progress and put it in a string
                progress_str = progressTime.get_str(frame_number)
                print_flush(base_name + ' ' + progress_str)

            # finish process
            if ret == 0:
                break

        # close the video
        vid.release()

    # save fovsplitting data
    if is_fov_tosplit:
        fovsplitter.write_fov_wells_to_file(masked_image_file)

    read_and_save_timestamp(masked_image_file)
    print_flush(base_name + ' Compressed video done.')
