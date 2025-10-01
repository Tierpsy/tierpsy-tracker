'''
List of default values and description of the tracker parameters.
'''

from tierpsy import DFLT_SPLITFOV_PARAMS_FILES
from .docs_analysis_points import valid_analysis_types, dlft_analysis_type
from .helper import repack_dflt_list




dflt_param_list = [
    ('analysis_type',
        dlft_analysis_type,
        'Flag that defines the type of data expected and the consequent code that would be executed.'
        ),
    ('analysis_checkpoints',
        [],
        'Analysis points to be executed, if it is empty the points will be deduced from the analysis_type.'
        ),
    ('mask_min_area',
        50,
        'Minimum area in pixels for an object to be included in the compression mask.'
        ),
    ('mask_max_area',
        int(1e8),
        'Maximum area in pixels for an object to be included in the compression mask.'
        ),
    ('thresh_C',
        15,
        '''
        Used to calculate Mask.  Used by the adaptative thresholding algorithm.
        The threshold for a pixel is the block mean minus thresh_C.
        <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>
        '''
        ),
    ('thresh_block_size',
        61,
        '''
        Used to calculate Mask.
        The size of a pixel neighborhood (block) that is used to calculate
        a threshold value for the pixel by the adaptative thresholding.
        <http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>
        '''
        ),
    ('dilation_size',
        9,
        'Size of the structural element used by morphological operations to calculate the worm mask.'
        ),

    ('save_full_interval',
        -1,
        '''
        Frequence in frames that an unprocessed frame is going to be saved in /full_data.
        If the value is negative, it would be calculated as 200*expected_fps.
        '''
        ),

    ('compression_buff',
        -1,
        '''Number of images "min-averaged" used to calculate the image mask.
        If the value is negative it would be set using the value of expected_fps.
        '''
        ),
    ('keep_border_data',
        False,
        '''
        Set it to false if you want to remove any
        binary blob that touches the border.
        '''),


    ('is_light_background',
        True,
        '''
        Set to **true** for dark objects over a light ground
        and **false** for light objects over a dark background.
        '''
        ),

    ('is_extract_timestamp',
        True,
        '''
        Set to **true** to extract metadata (timestamps)
        from the original video file. The timestamp can be used
        to calculate the FPS and to identify drop frames.
        Extract the timestamp is a slow step since it uses ffprobe to read whole video.
        If you believe that your video does not have a significative number of drop
        frames and you know the frame rate, or simply realise that
        ffprobe cannot extract the timestamp correctly,
        it is recommended to set this value to **false**.
        '''
        ),
    ('expected_fps',
        -1,
        '''
        Expected frames per seconds. If the value is negative it would be set to 1
        and the units to *frames* to calculate the worm features.
        This value will be superseded if there is a valid timestamp in the video
        (there are not nan values and they increase monotonically), but it can be used
        to identify problems with the timestamp.
        '''
        ),
    ('microns_per_pixel', -1.,
        '''
        Pixel size in micrometers.
        If the value is negative it would be set to 1
        and the units to *pixels* to calculate the worm features.
        '''
        ),

    ('mask_bgnd_buff_size',
        -1,
        '''
        Number of images used to estimate the background subtracted during compression.
        If it is a negative number the background subtraction is deactivated.
        '''
        ),
    ('mask_bgnd_frame_gap',
        -1,
        '''
        Frame gap between the images used to estimate the background subtracted during compression.
        If it is a negative number the background subtraction is deactivated.
        '''
        ),

    ('is_full_bgnd_subtraction',
        False,
        '''
        Set True if you want to use background subtraction even in the compressed files. The background subtraction
        could introduce artifacts in the worm shape if the background calculation is less than optimal. This has
        little impact in the compression since we only need an approximate location of the worm but could
        affect the skeletonization.
        '''
        ),

    ('worm_bw_thresh_factor',
        1.05,
        '''
        Factor multiplied by the threshold used to create invidual binary images
        used to create the trajectories and calculate the skeletons.
        If the particle mask is too big after tracking increase this value, if it is too small decrease it.
        '''
        ),
    ('strel_size',
        5,
        'Structural element size. Used to calculate the binary masks used for the skeletons and trajectories.'
        ),
    ('traj_min_area',
        25,
        '''
        Minimum area in pixels for an object to be considered as a part of a trajectory.
        '''
        ),
    ('traj_min_box_width',
        5,
        'Minimum width of bounding box in pixels for an object to be considered as a part of a trajectory.'
        ),

    ('traj_max_allowed_dist',
        25,
        'Maximum displacement between frames for two particles to consider part of the same track.'
        ),
    ('traj_max_frames_gap',
        0,
        'Max gap in frames allowed between joined trajectories.'
        ),

    ('traj_area_ratio_lim',
        2,
        'Area ratio between blob areas in different frames to be considered part of the same trajectory.'
        ),


    ('roi_size',
        -1,
        '''
        Size of the Region Of Interest used to calculate the skeleton.
        If it is set to -1 it would be calculated from the data.
        '''
        ),

    ('w_num_segments',
        24,
        '''
        Number of segments used to calculate the skeleton curvature
        (or half the number of segments used for the contour curvature).
        Reduced for rounder objects and increase for sharper organisms.'
        '''
        ),
    ('w_head_angle_thresh',
        60,
        'Angle threshold for a peak in the contour curvature to be considered as the head or tail.'
        ),

    ('resampling_N',
        49,
        'Number of segments used to normalize the worm skeleton and contours.'
        ),

    ('max_gap_allowed_block',
        -1,
        '''
        Maximum number of missing frames for a group of skeletons to be considered
        part of the same group in the head/tail correction using MOVEMENT.
        If this value is negative it would be set as fps/2.
        '''
        ),
    ('ht_orient_segment',
        -1,
        '''
        Segment size used to calculate the head/tail angular speed
        used to orient the head/tail by MOVEMENT.
        If this value is negative it would be set as round(resampling_N/10).
        '''
        ),
    ('filt_bad_seg_thresh',
        0.8,
        'minimum fraction of succesfully skeletonized frames in a worm trajectory to be considered valid.'
        ),
    ('filt_max_width_ratio',
        2.25,
        '''
        Maximum width radio between midbody and head or tail to be considered as a valid skeleton.
        If the worm more than double its width from the head/tail it might be coiled.
        '''
        ),
    ('filt_max_area_ratio',
        6,
        '''
        Maximum area ratio between head+tail and the rest of the body to be a valid skeleton.
        Find if the head and tail too small or the body too large to be a realistic worm.'
        '''
        ),

    ('filt_min_displacement',
        10,
        '''
        Minimum total displacement of a trajectory to be used to calculate the threshold to detect bad skeletons.
        Useful to detect pair of worms detected as a single particle.
        '''
        ),
    ('filt_critical_alpha',
        0.01,
        '''Critical chi2 alpha used in the mahalanobis distance to considered a skeleton a global outlier (bad skeleton).
        Useful to detect pair of worms detected as a single particle.
        '''
        ),


    ('int_save_maps',
        False,
        '**true** to save the intensity maps and not only the profile along the worm major axis.'
        ),
    ('int_avg_width_frac',
        0.3,
        '''
        Width fraction of the intensity maps used to calculate the
        profile along the worm major axis in the head/tail correction by intensity.
        '''
        ),
    ('int_width_resampling',
        15,
        'Width in pixels of the intensity maps used for the head/tail correction by intensity.'
        ),
    ('int_length_resampling',
        131,
        'Length in pixels of the intensity maps used for the head/tail correction by intensity.'
        ),
    ('int_max_gap_allowed_block',
        -1,
        '''
        Maximum number of missing frames for a group of skeletons to be considered
        part of the same group in the head/tail correction using INTENSITY.
        If this value is negative it would be set as fps/4.
        '''
        ),

    ('head_tail_int_method',
        'MEDIAN_INT',
        'Method to used to correct head/tail based on the intensity profile.'
        ),
    ('split_traj_time',
        90,
        'Time in SECONDS that a worm trajectory will be subdivided to calculate the splitted features.'
        ),

    ('ventral_side',
        '',
        'Ventral side orientation. Used only if "analysis_type" is set to "WT2".'
        ),

    ('feat_skel_smooth_window',
        5,
        'Window (in number segments) used to smooth each individual skeleton by Tierpsy Features.'
        ),
    ('feat_coords_smooth_window_s',
        0.25,
        'Window (in number seconds) used to smooth each of the skeletons coordinates over time by Tierpsy Features. Set to zero to deactivate the smoothing over time.'
        ),
    ('feat_gap_to_interp_s',
        0.25,
        'Maximum time gap between missing skeletons be interpolated by Tierpsy Features. Set to zero to deactivate the interpolation over time.'
        ),
    ('feat_derivate_delta_time',
        1/3,
        'Time window used to calculate the derivatives (including velocity) by Tierpsy Features.'
        ),

    ('n_cores_used',
        1,
        '''
        EXPERIMENTAL. Number of core used.
        Currently it is only suported by TRAJ_CREATE and it is only recommended at high particle densities.
        '''),

    ('nn_filter_to_use',
        'none',
        """
        Set to "pytorch_default" or "tensorflow_default" if you want to
        use a pretrained neural network model to filter worms.
        Set to "none" if you'd rather not use a NN filter.
        EXPERIMENTAL: set to "custom" to provide your own pytorch model.
        Do contact the developers to collaborate on this.
        The default models are optimized for AEX setup and might not work with
        other setups.
        "pytorch_default" uses a NN that was trained using Hydra data, while
        "tensorflow_default" uses the legacy NN,trained on Phenix data only.
        If analysis_type is setup to _AEX the "tensorflow_default" filter
        will be used by default (AEX lab legacy behaviour).
        """
        ),
    ('path_to_custom_pytorch_model',
        '',
        """
        EXPERIMENTAL. Path to a saved model_state for a pytorch CNN.
        At the moment, only the architecture and preprocessing pipeline defined in
        tierpsy-tracker/tierpsy/analysis/ske_init/pytorchModelHelper.py
        is supported.
        """
     ),
    ('use_nn_food_cnt',
        True,
        'If true a pretrained neural network model is going to use in  FOOD_CNT, else it will attempt to use morphological operations to calculate the food contour.'
        ),

    # ('MWP_total_n_wells',
    #     -1,
    #     '''
    #     Deprecated
    #     If using a Multi Well Plate so that you have more than one well in your field of view, and you want to analyse each well separately.
    #     This is the total number of wells in your MWP: 24 if you are using a 24-well plate, 48 fr a 48-well plate, etc.
    #     If this number is -1, Tierpsy will *not* analyse the wells individually.
    #     '''),
    # ('MWP_whichsideup',
    #     'upright',
    #     '''
    #     Deprecated!
    #     If using a Multi Well Plate:
    #     was it placed under the microscope upright or upside-down?
    #     '''
    #     ),
    # ('MWP_well_shape',
    #     'square',
    #     '''
    #     If using a Multi Well Plate: is each well a circle or a square?
    #     '''
    #     ),
    ('MWP_mapping',
        '',
        '''
        If using a Multi Well Plate: which json file to use for understanding
        the mappings between filename -> channel (position on a plate)
        and channel -> wells disposition?
        If this field is left empty, Tierpsy will *not* analyse the wells
        individually.
        '''
     ),
    ('MWP_path_to_custom_mapping',
        '',
        """
        Path to a json file with information about FOV splitting.
        Needs to be modelled after the ones in
        tierpsy-tracker/tierpsy/extras/splitfov_param_files/
        """
     ),
    # ('MWP_well_masked_edge',
    #     0.1,
    #     '''
    #     If using a Multi Well Plate: close to the edge of the wells the
    #     illumination is pathologically uneven (because of the agar's meniscus).
    #     To avoid false positive in the tracking we apply a mask to the video,
    #     to suppress the signal close to the edge of the wells.
    #     This parameter controls how much, relative to the widht/height of
    #     each well, this mask should extend. Upper limit is 0.5 as that would
    #     already mask out entirely a well.
    #     ''')
    ]

# #not tested (used for the zebra fish)
# ('zf_num_segments', 12, 'Number of segments to use in tail model.'),
# ('zf_min_angle', -90, 'The lowest angle to test for each segment. Angles are set relative to the angle of the previous segment.'),
# ('zf_max_angle', 90, 'The highest angle to test for each segment.'),
# ('zf_num_angles', 90, 'The total number of angles to test for each segment. Eg., If the min angle is -90 and the max is 90, setting this to 90 will test every 2 degrees, as follows: -90, -88, -86, ...88, 90.'),
# ('zf_tail_length', 60, 'The total length of the tail model in pixels.'),
# ('zf_tail_detection', 'MODEL_END_POINT', 'Algorithm to use to detect the fish tail point.'),
# ('zf_prune_retention', 1, 'Number of models to retain after scoring for each round of segment addition. Higher numbers will be much slower.'),
# ('zf_test_width', 2, 'Width of the tail in pixels. This is used only for scoring the model against the test frame.'),
# ('zf_draw_width', 2, 'Width of the tail in pixels. This is used for drawing the final model.'),
# ('zf_auto_detect_tail_length', True, 'Flag to determine whether zebrafish tail length detection is used. If set to True, values for zf_tail_length, zf_num_segments and zf_test_width are ignored.')

valid_options = {
    'analysis_type': valid_analysis_types,
    'ventral_side': ['', 'clockwise', 'anticlockwise', 'unknown'],
    'head_tail_int_method': ['MEDIAN_INT', 'HEAD_BRIGHTER'],
    'nn_filter_to_use': ['tensorflow_default', 'pytorch_default',
                         'custom', 'none'],
    'use_nn_filter': [True, False],
    'MWP_total_n_wells': [-1, 24, 48, 96], # caveat: whether the analysis will work or not depends on the code in tierpsy.analysis.compress.FOVMultiWellSplitter
    'MWP_whichsideup': ['upright', 'upside-down'],
    'MWP_well_shape': ['circle', 'square'],
    'MWP_mapping': DFLT_SPLITFOV_PARAMS_FILES + ['custom', '']
}

#repack data into dictionaries
default_param, info_param = repack_dflt_list(dflt_param_list, valid_options)
