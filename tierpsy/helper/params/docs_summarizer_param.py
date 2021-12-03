'''
List of default values and description of the tierpsy.gui.Summarizer
'''

from .helper import repack_dflt_list

try:
    from os import sched_getaffinity
    _max_n_processes = max(1, len(sched_getaffinity(0))-1)
except (AttributeError, ImportError):
    from multiprocessing import cpu_count
    _max_n_processes = max(1, cpu_count()-1)
except:
    _max_n_processes = 1


summarizer_valid_options = {
    'feature_type':['openworm','tierpsy'],
    'summary_type' : ['plate', 'trajectory', 'plate_augmented'],
    'time_units' : ['frame_numbers', 'seconds'],
    'select_feat' : ['all', 'tierpsy_8', 'tierpsy_16', 'tierpsy_256', 'tierpsy_2k','select_by_keywords'],
    'filter_time_units' : ['frame_numbers', 'seconds'],
    'filter_distance_units' : ['pixels', 'microns'],
}

dflt_args_list = [
    ('root_dir',
        '',
        'Root directory where the features files are located and the results are going to be saved.'
        ),
    ('feature_type',
        'tierpsy',
        '''
        Type of feature file to be used. Either the original OpenWorm features or the new Tierpsy Features.
        '''
        ),
    ('summary_type',
        'plate',
        '''
        Indicates if the summary is going to be done over each individual plate, each individual trajectory or
        is going to be a data augmentation by randomingly sampling over a subset of the plate trajectories.
        '''
        ),
    ('n_parallel',
        1,
        '''
        Number of files to be processed in parallel. Not limited in the GUI,
        but in reality limited by the resources available. Capped to one less
        than the number of cores available, unless running on PBS or SLURM HPC,
        in which case no upper cap is set as python does not know the number of
        cores allocated to it
        '''
        ),
    ('is_manual_index',
        False,
        'Set to true to calculate data using manually edited trajectories used Tierpsy Viewer.'
        ),
    ('time_windows',
        '0:end',
        '''
        Define time windows to get feature summaries from the parts of the video included in each window.
        Each window must be defined by the start_time and the end_time connected by ':',
        for example 'start_time:end_time'.
        Different windows must be separated by ',', for example 'start_time_1:end_time_1, start_time_2:end_time_2'.
        A sequence of equally sized windows can be defined using the format 'start_time:end_time:step'.
        A single window can be comprised of several time intervals using the format
        'start_time1:end_time1 + start_time2:end_time2'.
        ATTENTION: the start_time is included in the window, but the end_time is not included.
        '''
        ),
    ('time_units',
        'frame_numbers',
        'Units of start time and end time in Time Windows.'
        ),
    ('select_feat',
        'all',
        '''
        Get a pre-selected subset of tierpsy features or select features by keywords.
        '''
        ),
    ('abbreviate_features',
        False,
        'Shorten the feature names so that they are compatible with MATLAB'
        ),
    ('dorsal_side_known',
        False,
        '''
        Dorsal side of worm has been annotated on videos. If dorsal side is not
        known then only absolute values for d/v signed features are returned.
        (nb. d/v features are not included in the 2k, 512, 256, 16 and 8 sets anyway)
        ''',
        ),
    ('keywords_include',
        '',
        '''
        Select only features that contain any of the given keywords. Provide keywords separated by comma \',\'.
        '''
        ),
    ('keywords_exclude',
        '',
        '''
        Exclude features that contain any of the given keywords. Provide keywords separated by comma \',\'.
        '''
        ),
    ('filter_time_min',
        '',
        '''
        Minimum length that a trajetory must have to be included in the calculation of feature summaries.
        '''
        ),
    ('filter_time_units',
        'frame_numbers',
        'Units of min trajectory length threshold.'
        ),
    ('filter_travel_min',
        '',
        '''
        If the total distance traveled during a trajetory is above this threshold,
        the trajectory will be included in the calculation of feature summaries.
        '''
        ),
    ('filter_distance_units',
        'pixels',
        'Units of distance used in the filtering thresholds.'
        ),
    ('filter_length_min',
        '',
        '''
        If the average worm length within a trajectory is above this threshold,
        the trajectory will be included in the calculation of feature summaries.
        '''
        ),
    ('filter_length_max',
        '',
        '''
        If the average worm length within a trajectory is below this threshold,
        the trajectory will be included in the calculation of feature summaries.
        '''
        ),
    ('filter_width_min',
        '',
        '''
        If the average width of the worm's midbody within a trajectory is above this threshold,
        the trajectory will be included in the calculation of feature summaries.
        '''
        ),
    ('filter_width_max',
        '',
        '''
        If the average width of the worm's midbody within a trajectory is below this threshold,
        the trajectory will be included in the calculation of feature summaries.
        '''
        ),
    ('n_folds',
        5,
        '''
        Number of times each subsampling is going to be repeated (only for plate_augmentation).
        '''
        ),

    ('frac_worms_to_keep',
        0.8,
        'Fraction of the total number trajectories that is going to be keep for each subsampling.'
        ),
    ('time_sample_seconds',
        600,
        'Total time in seconds that is going to be keep for each subsampling.'
        )
    ]

summarizer_args_dflt, summarizer_args_info = repack_dflt_list(dflt_args_list, valid_options=summarizer_valid_options)
