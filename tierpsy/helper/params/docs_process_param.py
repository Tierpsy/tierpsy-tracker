'''
List of default values and description of the tierpsy.processing.progressMultipleFilesFun
'''

import os
from tierpsy import IS_FROZEN
from .helper import repack_dflt_list

#get 90% of the number of cores as the default max_num_process 
import multiprocessing as mp
_max_num_process = max(1, int(mp.cpu_count()*0.9)) 

#set this to false if it is a compiled version (we do not want to scare the users with the traceback information)
_is_debug = not IS_FROZEN


dflt_args_list = [
    ('video_dir_root', 
        '', 
        'Root directory where the raw videos are located.'
        ),
    ('mask_dir_root', 
        '', 
        '''
        Root directory where the masked videos (after COMPRESSION) are located or will be stored.
        If it is not given it will be created replacing RawVideos by MaskedVideos in the video_dir_root.
        '''
        ),
    ('results_dir_root', 
        '', 
        '''
        Root directory where the tracking results  are located or will be stored. 
        If it is not given it will be created replacing MaskedVideos by Results in the mask_dir_root.
        '''
        ),
    ('tmp_dir_root',
        os.path.join(os.path.expanduser("~"), 'Tmp'),
        'Temporary directory where the unfinished analysis files are going to be stored.'
        ),
    ('videos_list',
        '',
        '''
        File containing the full path of the files to be analyzed. 
        If it is not given files will be searched in video_dir_root or mask_dir_root 
        using pattern_include and pattern_exclude.
        '''
        ),
    
    ('json_file',
        '',
        'File (.json) containing the tracking parameters.'
        ),
    ('max_num_process',
        _max_num_process,
        'Maximum number of files to be processed simultaneously.'
        ),

    ('pattern_include',
        '*.mp4',
        'Pattern used to search files to be analyzed.'
        ),
    ('pattern_exclude',
        '',
        'Pattern used to exclude files to be analyzed.'
        ),
    
    ('is_copy_video',
        False,
        'Set **true** to copy the raw videos files to the temporary directory.'
        ),
    ('copy_unfinished',
        False,
        'Copy files to the final destination even if the analysis was not completed successfully.'
        ),
    ('analysis_checkpoints',
        [],
        'Points in the analysis to be executed.'),

    ('force_start_point',
        '',
        'Force the program to start at a specific point in the analysis.'
        ),
    ('end_point',
        '',
        'Stop the analysis at a specific point.'
        ),
    
    ('only_summary',
        False,
        'Set **true** if you only want to see a summary of how many files are going to be analyzed.'
        ),
    ('unmet_requirements',
        False,
        'Use this flag if you only want to print the unmet requirements of the invalid source files.'
        ),
    ('refresh_time',
        10.,
        'Refresh time in seconds of the progress screen.'
        ),
    
    ('is_debug',
        _is_debug,
        'Print debug information after an error.'
        ),

    ]

process_valid_options = {}
proccess_args_dflt, proccess_args_info = repack_dflt_list(dflt_args_list, valid_options=process_valid_options)
