# # -*- coding: utf-8 -*-
__version__ = '1.5.3a'

try:
    import os
    import subprocess

    cwd = os.path.dirname(os.path.abspath(__file__))
    command = ['git', 'rev-parse', 'HEAD']
    sha = subprocess.check_output(command, cwd=cwd, stderr = subprocess.DEVNULL).decode('ascii').strip()
    __version__ += '+' + sha[:7]

except Exception:
    pass

'''

1.5.3alpha
- Docker workflow for easy image building
- Bug fixes:
  * getFoodContourMorph: fixed bug in catching the output of cv2.findContours

1.5.2
- Bug fixes:
  * fixed GUI zoom sensitivity with touchpad
  * fixed dead zenodo link for test data
  * fixed bug affecting the detection of events
    (motion mode, on/off food, turning)
  * fixed bug where using the Viewer's feature to annotate worms as good/bad
    a column of booleans would appear in the results hdf5 files
    (causing incompatibility with downstream Mathematica analysis)
  * fixed bug in normalisation of area features by worm length
- Analysis:
  * Added support for multiple wells in a field of view
    (this will be made easier to taylor to other setups in the near future)
  * New pytorch based neural network for worm/non-worm classification.
    The included model might struggle to generalise to videos recorded in other
    labs, but it is possible to provide Tierpsy with a custom-trained model.
- Summarizer:
  * Summarize features by time window (can specify multiple time windows)
  * Filter trajectories by:
    - average length of worm
    - average width of worm
    - trajectory's duration
    - distance traveled
  * Optionally shorten features names to get under 50 characters
    (useful for MATLAB users)
  * Optionally drop ventrally signed features (useful if dorsal/ventral unknown)
  * Optionally select only a set of features
    (choice among a few pre-made sets, or select by keyword)
  * Summarizer writes to output(s) line-by-line as files are analysed
  * `file_name` changed to `filename` in summarizer output
  * Added a header to summarizer output with info about summarizer parameters
  * Output files contain the number of skeletons that yielded the feature stats
  * Added support to multiple wells in a field of view
- Viewer:
  * Added support to raw videos (imgstore format only)
  * Choice of plotting skeletons and contours on the main window as well
  * Added support to multiple wells in a field of view
  * Wells can be marked as good/bad
    (useful for downstream analysis outside of tierpsy)
- Installation:
  * Preferred installation route is using Docker
    (extensive instructions provided)
  * Installation from source is a good option on mac and linux


1.5.1
- Bug fixes
- Improved documentation

1.5.1-beta
- Create a conda package. This will be the new way to distribute the package.
- Merge part of the code in tierpsy-features and open-worm-analysis-toolbox to remove them as dependencies.
- Merge test suite into the main package.

1.5.0
- Bug corrections.

1.5.0-beta
- Complete the integration with tierpsy features formalizing two different feature paths.
- Reorganize the types of analysis, and deal with deprecated values.
- Add plot feature option in the tracker viewer to visualize individual worm features.
- Make the tracker viewer compatible with the _features.hdf5 files and deprecate the WT2 viewer
- Add app to collect the feature summary of different videos.



1.5.0-alpha
- Add tierpsy features as FEAT_INIT, FEAT_TIERPSY.
- Reorganize and improve GUIs, particularly "Set Parameters".
- Fix background subtraction to deal with light background.
- Add analysis point ('WORM_SINGLE') to tell the algorithm that it is expected only one worm in the video.
- Make the calculation of FPS from timestamp more tolerant to missed/repeated frames.
- Change head/tail identification to deal with worms with very low to none global displacement.

1.4.0
- Schafer's lab tracker ready for release:
	* Remove CNT_ORIENT as a separated checkpoint and add it rather as a pre-step using a decorator.
	* Stage aligment failures throw errors instead of continue silently and setting an error flag.
- Bug fixes

1.4.0b0
- Remove MATLAB dependency.
- Uniformly the naming event features (coil/coils omega_turn/omega_turns forward_motion/forward ...)
- Add food features and food contour analysis (experimental)
- Improvements to the GUI
- Bug fixes

1.3
- Major internal organization.
- Documentation
- First oficial release.

1.2.1
- Major changes in internal organization of TRAJ_CREATE TRAJ_JOIN
- _trajectories.hdf5 is deprecated. The results of this file are going to be saved in _skeletons.hdf5
- GUI Multi-worm tracker add the option of show trajectories.

1.2.0
- Major refactoring
- Add capability of identifying worms using a pre-trained neural network (not activated by default).
- Separated the creation of the control table in the skeletons file (trajectories_data) from the actually
skeletons calculation. The point SKEL_INIT now preceds SKEL_CREATION.

1.1.1
- Cumulative changes and bug corrections.

1.1.0
- Fish tracking (experimental) and fluorescence tracking.
- major changes in the internal paralization, should make easier to add or remove steps on the tracking.
- correct several bugs.
'''
