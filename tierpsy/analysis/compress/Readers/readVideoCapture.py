#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:56:55 2016

@author: ajaver
"""

import cv2
import warnings


class readVideoCapture():
    def __init__(self, video_file):
        vid = cv2.VideoCapture(video_file)
        # sometimes video capture seems to give the wrong dimensions read the
        # first image and try again
        # get video frame, stop program when no frame is retrive (end of file)
        ret, image = vid.read()
        vid.release()

        if ret:
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.dtype = image.dtype
            self.vid = cv2.VideoCapture(video_file)
            self.video_file = video_file
            self._n_frames_read = 0
            self._is_eof = False
        else:
            raise OSError(
                'Cannot get an image from %s.\n It is likely that either the file name is wrong, the file is corrupt or OpenCV was not installed with ffmpeg support.' %
                video_file)

    def read(self):
        ret, image = self.vid.read()
        if ret:
            self._n_frames_read += 1
        else:
            self._is_eof = True
        return ret, image

    def __len__(self):
        if self._is_eof:
            return self._n_frames_read
        else:
            # warnings.warn(('Number of frames being estimated. '
            #                + 'Could be inaccurate, try again after reading '
            #                + 'the whole video'))
            return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        return self.vid.release()
