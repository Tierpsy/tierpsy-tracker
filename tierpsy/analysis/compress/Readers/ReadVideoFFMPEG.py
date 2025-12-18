# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:15:59 2015

@author: ajaver
"""

import os
import re
import warnings
import subprocess as sp

import numpy as np

from tierpsy.helper.misc import FFMPEG_CMD, ReadEnqueue


class ReadVideoFFMPEG:
    '''
    Read video frame using ffmpeg. Assumes 8bits gray video.
    Requires that ffmpeg is installed in the computer.
    This class is an alternative of the captureframe of opencv since:
    -> it can be a pain to compile opencv with ffmpeg compatibility.
    -> this function is a bit faster (less overhead), but only works with gecko's mjpeg
    '''

    def __init__(self, fileName):

        if not os.path.exists(fileName):
            raise FileNotFoundError(fileName)

        if not FFMPEG_CMD:
            raise FileNotFoundError('ffmpeg do not found. Cannot process this video.')

        # try to open the file and determine the frame size. Raise an exception
        # otherwise.
        # command = [FFMPEG_CMD, '-i', fileName, '-']  # this works for sure
        command = [FFMPEG_CMD, '-i', fileName, '-map', '0:v:0', '-c', 'copy',
                   '-f', 'null']
        # different null for windows/posix
        if os.name == 'nt':
            command.append('NULL')
        else:
            command.append('/dev/null')
        proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
        buff = proc.stderr.read()
        proc.terminate()

        try:
            # the frame size is somewhere printed at the beginning by ffmpeg
            dd = str(buff).partition('Video: ')[2].split(',')[2]
            dd = re.findall(r'\d*x\d*', dd)[0].split('x')
            self.height = int(dd[1])
            self.width = int(dd[0])
            self.dtype = np.uint8

        except (IndexError, ValueError):
            raise OSError(('Error while getting the width and height using ffmpeg. Buffer output:', buff))

        self.tot_pix = self.height * self.width

        try:
            nn = re.findall(r'(?<=frame=)\ *\d*', str(buff))[0].strip()
            self._tot_frames_estimate = int(nn)
        except (IndexError, ValueError):
            raise OSError(('Error while getting the number of frames using '
                           + 'ffmpeg. Buffer output:'
                           + buff))

        command = [FFMPEG_CMD,
                   '-i', fileName,
                   '-f', 'image2pipe',
                   '-vsync', 'drop',  # avoid repeating frames due to changes in the time stamp, it is better to solve those situations manually after
                   '-threads', '0',
                   '-vf', 'showinfo',
                   '-pix_fmt', 'gray',
                   '-vcodec', 'rawvideo', '-']

        self.vid_frame_pos = []
        self.vid_time_pos = []

        # devnull = open(os.devnull, 'w') #use devnull to avoid printing the
        # ffmpeg command output in the screen
        self.proc = sp.Popen(command, stdout=sp.PIPE,
                             bufsize=self.tot_pix, stderr=sp.PIPE)

        self.buf_reader = ReadEnqueue(self.proc.stderr)
        # for number of frames
        self._n_frames_read = 0
        self._is_eof = False

        # use a buffer size as small as possible (frame size), makes things
        # faster

    def get_timestamp(self):
        while True:
            # read line without blocking
            line = self.buf_reader.read()
            if line is None:
                break
            # self.err_out.append(line)

            frame_N = line.partition(' n:')[-1].partition(' ')[0]
            timestamp = line.partition(' pts_time:')[-1].partition(' ')[0]

            if frame_N and timestamp:
                self.vid_frame_pos.append(int(frame_N))
                self.vid_time_pos.append(float(timestamp))

    def read(self):
        # retrieve an image as numpy array
        raw_image = self.proc.stdout.read(self.tot_pix)
        if len(raw_image) < self.tot_pix:
            self._is_eof = True
            return (0, [])

        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape(self.height, self.width)

        # i need to read this here because otherwise the err buff will get
        # full.
        self.get_timestamp()

        # increase read frames counter
        self._n_frames_read += 1

        return (1, image)

    def __len__(self):
        if not self._is_eof:
            # warnings.warn('Number of frames being estimated. '
            #                + 'Could be inaccurate, try again after reading '
            #                + 'the whole video')
            return self._tot_frames_estimate
        else:
            return self._n_frames_read

    def release(self):
        # close the buffer
        self.proc.stdout.flush()
        self.proc.stderr.flush()
        self.get_timestamp()

        self.proc.terminate()
        self.proc.stdout.close()
        self.proc.wait()
