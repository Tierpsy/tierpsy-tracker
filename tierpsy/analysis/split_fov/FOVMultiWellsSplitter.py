#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:55:41 2019
@author: lferiani
"""


#%% import statements

import re
import cv2
import pdb
import tables
import itertools
import numpy as np
import pandas as pd
import scipy.optimize

from pathlib import Path
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.fftpack import next_fast_len
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors

from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.split_fov.helper import get_well_color
from tierpsy.analysis.split_fov.helper import naive_normalise
from tierpsy.analysis.split_fov.helper import WELLS_ATTRIBUTES
from tierpsy.analysis.split_fov.helper import make_square_template
from tierpsy.analysis.split_fov.helper import simulate_wells_lattice
from tierpsy.analysis.split_fov.helper import get_bgnd_from_masked
# from tierpsy.analysis.split_fov.helper import get_mwp_map, serial2channel
from tierpsy.helper.misc import TABLE_FILTERS


#%% Class definition
class FOVMultiWellsSplitter(object):
    """Class tasked with finding how to split a full-FOV image into
    single-wells images, and then splitting new images that are passed to it.
    """

    def __init__(
            self,
            fname_or_img,
            json_file=None,
            microns_per_pixel=None,
            well_shape=None,
            well_size_mm=None,
            well_masked_edge=0.1,
            camera_serial=None,
            rig=None,
            channel=None,
            wells_map=None,
            ):
        """
        Class constructor
        According to what the input is, will call different constructors
        Creates wells, and parses the image to fill up the wells property.
        You can use it in 3 ways:
        a. pass the name of a results file or masked videos containing
           /fov_wells - nothing else is needed.
        b. pass the name of a masked videos that does not have /fov_wells, and
           either the path to a splitfov json_file *or* all other params.
           If you pass it both json_file and other parameters, json_file will
           take precedence.
        c. pass a brightfield image,
           and all other parameters except the json_file.

        """

        # there can be 3 cases for input:
        # a. name of masked video with /fov_wells - nothing needed
        # b. name of masked video, no /fov_wells + json file (or all pars)
        # c. an image => all parameters needed

        if isinstance(fname_or_img, (str, Path)):
            # it is a string, or a path. cast to string for convenience
            fname_or_img = str(fname_or_img)
            # convert skel to feats as no fov_wells in skel
            if fname_or_img.endswith('_skeletons.hdf5'):
                fname_or_img.replace(
                    '_skeletons.hdf5', '_featuresN.hdf5')
            # this either features or masked.
            # have the wells been detected already?
            with pd.HDFStore(fname_or_img, 'r') as fid:
                has_fov_wells = '/fov_wells' in fid
            # if it has fov wells, easy - case a.
            if has_fov_wells:
                # construct from wells info
                self.constructor_from_fov_wells(
                    fname_or_img)
            else:
                # this should be case b - parse the file,
                # get all the parameters
                # then overwrite the filename with an image and do nothing else
                # first get image and conversion factor from masked video
                img, _, microns_per_pixel = get_bgnd_from_masked(
                    fname_or_img)
                # then parse json file for parameters
                sfparams = SplitFOVParams(json_file=json_file)
                well_shape, well_masked_edge, well_size_mm = (
                    sfparams.get_common_params())
                camera_serial, rig, channel, wells_map = (
                    sfparams.get_params_from_filename(fname_or_img))
                fname_or_img = img

        if isinstance(fname_or_img, np.ndarray):
            # now we're here either if we passed an image directly,
            # or if we parsed the filename + json file
            # in any case, we need all parameters now!
            assert all(
                    v is not None for k, v in locals().items()
                    if k != 'json_file')
            assert wells_map.ndim == 2, '`wells_map` should be 2D'
            # we can now store them so we don't have to pass them to the
            # image constructor
            self.microns_per_pixel = microns_per_pixel
            self.well_shape = well_shape
            self.well_size_mm = well_size_mm
            self.well_size_px = well_size_mm * 1000 / microns_per_pixel
            # self.well_masked_edge = well_masked_edge  # done later
            self.camera_serial = camera_serial
            self.rig = rig
            self.channel = channel
            self.wells_map = wells_map
            self.n_rows, self.n_cols = wells_map.shape
            self.n_wells_in_fov = self.wells_map.size
            self.is_dubious = None
            # and now the constructor
            self.constructor_from_image(fname_or_img)

        # this is common to the two constructors paths
        # assume all undefined wells are good
        self.wells['is_good_well'].fillna(1, inplace=True)
        self.well_masked_edge = well_masked_edge
        self.wells_mask = self.create_mask_wells()

    def constructor_from_image(self, img):

        # save the input image just to make some things easier
        if len(img.shape) == 2:
            self.img = img.copy()
        elif len(img.shape) == 3:
            # convert to grey
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # save height and width of image
        self.img_shape = self.img.shape

        # create a scaled down, blurry image which is faster to analyse
        self.blur_im = self.get_blur_im()
        # wells is the most important property.
        # It's the dataframe that contains the coordinates of each recognised
        # well in the original image
        # In particular
        #   x, y = well's centre coords, in px (x a column, y a row index)
        #   r    = radius of the circle, in pixel, if the wells are circular,
        #          or the circle inscribed into the wells template if squares
        #   row, col = indices of a well in the grid of detected wells
        #   *_max, *_min = coordinates for cropping the FOV so 1 roi = 1 well
        self.wells = pd.DataFrame(columns=WELLS_ATTRIBUTES)

        # METHODS
        # call method to fill in the wells variable
        # find_wells_on_grid is atm only implemented for square wells
        if self.well_shape == 'circle':
            self.find_circular_wells()
            self.remove_half_circles()
            self.find_row_col_wells()
            self.fill_lattice_defects()
            self.check_wells_grid_shape()
            self.find_wells_boundaries()
            self.calculate_wells_dimensions()
        elif self.well_shape == 'square':
            self.find_wells_on_grid()
            self.calculate_wells_dimensions()
            self.find_row_col_wells()
            self.check_wells_grid_shape()
        self.name_wells()

    def constructor_from_fov_wells(self, filename):
        print('constructor from /fov_wells')
        with tables.File(filename, 'r') as fid:
            wells_node_attrs = fid.get_node('/fov_wells')._v_attrs
            self.img_shape = wells_node_attrs['img_shape']
            self.camera_serial = wells_node_attrs['camera_serial']
            self.microns_per_pixel = wells_node_attrs['px2um']
            self.channel = wells_node_attrs['channel']
            self.well_shape = wells_node_attrs['well_shape']
            # new so might not be there
            if 'n_wells_in_fov' in wells_node_attrs:
                self.n_wells_in_fov = wells_node_attrs['n_wells_in_fov']
            # legacy:
            if 'whichsideup' in wells_node_attrs:
                self.whichsideup = wells_node_attrs['whichsideup']
            if 'n_wells' in wells_node_attrs:
                self.n_wells = wells_node_attrs['n_wells']
            # this only exists if well fitting algo unhappy
            if 'is_dubious' in wells_node_attrs:
                self.is_dubious = wells_node_attrs['is_dubious']
                if self.is_dubious:
                    print(f'Check {filename} for plate alignment')

        # is this a masked file or a features file? doesn't matter
        self.img = None
        masked_image_file = filename.replace('Results', 'MaskedVideos')
        masked_image_file = masked_image_file.replace(
            '_featuresN.hdf5', '.hdf5')
        if Path(masked_image_file).exists():
            with tables.File(masked_image_file, 'r') as fid:
                if '/bgnd' in fid:
                    self.img = fid.get_node('/bgnd').read(0)[0].copy()
                else:
                    # maybe bgnd was not in the masked video?
                    # for speed, let's just get the first full frame
                    self.img = fid.get_node('/full_data').read(0)[0].copy()

        # initialise the dataframe
        self.wells = pd.DataFrame(columns=WELLS_ATTRIBUTES)
        wells_table = pd.read_hdf(filename, '/fov_wells')
        for colname in wells_table.columns:
            self.wells[colname] = wells_table[colname]
        self.wells['x'] = 0.5 * (self.wells['x_min'] + self.wells['x_max'])
        self.wells['y'] = 0.5 * (self.wells['y_min'] + self.wells['y_max'])
        self.wells['r'] = self.wells['x_max'] - self.wells['x']
        for col in ['x', 'y', 'r']:
            self.wells[col] = self.wells[col].round().astype(int)

        self.calculate_wells_dimensions()
        self.find_row_col_wells()

    def write_fov_wells_to_file(self, filename, table_name='fov_wells'):
        table_path = '/'+table_name
        with tables.File(filename, 'r+') as fid:
            if table_path in fid:
                fid.remove_node(table_path)
            fid.create_table(
                '/',
                table_name,
                obj=self.get_wells_data().to_records(index=False),
                filters=TABLE_FILTERS)
            node_table = fid.get_node(table_path)
            node_table._v_attrs['img_shape'] = self.img_shape
            node_table._v_attrs['camera_serial'] = self.camera_serial
            node_table._v_attrs['px2um'] = self.microns_per_pixel
            node_table._v_attrs['channel'] = self.channel
            node_table._v_attrs['well_shape'] = self.well_shape
            if hasattr(self, 'n_wells_in_fov'):
                node_table._v_attrs['n_wells_in_fov'] = self.n_wells_in_fov
            if hasattr(self, 'whichsideup'):
                node_table._v_attrs['whichsideup'] = self.whichsideup
            if hasattr(self, 'n_wells'):
                node_table._v_attrs['n_wells'] = self.n_wells
            if hasattr(self, 'is_dubious'):
                node_table._v_attrs['is_dubious'] = self.is_dubious

    def get_blur_im(self):
        """downscale and blur the image"""
        # preprocess image
        dwnscl_factor = 4  # Hydra images' shape is divisible by 4
        blr_sigma = 17  # blur the image a bit, seems to work better
        new_shape = (self.img.shape[1]//dwnscl_factor,  # as x,y, not row, col
                     self.img.shape[0]//dwnscl_factor)

        try:
            dwn_gray_im = cv2.resize(self.img, new_shape)
        except:
            pdb.set_trace()
        # apply blurring
        blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma, blr_sigma), 0)
        # normalise between 0 and 255
        blur_im = cv2.normalize(
            blur_im, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return blur_im


    def find_wells_on_grid(self):
        """
        New method to find wells. Instead of trying to find all wells
        individually, minimise the abs diff between the real image and a
        mock image created by placing a template on a grid.
        Minimisation yields the lattice parameters and thus the position of
        the wells
        """

        # pad the image for better fft2 performance:
        # TODO: is this still needed?
        rowcol_padding = tuple(next_fast_len(size) - size
                               for size in self.blur_im.shape)
        rowcol_split_padding = tuple((pad//2, -(-pad//2)) # -(-x//2) == np.ceil(x/2)
                                     for pad in rowcol_padding)
        img = np.pad(self.blur_im, rowcol_split_padding, 'edge') # now padded
        img = 1 - naive_normalise(img) # normalised and inverted. This is a float
        meanimg = np.mean(img)
        npixels = img.size

        # define function to minimise

        # nwells = int(np.sqrt(self.n_wells_in_fov))
        # fun_to_minimise = lambda x: np.abs(
        #         img - simulate_wells_lattice(
        #                 img.shape,
        #                 x[0], x[1], x[2],
        #                 nwells=(self.n_rows, self.n_cols)
        #                 )
        #         ).sum()
        foo = simulate_wells_lattice(
                img.shape, 0.125, 0.125, 0.25,
                nwells=(self.n_rows, self.n_cols),
                template_shape=self.well_shape,
                )
        def fun_to_minimise(x):
            simulated_image = simulate_wells_lattice(
                img.shape, x[0], x[1], x[2],
                nwells=(self.n_rows, self.n_cols),
                template_shape=self.well_shape,
                )
            objective = np.abs(img - simulated_image).sum()
            return objective

        # actual minimisation
        # criterion for bounds choice:
        # 1/2n is if well starts at edge, 1/well if there is another half well!
        # bounds are relative to the size of the image (along the y axis)
        # 1/(nwells+0.5) spacing allows for 1/4 an extra well on both side
        # 1/(nwells-0.5) spacing allows for cut wells at the edges I guess
        # bounds = [(1/(2*nwells), 1/nwells),  # x_offset
        #           (1/(2*nwells), 1/nwells),  # y_offset
        #           (1/(nwells+0.5), 1/(nwells-0.5))]  # spacing
        guess_offset = 1/(2*self.n_rows)
        # guess_spacing = 1/self.n_rows
        # now that I pass the wells size, I know the spacing is
        # well size in px / img height in px
        # the spacing might actually not be something we need to minimise for
        guess_spacing = self.well_size_px / self.img_shape[0]
        bounds = [(0.75 * guess_offset, 1.25 * guess_offset),
                  (0.75 * guess_offset, 1.25 * guess_offset),
                  (0.95 * guess_spacing, 1.05 * guess_spacing)]
        result = scipy.optimize.differential_evolution(
            fun_to_minimise, bounds, polish=True, seed = 42)
        # extract output parameters for spacing grid
        x_offset, y_offset, spacing = result.x.copy()

        # convert from relative to pixels
        def _to_px(rel):
            return rel * self.img.shape[0]
        x_offset_px = _to_px(x_offset)
        y_offset_px = _to_px(y_offset)
        spacing_px = _to_px(spacing)
        # create list of centres and sizes
        # row and column could now come automatically as x and y are ordered
        # but since odd and even channel need to be treated diferently,
        # leave it to the specialised function
        xyr = np.array([
            (x, y, spacing_px/2)
            for x in np.arange(
                x_offset_px, self.img.shape[1], spacing_px)[:self.n_cols]
            for y in np.arange(
                y_offset_px, self.img.shape[0], spacing_px)[:self.n_rows]
            ])
        # write into dataframe
        self.wells = pd.DataFrame(columns=WELLS_ATTRIBUTES)
        self.wells['x'] = xyr[:, 0].astype(int)  # centre
        self.wells['y'] = xyr[:, 1].astype(int)  # centre
        self.wells['r'] = xyr[:, 2].astype(int)  # half-width
        # now calculate the rest. Don't need all the cleaning-up
        for d in ['x', 'y']:
            self.wells[d+'_min'] = self.wells[d] - self.wells['r']
            self.wells[d+'_max'] = self.wells[d] + self.wells['r']
        # bound wells to be in frame (to avoid pains later)
        # np.maximum does entrywise max(a,b)
        self.wells['x_min'] = np.maximum(self.wells['x_min'], 0)
        self.wells['y_min'] = np.maximum(self.wells['y_min'], 0)
        self.wells['x_max'] = np.minimum(
            self.wells['x_max'], self.img_shape[1])
        self.wells['y_max'] = np.minimum(
            self.wells['y_max'], self.img_shape[0])
        # and for debugging
        self._gridminres = (result, meanimg, npixels)  # save output of diff evo
        self._imgformin = img
        # looked at ~10k FOV splits, 0.6 is a good threshold to at least flag
        # FOV splits that result in too high residual
        self.is_dubious = (result.fun / meanimg / npixels > 0.6)


    def find_circular_wells(self):
        """Simply use Hough transform to find circles in MultiWell Plate
        rgb image.
        The parameters used are optimised for 24 or 48WP"""

        dwnscl_factor = self.img_shape[0]/self.blur_im.shape[0]

        # find circles
        # parameters in downscaled units
        circle_goodness = 70
        highest_canny_thresh = 10
        min_well_dist = .9 * (self.blur_im.shape[0] / self.n_rows)
        expected_radius = (self.well_size_px / 2) / dwnscl_factor
        min_well_radius = int(np.round(0.85 * expected_radius))
        max_well_radius = int(np.round(1.15 * expected_radius))
        # min_well_radius = self.blur_im.shape[1]//7; # if 48WP 3 wells on short side ==> radius <= side/6
        # max_well_radius = self.blur_im.shape[1]//4; # if 24WP 2 wells on short side. COnsidering intrawells space, radius <= side/4
        # find circles
        _circles = cv2.HoughCircles(
            self.blur_im,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_well_dist,
            param1=highest_canny_thresh,
            param2=circle_goodness,
            minRadius=min_well_radius,
            maxRadius=max_well_radius)
        _circles = np.squeeze(_circles)  # remove empty dimension

        # convert back to pixels
        _circles *= dwnscl_factor

        # output back into class property
        self.wells['x'] = _circles[:, 0].astype(int)
        self.wells['y'] = _circles[:, 1].astype(int)
        self.wells['r'] = _circles[:, 2].astype(int)
        # take the decision to use the median radius.
        # no reason for wells to have different radii
        # made the following change due to the fact that on recent versions of pandas, median() only returns a float and has no astype() attribute 
        # self.wells['r'] = self.wells['r'].median().astype(int)
        self.wells['r'] = int(self.wells['r'].median())
        return


    def find_row_col_wells(self):
        """
        The circular wells are aligned in a grid, but are not found in such
        order by the Hough Transform.
        Find (row, column) index for each well.
        Algorithm: (same for rows and columns)
            - Scan the found wells, pick up the topmost [leftmost] one
            - Find all other wells that are within the specified interval of the
                first along the considered dimension.
            - Assign the same row [column] label to all of them
            - Repeat for the wells that do not yet have an assigned label,
                increasing the label index
        """

        # number of pixels within which found wells are considered to be within the same row
        if self.well_shape == 'circle':
            interval = self.wells['r'].mean() # average radius across circles (3rd column)
        elif self.well_shape == 'square':
            interval = self.wells['r'].mean() # this is just half the template size anyway
            # maybe change that?

        # execute same loop for both rows and columns
        for d,lp in zip(['x','y'],['col','row']): # d = dimension, lp = lattice place
            # initialise array or row/column labels. This is a temporary variable, I could have just used self.wells[lp]
            d_ind = np.full(self.wells.shape[0],np.nan)
            cc = 0; # what label are we assigning right now
            # loop until all the labels have been assigned
            while any(np.isnan(d_ind)):
                # find coordinate of first (leftmost or topmost) non-labeled well
                idx_unlabelled_wells = np.isnan(d_ind)
                unlabelled_wells = self.wells.loc[idx_unlabelled_wells]
                coord_first_well = np.min(unlabelled_wells[d])
                # find distance between this and *all* wells along the considered dimension
                d_dists = self.wells[d] - coord_first_well;
                # find wells within interval. d_dists>=0 discards previous rows [columns]
                # could have taken the absolute value instead but meh I like this logic better
                idx_same = np.logical_and((d_dists >= 0),(d_dists < interval))
                # doublecheck we are not overwriting an existing label:
                # idx_same should point to positions that are still nan in d_ind
                if any(np.isnan(d_ind[idx_same])==False):
                    pdb.set_trace()
                elif not any(idx_same): # if no wells found within the interval
                    pdb.set_trace()
                else:
                    # assign the row [col] label to the wells closer than
                    # interval to the topmost [leftmost] unlabelled well
                    d_ind[idx_same] = cc
                # increment label
                cc+=1
            # end while
            # assign label array to right dimension
            self.wells[lp] = d_ind.astype(int)

        return

    def check_wells_grid_shape(self):
        # check results (unless this is from an old masked videos fov_wells)
        # in which case it's fine
        if hasattr(self, 'n_wells_in_fov'):
            if not (self.wells.shape[0] == self.n_wells_in_fov):
                self.plot_wells()
                raise Exception("Found wells do not match expected arrangement")
        if hasattr(self, 'n_rows'):
            if not all([
                    self.wells['row'].max() == (self.n_rows - 1),
                    self.wells['col'].max() == (self.n_cols - 1)]):
                self.plot_wells()
                raise Exception("Found wells do not match expected arrangement")

        return


    def remove_half_circles(self, max_radius_portion_missing=0.7):
        """
        Only keep circles whose centre is at least
        (1-max_radius_portion_missing)*radius away
        from the edge of the image
        """

        assert self.well_shape == 'circle', "This method is only to be used with circular wells"

        # average radius across circles (3rd column)
        avg_radius = self.wells['r'].mean()
        # keep only circles missing less than 0.5 radius
        extra_space = avg_radius*(1-max_radius_portion_missing);
        # bad circles = centre of circles is not too close to image edge
        idx_bad_circles =   (self.wells['x'] - extra_space < 0) | \
                            (self.wells['x'] + extra_space >= self.img_shape[1]) | \
                            (self.wells['y'] - extra_space < 0) | \
                            (self.wells['y'] + extra_space >= self.img_shape[0])

        # remove entries that did not satisfy the initial requests
        self.wells.drop(self.wells[idx_bad_circles].index, inplace=True)
        return


    def find_wells_boundaries(self):
        """
        Find lines along which to crop the FOV.
        Lines separating rows/columns are halfway between the grouped medians of
        the relevant coordinate.
        Lines before the first and after the last row/column are the median
        coordinate +- 0.5 the median lattice spacing.
        """
        # loop on dimension (and lattice place). di = dimension counter
        # di is needed to index on self.img_shape
        for di,(d,lp) in enumerate(zip(['x','y'],['col','row'])):
            # only look at correct column of dataframe. temporary variables for shortening purposes
            labels = self.wells[lp]
            coords = self.wells[d]
            # average distance between rows [cols]
            avg_lattice_spacing = np.diff(coords.groupby(labels).mean()).mean()
            max_ind = np.max(labels) # max label of rows [columns]
            # initialise array that will hold info re where to put lines
            # N lines = N rows + 1 = max row + 2 b.c. 0 indexing
            lines_coords = np.zeros(max_ind+2)
            # take care of lfirst and last edge
            lines_coords[0] = np.median(coords[labels==0]) - avg_lattice_spacing/2
            lines_coords[0] = max(lines_coords[0], 0); # line has to be within image bounds
            lines_coords[-1] = np.median(coords[labels==max_ind]) + avg_lattice_spacing/2
            lines_coords[-1] = min(lines_coords[-1], self.img_shape[1-di]); # line has to be within image bounds
            # for each row [col] find the middle point with the next one,
            # write it into the lines_coord variable
            for ii in range(max_ind):
                jj = ii+1; # index on lines_coords
                lines_coords[jj] = np.median(np.array([
                        np.mean(coords[labels==ii]),
                        np.mean(coords[labels==ii+1])]));
            # store into self.wells for return
            self.wells[d+'_min'] = lines_coords.copy().astype(np.int)[labels]
            self.wells[d+'_max'] = lines_coords.copy().astype(np.int)[labels+1]

        return


    def fill_lattice_defects(self):
        """
        If a grid of wells was detected but there are entries missing, try to
        return a guesstimate of where the missing well(s) may be.
        """
        # find, in theory, how many wells does the detected grid allow for
        n_rows = self.wells['row'].max()+1
        n_cols = self.wells['col'].max()+1
        n_expected_wells = n_rows * n_cols;
        n_detected_wells = len(self.wells)

        if n_detected_wells == n_expected_wells:
            # nothing to do here
            return
        elif n_detected_wells > n_expected_wells:
            # uncropped image? other errors?
            import pdb
            pdb.set_trace()
            raise Exception("Found more wells than expected. Aborting now.")
        # I only get here if n_detected_wells < n_expected_wells
        assert n_detected_wells < n_expected_wells, \
            "Something wrong in the logic in fill_lattice_defects()"
        # some wells got missed. Using the lattice structure to find them
        expected_rowcols = set(itertools.product(range(n_rows), range(n_cols)))
        detected_rowcols = set((rr,cc) for rr,cc in self.wells[['row','col']].values)
        missing_rowcols = list(expected_rowcols - detected_rowcols)
        # now add the missing rowcols combinations
        for rr,cc in missing_rowcols:
            new_well = {}
            # calculate x,y,r
            y = int(self.wells[self.wells['row'] == rr]['y'].median())
            x = int(self.wells[self.wells['col'] == cc]['x'].median())
            r = int(self.wells['r'].mean())
            # append to temporary dict
            new_well['x'] = [x,]
            new_well['y'] = [y,]
            new_well['r'] = [r,]
            new_well['row'] = [rr,]
            new_well['col'] = [cc,]
            new_df = pd.DataFrame(new_well)
#            print(new_df)
            self.wells = pd.concat([self.wells, new_df], ignore_index=True, sort=False)
        return


    def calculate_wells_dimensions(self):
        """
        Finds width, height of each well
        """
        self.wells['width'] = self.wells['x_max']-self.wells['x_min']
        self.wells['height'] = self.wells['y_max']-self.wells['y_min']
        return


    def name_wells(self):
        """
        Assign name to the detected wells.
        Need to know what channel, how many wells in total, if mwp was upright,
        and in the future where was A1 or if the video with A1 has got 6 or 9 wells
        """

        max_row = self.wells['row'].max()
        max_col = self.wells['col'].max()

        # provided that the user-defined well maps were correct,
        # it is just a matter of indexing in the 2d array
        self.wells['well_name'] = self.wells_map[
            self.wells['row'], self.wells['col']]

        return


    def tile_FOV(self, img_or_stack):
        """
        Function that tiles the input image or stack and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function
        for ROI making, could be a lot quicker
        """
        if len(img_or_stack.shape) == 2:
            return self.tile_FOV_2D(img_or_stack)
        elif len(img_or_stack.shape) == 3:
            return self.tile_FOV_3D(img_or_stack)
        else:
            raise Exception("Can only tile 2D or 3D objects")
            return


    def tile_FOV_2D(self, img):
        """
        Function that chops an image according to the x/y_min/max coordinates in
        wells, and returns a list of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            xmin = max(well['x_min'], 0)
            ymin = max(well['y_min'], 0)
            xmax = min(well['x_max'], self.img_shape[1])
            ymax = min(well['y_max'], self.img_shape[0])
            roi_img = img[ymin:ymax, xmin:xmax]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def tile_FOV_3D(self, img):
        """
        Function that chops an image stack (1st dimension is n_frames)
        according to the x/y_min/max coordinates in
        wells, and returns a list of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            xmin = max(well['x_min'], 0)
            ymin = max(well['y_min'], 0)
            xmax = min(well['x_max'], self.img_shape[1])
            ymax = min(well['y_max'], self.img_shape[0])
            roi_img = img[:, ymin:ymax, xmin:xmax]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def plot_wells(self, is_rotate180=False, ax=None, line_thickness=20):
        """
        Plot the fitted wells, the wells separation, and the name of the well.
        (only if these things are present!)"""

        # make sure I'm not working on the original image
        if is_rotate180:
            # a rotation is 2 reflections
            _img = cv2.cvtColor(self.img.copy()[::-1, ::-1],
                                cv2.COLOR_GRAY2BGR)
            _wells = self.wells.copy()
            for c in ['x_min', 'x_max', 'x']:
                _wells[c] = _img.shape[1] - _wells[c]
            for c in ['y_min', 'y_max', 'y']:
                _wells[c] = _img.shape[0] - _wells[c]
            _wells.rename(columns={'x_min':'x_max',
                                   'x_max':'x_min',
                                   'y_min':'y_max',
                                   'y_max':'y_min'},
                          inplace=True)
        else:
            _img = cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
            _wells = self.wells.copy()

#        pdb.set_trace()
        # flags: according to dataframe state, do or do not do
        _is_wells = _wells.shape[0] > 0;
        _is_rois = np.logical_not(_wells['x_min'].isnull()).all() and _is_wells;
        _is_wellnames = np.logical_not(_wells['well_name'].isnull()).all() and _is_rois;
        # TODO: deal with grayscale image
        # burn the circles into the rgb image
        if _is_wells and self.well_shape == 'circle':
            for i, _circle in _wells.iterrows():
                # draw the outer circle
                cv2.circle(_img,(_circle.x,_circle.y),_circle.r,(255,0,0),5)
                # draw the center of the circle
                cv2.circle(_img,(_circle.x,_circle.y),5,(0,255,255),5)
        # burn the boxes edges into the RGB image
        if _is_rois:
            #normalize item number values to colormap
            # normcol = colors.Normalize(vmin=0, vmax=self.wells.shape[0])
#            print(self.wells.shape[0])
            for i, _well in _wells.iterrows():
                color = get_well_color(_well.is_good_well,
                                       forCV=True)
                cv2.rectangle(_img,
                              (_well.x_min, _well.y_min),
                              (_well.x_max, _well.y_max),
#                              colors[0], 20)
                               color, line_thickness)

        # add names of wells
        # plot, don't close
        if not ax:
            figsize = (8, 8*_img.shape[0]/_img.shape[1])
            fig = plt.figure(figsize=figsize)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig = ax.figure
            ax.set_axis_off()

        ax.imshow(_img)
        if _is_wellnames:
            for i, _well in _wells.iterrows():
                try:
                    txt = "{} ({:d},{:d})".format(_well['well_name'],
                           int(_well['row']),
                           int(_well['col']))
                except:  # could not have row, col if from /fov_wells
                    txt = "{}".format(_well['well_name'])
                ax.text(_well['x_min']+_well['width']*0.05,
                        _well['y_min']+_well['height']*0.12,
                        txt,
                        fontsize=10,
                        color=np.array(get_well_color(_well['is_good_well'],
                                                       forCV=False))
                        )
                         # color='r')
        elif _is_rois:
            for i, _well in _wells.iterrows():
                ax.text(_well['x'], _well['y'],
                        "({:d},{:d})".format(int(_well['row']),
                                             int(_well['col'])),
                        fontsize=12,
                        weight='bold',
                        color='r')
#        plt.axis('off')
        # plt.tight_layout()
        return fig


    def find_well_of_xy(self, x, y):
        """
        Takes two numpy arrays (or pandas columns), returns an array of strings
        of the same with the name of the well each x,y, pair falls into
        """
        # I think a quick way is by using implicit expansion
        # treat the x array as column, and the *_min and *_max as rows
        # these are all matrices len(x)-by-len(self.wells)
        # none creates new axis
        x = np.asarray(x)
        y = np.asarray(y)
        # making sure all are numpy arrays
        if np.isscalar(x):
            x = np.array([x])
            y = np.array([y])

        x_min = self.wells['x_min'].to_numpy()
        x_max = self.wells['x_max'].to_numpy()
        y_min = self.wells['y_min'].to_numpy()
        y_max = self.wells['y_max'].to_numpy()

        within_x = np.logical_and(
            (x[:, None] - x_min[None, :]) >= 0,
            (x[:, None] - x_max[None, :]) <= 0)
        within_y = np.logical_and(
            (y[:, None] - y_min[None, :]) >= 0,
            (y[:, None] - y_max[None, :]) <= 0)
        within_well = np.logical_and(within_x, within_y)
        # in each row of within_well, the column index of the "true" value is the well index

        # sanity check:
        assert (within_well.sum(axis=1)>1).any() == False, \
        "a coordinate is being assigned to more than one well?"
        # now find index
        ind_worms_in_wells, ind_well = np.nonzero(within_well)

        # prepare the output panda series (as long as the input variable)
        well_names = pd.Series(data=['n/a']*len(x), dtype='S3', name='well_name')
        # and assign the well name (read using the ind_well variable from the self.well)
        well_names.loc[ind_worms_in_wells] = self.wells.iloc[ind_well]['well_name'].astype('S3').values

        return well_names


    def find_well_from_trajectories_data(self, trajectories_data):
        """Wrapper for find_well_of_xy,
        reads the coordinates from the right columns of trajectories_data"""
        return self.find_well_of_xy(trajectories_data['coord_x'],
                                    trajectories_data['coord_y'])


    def get_wells_data(self):
        """
        Returns info about the wells for storage purposes, in hdf5 friendly format
        """
        wells_out = self.wells[['x_min','x_max','y_min','y_max']].copy()
        wells_out['well_name'] = self.wells['well_name'].astype('S3')
#        import pdb;pdb.set_trace()
        is_good_well = self.wells['is_good_well'].copy()
        wells_out['is_good_well'] = is_good_well.fillna(-1).astype(int)

        return wells_out


    def create_mask_wells(self):
        """
        create_mask_wells create a black mask covering the space between wells
        and/or a user-defined region at the edges of each well. Can be used
        on square and circular wells

        Returns
        -------
        mask [np.ndarray]
            mask of 0 and 1s, to multiply or bitwise_and to an image to
            delete unwanted regions
        """

        assert self.well_masked_edge < 0.5, \
            "well_masked_edge has to be less than 50% or no data left"

        if self.well_shape == 'square':
            mask = self._create_mask_wells_square()
        elif self.well_shape == 'circle':
            mask = self._create_mask_wells_circle()
        else:
            raise Exception('call to create_mask_wells without a well shape')
        return mask


    def _create_mask_wells_circle(self):
        """
        create_mask_wells_circle Create a black mask covering the space between
        round wells, and however much inside the detected circular shape
        as defined by well_masked_edge
        """
        # start with empty mask, fill it up with circles as found by the algo.
        mask = np.zeros(self.img_shape).astype(np.uint8)
        draw_radius = self.wells['r'].median() * (1 - self.well_masked_edge)
        draw_radius = int(draw_radius)
        centres = self.wells[['x', 'y']].values
        for point in centres:
            # draw the outer circle
            cv2.circle(mask,tuple(point), draw_radius, 255, -1)
        # bring down to [0, 1] range for multiplying later
        mask = mask // 255
        return mask



    def _create_mask_wells_square(self):
        """
        Create a black mask covering a thick edge of the square region covering
        each well. Thickness specified by well_masked_edge
        """


        mask = np.ones(self.img_shape).astype(np.uint8)
        # average size of wells
        mean_wells_width = np.round(np.mean(self.wells['x_max']-self.wells['x_min']))
        mean_wells_height = np.round(np.mean(self.wells['y_max']-self.wells['y_min']))
        # size of black edge
        horz_edge = np.round(mean_wells_width * self.well_masked_edge).astype(int)
        vert_edge = np.round(mean_wells_height * self.well_masked_edge).astype(int)
        for x in np.unique(self.wells[['x_min','x_max']]):
            m = max(x-horz_edge,0)
            M = min(x+horz_edge, self.img_shape[1])
            mask[:,m:M] = 0
        for y in np.unique(self.wells[['y_min','y_max']]):
            m = max(y-vert_edge,0)
            M = min(y+vert_edge, self.img_shape[0])
            mask[m:M,:] = 0
        # mask everything outside the wells
        M = self.wells['x_max'].max()
        mask[:, M:] = 0
        m = self.wells['x_min'].min()
        mask[:, :m] = 0
        M = self.wells['y_max'].max()
        mask[M:, :] = 0
        m = self.wells['y_min'].min()
        mask[:m, :] = 0

        return mask


    def apply_wells_mask(self, img):
        """
        performs img*= mask"""
        img *= self.wells_mask



# end of class

def process_image_from_name(image_name, json_fname, is_plot=True, is_save=True):
    # read image
    fname = str(image_name)
    img_ = cv2.imread(fname)
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    # parse pars
    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()
    uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(fname)
    px2um = 12.4

    # run fov splitting
    fovsplitter = FOVMultiWellsSplitter(
        img,
        camera_serial=uid,
        wells_map=mwp_map,
        well_size_mm=sz_mm,
        well_shape=shape,
        well_masked_edge=edge_frac,
        rig=rig,
        channel=ch,
        microns_per_pixel=px2um)

    if is_plot:
        fig = fovsplitter.plot_wells()
        if is_save:
            fig.savefig(fname.replace('.png','_wells_new.png'))

    return fovsplitter

#%% main

if __name__ == '__main__':

    import os
    import time
    import re
    import tqdm
    from pathlib import Path

    from tierpsy import DFLT_SPLITFOV_PARAMS_PATH, DFLT_SPLITFOV_PARAMS_FILES
    from tierpsy.helper.params.tracker_param import SplitFOVParams
    from tierpsy.analysis.compress.selectVideoReader import selectVideoReader

    # plt.close("all")

    # %%
    # test from raw/frame (like in COMPRESS)

    # where are things
    # wd = Path('~/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/').expanduser()
    # raw_fname = (
    #     wd / 'RawVideos' / '_20191205' /
    #     'syngenta_screen_run1_bluelight_20191205_151104.22956805' /
    #     'metadata.yaml'
    #     )
    wd = Path(
        '/Volumes/behavgenom$/Tom/Data/Hydra/Ivermectin_food_test/RawData'
        ).expanduser()
    raw_fnames = list((wd / 'RawVideos').rglob('metadata.yaml'))
    masked_fnames = [(
        str(f).replace('RawVideos',  'MaskedVideos')
        .replace('.yaml',  '.hdf5')) for f in raw_fnames]
    masked_fnames = [Path(f) for f in masked_fnames]

    (wd / 'MaskedVideos').mkdir(parents=True, exist_ok=True)

    # common parameters
    json_fname = Path(DFLT_SPLITFOV_PARAMS_PATH) / 'HYDRA_96WP_ROUND_UPRIGHT.json'
    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()
    px2um = 12.4
    # %%
    for raw_fname, masked_fname in tqdm.tqdm(zip(raw_fnames, masked_fnames)):

        masked_fname.parent.mkdir(exist_ok=True, parents=True)
        if masked_fname.exists():
            masked_fname.unlink()


        uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(
            masked_fname)

        # read image
        vid = selectVideoReader(str(raw_fname))
        status, img = vid.read_frame(0)

        fovsplitter = FOVMultiWellsSplitter(
            img,
            microns_per_pixel=px2um,
            well_shape=shape,
            well_size_mm=sz_mm,
            well_masked_edge=edge_frac,
            camera_serial=uid,
            rig=rig,
            channel=ch,
            wells_map=mwp_map)

        fig = fovsplitter.plot_wells()
        fig.savefig(wd / (raw_fname.parent.name + '_default.png'))
        plt.close(fig)

        wells_df = fovsplitter.get_wells_data()

        with open(masked_fname, 'w') as fid:
            pass
        fovsplitter.write_fov_wells_to_file(masked_fname)

        # fovsplitter.apply_wells_mask(img)
        # fig, ax = plt.subplots(figsize=(8, 8*img.shape[0]/img.shape[1]))
        # ax.set_axis_off()
        # ax.imshow(img)


    # %%
    # test from masked video with new /fov_wells
    # when building from wells, no need for json
    masked_fname = Path(
        '/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/'
        'MaskedVideos/20191205/'
        'syngenta_screen_run1_bluelight_20191205_151104.22956805/metadata.hdf5'
        )

    fs_from_wells = FOVMultiWellsSplitter(masked_fname)

    # %%
    # test from masked video with old /fov_wells
    # when building from wells, no need for json
    masked_fname = Path(
        '/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/'
        '_MaskedVideos/20191205/'
        'syngenta_screen_run1_bluelight_20191205_151104.22956805/metadata.hdf5'
        )

    fs_from_old_wells = FOVMultiWellsSplitter(masked_fname)
    # %%
    # test from many images:
    wd = Path('/Volumes/behavgenom$/Luigi/Data/LoopBio_calibrations/wells_mapping/20190710/')
    img_dir = wd
    fnames = list(img_dir.rglob('*.png'))
    fnames = [str(f) for f in fnames if '_wells' not in str(f)]

    plt.ioff()
    for fname in tqdm.tqdm(fnames):
        process_image_from_name(fname, json_fname)
    plt.ion()