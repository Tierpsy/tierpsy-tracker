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

from tierpsy.analysis.split_fov.helper import get_well_color
from tierpsy.analysis.split_fov.helper import naive_normalise
from tierpsy.analysis.split_fov.helper import WELLS_ATTRIBUTES
from tierpsy.analysis.split_fov.helper import make_square_template
from tierpsy.analysis.split_fov.helper import simulate_wells_lattice
from tierpsy.analysis.split_fov.helper import calculate_bgnd_from_masked_fulldata
from tierpsy.analysis.split_fov.helper import get_mwp_map, serial2channel
from tierpsy.helper.misc import TABLE_FILTERS


#%% Class definition
class FOVMultiWellsSplitter(object):
    """Class tasked with finding how to split a full-FOV image into single-wells images,
    and then splitting new images that are passed to it."""

    def __init__(self, masked_or_features_or_image, **kwargs):
        """
        Class constructor
        According to what the input is, will call different constructors
        Creates wells, and parses the image to fill up the wells property
        Either give the masked_image_file name or features file,
        or pass an image AND camera serial number.
        If the masked_image_file name is given, any img, camera_serial, or px2um
        will be ignored and read from the masked_image_file
        img = a brightfield frame that will be used for well-finding
        n_wells = how many wells *in the entire multiwell plate*
        """

        # remove kwargs variables that don't need to propagate further
        well_masked_edge = kwargs.pop('well_masked_edge', 0.1)

        # is it a string?
        if not isinstance(masked_or_features_or_image, (str, Path)):
            # assume it is an image
            self.constructor_from_image(masked_or_features_or_image,
                                        **kwargs)

        else:
            # it is a string, or a path. cast to string for convenience
            masked_or_features_or_image = str(masked_or_features_or_image)
            is_skeletons = '_skeletons.hdf5' in masked_or_features_or_image
            is_featuresN = '_featuresN.hdf5' in masked_or_features_or_image
            is_masked = (is_skeletons == False) and (is_featuresN == False) and \
                ('_features.hdf5' not in masked_or_features_or_image) and \
                ('.hdf5' in masked_or_features_or_image)

            if is_skeletons:
                # this is a skeletons file
                raise ValueError("only works with masked videos or featuresN files")
            if is_featuresN or is_masked:
                # this either features or masked.
                # have the wells been detected already?
                with pd.HDFStore(masked_or_features_or_image,'r') as fid:
                    has_fov_wells = '/fov_wells' in fid

                if has_fov_wells:
                    # construct from wells info
                    self.constructor_from_fov_wells(masked_or_features_or_image)
                else:
                    # fall back on constructing from masked
                    masked_image_file = masked_or_features_or_image.replace('_featuresN.hdf5','.hdf5')
                    img, camera_serial, px2um = calculate_bgnd_from_masked_fulldata(masked_image_file)
#                    print(img, camera_serial, px2um)
                    self.constructor_from_image(img,
                                                camera_serial=camera_serial,
                                                px2um=px2um,
                                                **kwargs)

        # this is common to the two constructors paths
        # assume all undefined wells are good
        self.wells['is_good_well'].fillna(1, inplace=True)
        self.well_masked_edge = well_masked_edge
        self.wells_mask = self.create_mask_wells()


    def constructor_from_image(self,
                               img,
                               camera_serial=None,
                               px2um=None,
                               total_n_wells=96,
                               whichsideup='upright',
                               well_shape='square',
                               **kwargs):
        # kwargs is there so i
        print('constructor from image')
#        print(camera_serial, px2um)
        # very needed inputs
        if (camera_serial is None) or (px2um is None):
            raise ValueError('Either provide the masked video filename or an' +\
                             ' image, camera_serial, and px2um.')

        # save the input image just to make some things easier
#        print('image shape: {}'.format(img.shape))
        if len(img.shape) == 2:
            self.img = img.copy()
        elif len(img.shape) == 3:
            # convert to grey
            self.img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # and store serial number and px2um
        self.camera_serial = camera_serial
        self.px2um = px2um         # px2um = 12.4 is default for Hydra rig

        # assert that other input was given correctly
        # whichsideup: was the cell upright or upside-down
        assert whichsideup in ['upside-down', 'upright'], \
            "Whichsideup can only be 'upside-down' or 'upright'"
        # well_shape: either 'square' or 'circle'
        assert well_shape in ['square', 'circle'], \
            "well_shape can only be 'square' or 'circle'"

        # save height and width of image
        self.img_shape = self.img.shape
        # where was the camera on the rig?
        self.channel = serial2channel(self.camera_serial)
        # number of wells in the multiwell plate: 6 12 24 48 96?
        # TODO: input check. Dunno if it will be kept like this or parsed from a filename
        self.n_wells = total_n_wells
        # whichsideup: was the cell upright or upside-down
        self.whichsideup = whichsideup
        # well_shape: either 'square' or 'circle'
        self.well_shape = well_shape
        # according to n_wells and whichsideup choose a dictionary for
        #(channel,position) <==> well
        # TODO?: make user specify (if 48wp) which channels have 6 wells and not 9
        self.mwp_df = get_mwp_map(self.n_wells, self.whichsideup)
        self.wellsmap_in_fov_df = self.mwp_df[self.channel]
        self.n_wells_in_fov = self.wellsmap_in_fov_df.size
        # create a scaled down, blurry image which is faster to analyse
        self.blur_im = self.get_blur_im()
        # wells is the most important property.
        # It's the dataframe that contains the coordinates of each recognised
        # well in the original image
        # In particular
        #   x, y         = coordinates of the well's centre, in pixel (so x is a column index, y a row index)
        #   r            = radius of the circle, in pixel, if the wells are circular,
        #                   or the circle inscribed into the wells template if squares
        #   row, col     = indices of a well in the grid of detected wells
        #   *_max, *_min = coordinates for cropping the FOV so 1 roi = 1 well
        self.wells = pd.DataFrame(columns = WELLS_ATTRIBUTES)

        # METHODS
        # call method to fill in the wells variable
        # find_wells_on_grid is atm only implemented for square wells
        if self.well_shape == 'circle':
            self.find_circular_wells()
            self.remove_half_circles()
            self.find_row_col_wells()
            self.fill_lattice_defects()
            self.find_wells_boundaries()
            self.calculate_wells_dimensions()
        elif self.well_shape == 'square':
            self.find_wells_on_grid()
            self.calculate_wells_dimensions()
            self.find_row_col_wells()
        self.name_wells()

    def constructor_from_fov_wells(self, filename):
        print('constructor from /fov_wells')
        with tables.File(filename, 'r') as fid:
            self.img_shape     = fid.get_node('/fov_wells')._v_attrs['img_shape']
            self.camera_serial = fid.get_node('/fov_wells')._v_attrs['camera_serial']
            self.px2um         = fid.get_node('/fov_wells')._v_attrs['px2um']
            self.channel       = fid.get_node('/fov_wells')._v_attrs['channel']
            self.n_wells       = fid.get_node('/fov_wells')._v_attrs['n_wells']
            self.whichsideup   = fid.get_node('/fov_wells')._v_attrs['whichsideup']
            self.well_shape    = fid.get_node('/fov_wells')._v_attrs['well_shape']

        # is this a masked file or a features file? doesn't matter
        self.img = None
        masked_image_file = filename.replace('Results','MaskedVideos')
        masked_image_file = masked_image_file.replace('_featuresN.hdf5',
                                                      '.hdf5')
        if Path(masked_image_file).exists():
            with tables.File(masked_image_file, 'r') as fid:
                if '/bgnd' in fid:
                    self.img = fid.get_node('/bgnd')[0]
                else:
                    # maybe bgnd was not in the masked video?
                    # for speed, let's just get the first full frame
                    self.img = fid.get_node('/full_data')[0]

        # initialise the dataframe
        self.wells = pd.DataFrame(columns = WELLS_ATTRIBUTES)
        with pd.HDFStore(filename,'r') as fid:
            wells_table = fid['/fov_wells']
        for colname in wells_table.columns:
            self.wells[colname] = wells_table[colname]
        self.wells['x'] = 0.5 * (self.wells['x_min'] + self.wells['x_max'])
        self.wells['y'] = 0.5 * (self.wells['y_min'] + self.wells['y_max'])
        self.wells['r'] = self.wells['x_max'] - self.wells['x']

        self.calculate_wells_dimensions()
        self.find_row_col_wells()



    def write_fov_wells_to_file(self, filename):
        with tables.File(filename, 'r+') as fid:
            if '/fov_wells' in fid:
                fid.remove_node('/fov_wells')
            fid.create_table('/',
                             'fov_wells',
                             obj = self.get_wells_data().to_records(index=False),
                             filters = TABLE_FILTERS)
            fid.get_node('/fov_wells')._v_attrs['img_shape']     = self.img_shape
            fid.get_node('/fov_wells')._v_attrs['camera_serial'] = self.camera_serial
            fid.get_node('/fov_wells')._v_attrs['px2um']         = self.px2um
            fid.get_node('/fov_wells')._v_attrs['channel']       = self.channel
            fid.get_node('/fov_wells')._v_attrs['n_wells']       = self.n_wells
            fid.get_node('/fov_wells')._v_attrs['whichsideup']   = self.whichsideup
            fid.get_node('/fov_wells')._v_attrs['well_shape']    = self.well_shape



    def get_blur_im(self):
        """downscale and blur the image"""
        # preprocess image
        dwnscl_factor = 4; # Hydra images' shape is divisible by 4
        blr_sigma = 17; # blur the image a bit, seems to work better
        new_shape = (self.img.shape[1]//dwnscl_factor, # as x,y, not row,columns
                     self.img.shape[0]//dwnscl_factor)

        try:
            dwn_gray_im = cv2.resize(self.img, new_shape)
        except:
            pdb.set_trace()
        # apply blurring
        blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma,blr_sigma),0)
        # normalise between 0 and 255
        blur_im = cv2.normalize(blur_im, None, alpha=0, beta=255,
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
        rowcol_padding = tuple(next_fast_len(size) - size
                               for size in self.blur_im.shape)
        rowcol_split_padding = tuple((pad//2, -(-pad//2)) # -(-x//2) == np.ceil(x/2)
                                     for pad in rowcol_padding)
        img = np.pad(self.blur_im, rowcol_split_padding, 'edge') # now padded
        img = 1 - naive_normalise(img) # normalised and inverted. This is a float

        # define function to minimise
        # TODO: enable nwells along x and y
        # not urgent as nwells only needed if more wells would fit into the fov
        # than we actually have
        nwells = int(np.sqrt(self.n_wells_in_fov))
        fun_to_minimise = lambda x: np.abs(
                img - simulate_wells_lattice(
                        img.shape,
                        x[0],x[1],x[2],
                        nwells=nwells,
                        template_shape = self.well_shape
                        )
                ).sum()
        # actual minimisation
        # criterion for bounds choice:
        # 1/2n is if well starts at edge, 1/well if there is another half well!
        # bounds are relative to the size of the image (along the y axis)
        # 1/(nwells+0.5) spacing allows for 1/4 an extra well on both side
        # 1/(nwells-0.5) spacing allows for cut wells at the edges I guess
        bounds = [(1/(2*nwells), 1/nwells),  # x_offset
                  (1/(2*nwells), 1/nwells),  # y_offset
                  (1/(nwells+0.5), 1/(nwells-0.5))]  # spacing
        result = scipy.optimize.differential_evolution(fun_to_minimise, bounds, polish=True)
        # extract output parameters for spacing grid
        x_offset, y_offset, spacing = result.x.copy()
        # convert to pixels
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
                for x in np.arange(x_offset_px,self.img.shape[1],spacing_px)[:nwells]
                for y in np.arange(y_offset_px,self.img.shape[0],spacing_px)[:nwells]
                ])
        # write into dataframe
        self.wells['x'] = xyr[:,0].astype(int)  # centre
        self.wells['y'] = xyr[:,1].astype(int)  # centre
        self.wells['r'] = xyr[:,2].astype(int)  # half-width
        # now calculate the rest. Don't need all the cleaning-up
        for d in ['x','y']:
            self.wells[d+'_min'] = self.wells[d] - self.wells['r']
            self.wells[d+'_max'] = self.wells[d] + self.wells['r']



    def find_square_wells(self, xcorr_threshold=0.85):
        """Cross-correlate the frame with a template approximating a square well.
        Then clean up the results by removing points too close together, or not on a nice grid
        - xcorr_threshold: how high the value of the cross-correlation between
            image and template has to be
        """
        is_debug = False
        # Downscale factor for blur_im. was not saved as a property of the class
        dwnscl_factor = self.img_shape[0]/self.blur_im.shape[0]
        # How many pixels in a square well, in the downscaled image?
        if self.n_wells == 96:
            # TODO: make a dictionary global and call it
            well_size_mm = 8 # roughly, well size of 96wp square wells. Maybe have it as input?
            well_size_px = well_size_mm*1000/self.px2um/dwnscl_factor
        else:
            raise Exception("This case hasn't been coded for yet")

        # make square template approximating a well
        template = make_square_template(n_pxls=well_size_px,
                                        rel_width=0.7,
                                        blurring=0.1,
                                        dtype_out='uint8')

        # find template in image: cross correlation, then threshold and segment
        res = cv2.matchTemplate(self.blur_im, template, cv2.TM_CCORR_NORMED)
        # find local maxima that are at least well_size_px apart
        X=peak_local_max(res, min_distance=well_size_px//2,
                         indices=True,
                         exclude_border=False,
                         threshold_abs=xcorr_threshold)
        xcorr_peaks = np.array( [res[r, c] for r,c in X] )
        print('Initially found {} wells. Removing the ones too close'.format(xcorr_peaks.shape[0]))

        # a bug in peak_local_max means that min_distance is sometimes overlooked.
        # https://github.com/scikit-image/scikit-image/issues/4048
        # seems to be only a problem with peaks with distance == 1
        # adding my own proximity removal system, keeps the highest xcorr point
        # of the conflicting ones

        # create matrix of distances, using implicit expansion
        pkdist2 = (X[:,[0,]] - X[:,0])**2 \
                + (X[:,[1,]] - X[:,1])**2 # column - row makes a matrix, **2 squares it element-wise
        # look for peaks closest than threshold (same as given before)
        dist2_thresh = (well_size_px//2)**2
        pkstooclose = pkdist2 <= dist2_thresh
        # look in upper-diag matrix only
        pkstooclose = np.triu(pkstooclose, k=0) # makes a upper-triangular matrix, keeps diag (k=0), and putls lower-triangular to 0

        # loop on the peaks (rows)
        idx_to_remove = np.zeros(X.shape[0], dtype=bool)
        for ind, row in enumerate(pkstooclose):
            if sum(row) == 1: # only hit was on the diagonal
                continue
            print('more than one hit')
            # more than one proximity hit
            # find xcorr values and position in the list
            pks = xcorr_peaks[row]
            print(pks)
            inds = np.argwhere(row)
            print(inds)
            # find where the highest peak is in the short selection of conflicting points
            ind_max_pk = inds[np.argmax(pks)]
            print(ind_max_pk)
            # store that we need to remove the conflicting points that are not the max peak
            inds_to_remove = np.setdiff1d(inds, ind_max_pk)
            idx_to_remove[inds_to_remove] = True
        # now remove the offending points from X and xcorr_peaks
        X = X[~idx_to_remove]
        xcorr_peaks = xcorr_peaks[~idx_to_remove]
        print('Found {} wells'.format(xcorr_peaks.shape[0]))

#        import pdb
#        pdb.set_trace()

        if is_debug:
            plt.figure()
            ax = plt.axes()
            hi = ax.imshow(res)
            plt.axis('off')
            plt.colorbar(hi, ax=ax, fraction=0.03378, pad=0.01)
            plt.tight_layout()
            plt.plot(X[:,1],X[:,0],'ro', mfc='none')
#            plt.plot(X2[:,1],X2[:,0],'bx')
            fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8))
            axs[0].imshow(self.img, cmap='gray')
            axs[1].imshow(res)
            axs[1].plot(X[:,1], X[:,0], 'ro', mfc='none')

        #NOTE: X contains row and column (y and x) of corner of template matching,
        # and are values within res.shape.
        # To get the centre, add half the template size.
        # This is to be done at the end, to avoid out-of-bound problems in res

        # now get rid of points not nicely on a grid.

        # first get a good estimate of the lattice parameter
        # points on a grid will always have 2 neighbours at
        # distance=lattice spacing
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:,1:]         # throw away the column of zeros (dist from itself)
        lattice_dist = np.median(distances) # this is average lattice spacing
        delta = lattice_dist/10             # this is "grace" interval
        # now look for wells not aligned along the x or y:
        # samecoord is N-by-N matrix. Element [i,j] is true when
        # the coordinate of well i and well j is (roughly) the same
        alonealongcoord = np.zeros(X.shape)
        for i in [0,1]: # dimension counter: y and x
            samecoord = abs(X[:,[i,]] - X[:,i]) < delta
            if is_debug: print(samecoord.astype(int))
            alonealongcoord[:,i] =  samecoord.sum(axis=1)==1
            if is_debug: print(alonealongcoord)
        # a point is not on a grid if it doesn't share its x or y coordinates
        # with at least another point
        idx_not_on_grid = alonealongcoord.any(axis=1)
        if is_debug: print(idx_not_on_grid)
        # apply the filtering
        X = X[~idx_not_on_grid]
        xcorr_peaks = xcorr_peaks[~idx_not_on_grid]
        if is_debug: print(X)

        if is_debug:
            axs[1].plot(X[:,1], X[:,0], 'rx')
            ax.plot(X[:,1], X[:,0], 'rx')

        # write in self.wells
        X = X * dwnscl_factor
        well_size_px *= dwnscl_factor
        self.wells['y'] = X[:,0] + well_size_px/2
        self.wells['x'] = X[:,1] + well_size_px/2
        self.wells['r'] = well_size_px/2

        if is_debug:
            axs[0].plot(self.wells['x'], self.wells['y'], 'rx')

        return


    def find_circular_wells(self):
        """Simply use Hough transform to find circles in MultiWell Plate rgb image.
        The parameters used are optimised for 24 or 48WP"""


        dwnscl_factor = self.img_shape[0]/self.blur_im.shape[0]

        # find circles
        # parameters in downscaled units
        circle_goodness = 70;
        highest_canny_thresh = 10;
        min_well_dist = self.blur_im.shape[1]/3;    # max 3 wells along short side. bank on FOV not taking in all the entirety of the well
        min_well_radius = self.blur_im.shape[1]//7; # if 48WP 3 wells on short side ==> radius <= side/6
        max_well_radius = self.blur_im.shape[1]//4; # if 24WP 2 wells on short side. COnsidering intrawells space, radius <= side/4
        # find circles
        _circles = cv2.HoughCircles(self.blur_im,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1,
                                   minDist=min_well_dist,
                                   param1=highest_canny_thresh,
                                   param2=circle_goodness,
                                   minRadius=min_well_radius,
                                   maxRadius=max_well_radius)
        _circles = np.squeeze(_circles); # because why the hell is there an empty dimension at the beginning?

        # convert back to pixels
        _circles *= dwnscl_factor;

        # output back into class property
        self.wells['x'] = _circles[:,0].astype(int)
        self.wells['y'] = _circles[:,1].astype(int)
        self.wells['r'] = _circles[:,2].astype(int)
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

        # checks: if 24 wells => 4 entries only, if 48 either 3x3 or 3x2
        if self.n_wells == 24:
            _is_2x2 = self.wells.shape[0] == 4 and \
                        self.wells.row.max() == 1 and \
                        self.wells.col.max() == 1
            if not _is_2x2:
                self.plot_wells()
                raise Exception("Found wells not in a 2x2 arrangement, results are unreliable");
        elif self.n_wells == 48:
            _is_3x2 = self.wells.shape[0] == 6 and \
                        self.wells.row.max() == 2 and \
                        self.wells.col.max() == 1
            _is_3x3 = self.wells.shape[0] == 9 and \
                        self.wells.row.max() == 2 and \
                        self.wells.col.max() == 2
            if not (_is_3x2 or _is_3x3):
                self.plot_wells()
                raise Exception("Found wells not in a 3x2 or 3x3 arrangement, results are unreliable");
        elif self.n_wells == 96:
            _is_4x4 = self.wells.shape[0] == 16 and \
                        self.wells.row.max() == 3 and \
                        self.wells.col.max() == 3
            if not _is_4x4:
                self.plot_wells()
                raise Exception("Found wells not in a 4x4 arrangement, "
                                + "results are unreliable");
        return


    def remove_half_circles(self, max_radius_portion_missing=0.5):
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
            y = self.wells[self.wells['row'] == rr]['y'].median()
            x = self.wells[self.wells['col'] == cc]['x'].median()
            r = self.wells['r'].mean()
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

        # odd and even channels have opposite orientation
        # ("up" in the camera is always towards outside of the rig)
        # so flip the row [col] labels before going to read from the MWP_dataframe
        # for odd channels
#        import pdb
#        pdb.set_trace()
        if int(self.channel[-1])%2==1:
            self.wells['well_name'] = \
                [self.mwp_df[self.channel].iloc[max_row-r, max_col-c] \
                 for r,c in self.wells[['row','col']].values]
        else:
            self.wells['well_name'] = [self.mwp_df[self.channel].iloc[r,c] \
                         for r,c in self.wells[['row','col']].values]
        # the above code is equivalent (but faster than) the following two alternatives:
#        ################### alternative 1
#        def flip_oddchannels_rowcol(r, c, chname):
#            if int(chname[-1])%2==1:
#                row = max_row-r
#                col = max_col-c
#            else:
#                row = r
#                col = c
#            return (row, col)
#        # define function that acts on each row
#        def _apply_dict_to_row(_well):
#            row, col = flip_oddchannels_rowcol(_well['row"], _well['col"], self.channel)
#            return self.mwp_df.iloc[row,col]
#        # apply function to each row
#        self.wells['well_name'] = self.wells.apply(_apply_dict_to_row, axis=1)
#        #################### alternative 2
#        # the lines above are equivalent to:
#        for _i, _well in self.wells.iterrows():
#            row, col = flip_oddchannels_rowcol(_well['row"], _well['col"], self.channel)
#            self.wells.loc[_i,'well_name'] = self.mwp_df.iloc[row,col]

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
        wells, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            roi_img = img[well['y_min']:well['y_max'],well['x_min']:well['x_max']]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def tile_FOV_3D(self, img):
        """
        Function that chops an image stack (1st dimension is n_frames)
        according to the x/y_min/max coordinates in
        wells, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            roi_img = img[:,well['y_min']:well['y_max'],well['x_min']:well['x_max']]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def plot_wells(self):
        """
        Plot the fitted wells, the wells separation, and the name of the well.
        (only if these things are present!)"""

        # make sure I'm not working on the original image
        _img = cv2.cvtColor(self.img.copy(),cv2.COLOR_GRAY2BGR)
#        pdb.set_trace()
        # flags: according to dataframe state, do or do not do
        _is_wells = self.wells.shape[0] > 0;
        _is_rois = np.logical_not(self.wells['x_min'].isnull()).all() and _is_wells;
        _is_wellnames = np.logical_not(self.wells['well_name'].isnull()).all() and _is_rois;
        # TODO: deal with grayscale image
        # burn the circles into the rgb image
        if _is_wells and self.well_shape == 'circle':
            for i, _circle in self.wells.iterrows():
                # draw the outer circle
                cv2.circle(_img,(_circle.x,_circle.y),_circle.r,(255,0,0),5)
                # draw the center of the circle
                cv2.circle(_img,(_circle.x,_circle.y),5,(0,255,255),5)
        # burn the boxes edges into the RGB image
        if _is_rois:
            #normalize item number values to colormap
            # normcol = colors.Normalize(vmin=0, vmax=self.wells.shape[0])
#            print(self.wells.shape[0])
            for i, _well in self.wells.iterrows():
                color = get_well_color(_well.is_good_well,
                                       forCV=True)
                cv2.rectangle(_img,
                              (_well.x_min, _well.y_min),
                              (_well.x_max, _well.y_max),
#                              colors[0], 20)
                               color, 20)

        # add names of wells
        # plot, don't close

        print(_img.shape)
        figsize = (8*_img.shape[0]/_img.shape[1], 8)
        hf = plt.figure(figsize=figsize);
        ha = plt.Axes(hf, [0., 0., 1., 1.])
        ha.set_axis_off()
        hf.add_axes(ha)
        ha.imshow(_img)
        if _is_wellnames:
            for i, _well in self.wells.iterrows():
                try:
                    txt = "{} ({:d},{:d})".format(_well['well_name'],
                           int(_well['row']),
                           int(_well['col']))
                except:  # could not have row, col if from /fov_wells
                    txt = "{}".format(_well['well_name'])
                plt.text(_well['x_min']+_well['width']*0.05,
                         _well['y_min']+_well['height']*0.12,
                         txt,
                         fontsize=10,
                         color=np.array(get_well_color(_well['is_good_well'],
                                                       forCV=False))
                         )
                         # color='r')
        elif _is_rois:
            for i, _well in self.wells.iterrows():
                plt.text(_well['x'], _well['y'],
                         "({:d},{:d})".format(int(_well['row']),int(_well['col'])),
                         fontsize=12,
                         weight='bold',
                         color='r')
#        plt.axis('off')
#        plt.tight_layout()
        return hf


    def find_well_of_xy(self, x, y):
        """
        Takes two numpy arrays (or pandas columns), returns an array of strings
        of the same with the name of the well each x,y, pair falls into
        """
        # I think a quick way is by using implicit expansion
        # treat the x array as column, and the *_min and *_max as rows
        # these are all matrices len(x)-by-len(self.wells)
        # none creates new axis
        if np.isscalar(x):
            x = np.array([x])
            y = np.array([y])

        within_x = np.logical_and(
                (x[:,None] - self.wells['x_min'][None,:]) >= 0,
                (x[:,None] - self.wells['x_max'][None,:]) <= 0)
        within_y = np.logical_and(
                (y[:,None] - self.wells['y_min'][None,:]) >= 0,
                (y[:,None] - self.wells['y_max'][None,:]) <= 0)
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
        Create a black mask covering a 10% thick edge of the square region covering each well
        """
        assert self.well_masked_edge < 0.5, \
            "well_masked_edge has to be less than 50% or no data left"

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

        return mask


    def apply_wells_mask(self, img):
        """
        performs img*= mask"""
        img *= self.wells_mask



# end of class

def process_image_from_name(image_name, is_plot=True, is_save=True):
    fname = str(image_name)
#        print(fname)
    img_ = cv2.imread(fname)
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    # find camera name
    regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
    camera_serial = re.findall(regex, str(fname).lower())[0]
    print(camera_serial)
    # run fov splitting
    fovsplitter = FOVMultiWellsSplitter(img,
                                        camera_serial=camera_serial,
                                        total_n_wells=96,
                                        whichsideup='upright',
                                        well_shape='square',
                                        px2um=12.4)
    if is_plot:
        fig = fovsplitter.plot_wells()
    if is_save:
        fig.savefig(fname.replace('.png','_wells.png'),
                    bbox_inches='tight',
                    dpi=img_.shape[0]/fig.get_size_inches()[0],
                    pad_inches=0,
                    transparent=True)
    return fovsplitter

#%% main

if __name__ == '__main__':

    import os
    import time
    import re
    import tqdm
    from pathlib import Path
#    from scipy.signal import find_peaks

    plt.close("all")

    # test from images:

#    wd = Path.home() / 'Desktop/Data_FOVsplitter'
#    img_dir = wd / 'RawVideos/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312_firstframes'
#    wd = Path.home() / 'Desktop/Data_FOVsplitter'
#    img_dir = wd / 'RawVideos/drugexperiment_1hrexposure'

#    wd = Path('/Volumes/behavgenom$/Luigi/Data/LoopBio_calibrations/wells_mapping/20190710/')
#    img_dir = wd
##    img_dir = wd / 'Hydra03'
#
#
#    fnames = list(img_dir.rglob('*.png'))
#    fnames = [str(f) for f in fnames if '_wells' not in str(f)]
###    fnames = fnames[2:3] # for code-review only
#    for fname in tqdm.tqdm(fnames):
#        process_image_from_name(fname)
#    plt.close('all')
#
    #%% this file didn't work, why?
#    masked_image_file = '/Volumes/behavgenom$/Ida/Data/Hydra/PilotDrugExps/20191003/MaskedVideos/pilotdrugs_run1_20191003_161308.22956807/metadata.hdf5'
#    features_file = masked_image_file.replace('MaskedVideos','Results').replace('.hdf5','_featuresN.hdf5')
#    img, camera_serial, px2um = calculate_bgnd_from_masked_fulldata(masked_image_file)
#    fovsplitter = FOVMultiWellsSplitter(img,
#                                        camera_serial=camera_serial,
#                                        total_n_wells=96,
#                                        whichsideup='upright',
#                                        well_shape='square',
#                                        px2um=12.4)


#    fovsplitter = FOVMultiWellsSplitter(masked_image_file)

    #%% test on filename

#    masked_image_file = '/Users/lferiani/Desktop/Data_FOVsplitter/short/MaskedVideos/drugexperiment_1hr30minexposure_set1_bluelight_20190722_173404.22436248/metadata.hdf5'
#    features_file = masked_image_file.replace('MaskedVideos','Results').replace('.hdf5','_featuresN.hdf5')
#    import shutil
#    shutil.copy(features_file.replace('.hdf5','.bk'), features_file)

#    fovsplitter = FOVMultiWellsSplitter(masked_image_file)

#    foo_wells = fovsplitter.get_wells_data()

#    plt.imshow(fovsplitter.apply_mask_wells(fovsplitter.img),cmap='gray')

    #%%


#    fig = fovsplitter.plot_wells()

#    with pd.HDFStore(features_file, 'r') as fid:
#        traj_data = fid['/trajectories_data']
#        well_names = fovsplitter.find_well_from_trajectories_data(traj_data)

     #%%
#
#    wd = Path('/Volumes/behavgenom$/Luigi/Data/Blue_LEDs_tests/RawVideos/20191104/')
##    wd = wd / 'blueled_tests_run01_20191104_172258_firstframes'
#    fnames_to_delete = list(wd.rglob('*_wells.png'))
#    for fname in fnames_to_delete:
#        os.remove(str(fname))
#    fnames = list(wd.rglob('*.png'))
#    fnames = [str(f) for f in fnames]
#    print('{} images to process'.format(len(fnames)))
##    tic = time.time()
##    process_image_from_name(fnames[2])
##    print('totatl time:',time.time()-tic)
#
#
#    import multiprocessing
#    import tqdm
#
#    if multiprocessing.get_start_method() != 'spawn':
#        multiprocessing.set_start_method('spawn', force=True)
#
#    batch_size = 6
#
#    with multiprocessing.Pool(batch_size) as p:
#        for ind in tqdm.tqdm(range(0, len(fnames), batch_size)):
#            batched_fnames = fnames[ind:ind+batch_size]
#            # process them in parallel
#            outs = p.imap_unordered(process_image_from_name, batched_fnames)
#            # do nothing with the outputs
#            for _ in outs:
#                pass
#
#    import subprocess as sp
#    wd = Path('/Volumes/behavgenom$/Luigi/Data/Blue_LEDs_tests/RawVideos/20191104/')
#    fnames_to_move = list(wd.rglob('*_wells.png'))
#    dst = wd / 'wells'
#    for fname in fnames_to_move:
#        cmdlist = ['mv', str(fname), str(dst)+'/']
##        print(cmd)
#        sp.run(cmdlist)

# %%

    wd = Path('~/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/').expanduser()

    masked_image_file = wd / 'MaskedVideos/20191205/'
    masked_image_file = (
            masked_image_file /
            'syngenta_screen_run1_bluelight_20191205_151104.22956805' /
            'metadata.hdf5'
            )
    features_file = str(masked_image_file).replace('MaskedVideos','Results')
    features_file = features_file.replace('.hdf5','_featuresN.hdf5')
    features_file = Path(features_file)
    import shutil
    shutil.copy(features_file.with_suffix('.bak'),
                features_file)

    fovsplitter = FOVMultiWellsSplitter(features_file)
    fovsplitter.plot_wells()
    fovsplitter.wells.loc[::3,'is_good_well']=False
    fovsplitter.wells.loc[1::3,'is_good_well']=True
    fovsplitter.wells.loc[2::3,'is_good_well']=-1
    print(fovsplitter.get_wells_data())
    fovsplitter.write_fov_wells_to_file(features_file)

    fovsplitter = FOVMultiWellsSplitter(features_file)
    fovsplitter.plot_wells()
