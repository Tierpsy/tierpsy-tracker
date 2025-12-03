import os
from functools import partial
import numpy as np
import pandas as pd
import tables
import matplotlib
import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QFont, QPen, QPolygonF, QColor, QKeySequence, QBrush
from PyQt5.QtWidgets import QApplication, QMessageBox

from tierpsy.analysis.ske_create.helperIterROI import getWormROI
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter

from tierpsy.gui.MWTrackerViewer_ui import Ui_MWTrackerViewer
from tierpsy.gui.TrackerViewerAux import (
    TrackerViewerAuxGUI, BAD_SKEL_COLOURS, GOOD_SKEL_COLOURS)
from tierpsy.gui.PlotFeatures import PlotFeatures

from tierpsy.helper.misc import WLAB, save_modified_table
from tierpsy.analysis.split_fov.helper import get_well_color


class WellsDrawer(TrackerViewerAuxGUI):
    '''
    Dummy class with the wells division drawer functions
    '''
    def __init__(self, ui):
        super().__init__(ui)
        # colour
        self.fovsplitter_mask = None
        self.fovsplitter_feat = None
        self.fovsplitter = None
        self.is_fov_tosplit = None


    def updateVideoFile(self, vfilename):
        super().updateVideoFile(vfilename)
        # check if /fov_wells exists in masked video
        if self.fid is not None:
            if '/fov_wells' not in self.fid:
                self.is_fov_tosplit = False
            else:
                self.is_fov_tosplit = True
            # if it exists, read it
            if self.is_fov_tosplit:
#                self.wells_in_mask = pd.DataFrame(
#                        self.fid.get_node('/fov_wells').read())
                self.fovsplitter_mask = FOVMultiWellsSplitter(self.vfilename)

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        # if no skeletons, skip
        if not self.skeletons_file:
            return
        # check if /fov_wells exists in features video
        with tables.File(self.skeletons_file, 'r') as fid:
            if '/fov_wells' not in fid:
                self.is_fov_tosplit = False
#                print("didn't find fov wells though")
            else:
                self.is_fov_tosplit = True
#                print("found fov wells in featuresN")
            # if it exists, read it
            if self.is_fov_tosplit:
#                print('reading fov_wells from featuresN')
#                print('pre-reading:')
#                print(self.wells)
#                self.wells_in_feat = pd.DataFrame(
#                        fid.get_node('/fov_wells').read())
                self.fovsplitter_feat = FOVMultiWellsSplitter(self.skeletons_file)


    def draw_wells(self, image):
        '''
        Draw wells.
        '''
        if self.is_fov_tosplit:
            if self.fovsplitter_feat is not None:
                self.fovsplitter = self.fovsplitter_feat
            else: # fall back to mask ones
                print('falling back')
                self.fovsplitter = self.fovsplitter_mask

            # prepare constants for drawing
            self.fontsize = max(1, max(image.height(), image.width()) // 60)
            penwidth = max(1, max(image.height(), image.width()) // 400)
            self.penwidth = penwidth if penwidth % 2 == 1 else penwidth + 1
#            self.wellsC = QColor(250, 140, 0)
            if 'is_good_well' in self.fovsplitter.wells.columns:
                is_color_by_well = True
            else:
                is_color_by_well = False
            # Qt drawing code
            painter = QPainter()
            painter.begin(image)
            pen = QPen()
            pen.setWidth(self.penwidth)
            painter.setFont(QFont('Decorative', self.fontsize))
            # loop on wells
            for _, well in self.fovsplitter.wells.iterrows():
                # update color every time
                if is_color_by_well:
                    wellC = get_well_color(well['is_good_well'], forCV=True)
                    wellC = QColor(*wellC)
                else:
                    wellC = QColor(250, 140, 0)
                pen.setColor(wellC)
                painter.setPen(pen)
                # draw well name
                painter.drawText(well['x_min'] + self.fontsize*0.4,
                                 well['y_min'] + self.fontsize*1.2,
                                 well['well_name'])
                # draw rectangle
                painter.drawRect(well['x_min'],
                                 well['y_min'],
                                 well['x_max'] - well['x_min'],
                                 well['y_max'] - well['y_min'])
                if well['is_good_well'] == False:
                    painter.drawLine(well['x_min'],
                                     well['y_min'],
                                     well['x_max'],
                                     well['y_max'])
            painter.end()




#        super().keyPressEvent(event)


class FoodContourDrawer(TrackerViewerAuxGUI):
    '''
    Dummy class with the contour functions
    '''
    def __init__(self, ui):
        super().__init__(ui)

        self.food_coordinates = None
        self.wlabC = {
            WLAB['U']: Qt.white,
            WLAB['WORM']: Qt.green,
            WLAB['WORMS']: Qt.blue,
            WLAB['BAD']: Qt.darkRed,
            WLAB['GOOD_SKE']: Qt.darkCyan
            }
        self.ui.checkBox_showFood.stateChanged.connect(self.updateImage)
        self.ui.checkBox_showFood.setEnabled(False)
        self.ui.checkBox_showFood.setChecked(True)

    def updateSkelFile(self, skeletons_file):

        super().updateSkelFile(skeletons_file)
        if not self.skeletons_file or self.trajectories_data is None:
            self.food_coordinates = None
            return

        with tables.File(self.skeletons_file, 'r') as fid:
            if not '/food_cnt_coord' in fid:
                self.food_coordinates = None
                self.ui.checkBox_showFood.setEnabled(False)
            else:
                #change from microns to pixels
                self.food_coordinates = fid.get_node('/food_cnt_coord')[:]
                self.food_coordinates /= self.microns_per_pixel

                self.ui.checkBox_showFood.setEnabled(True)

    def draw_food_contour(self, image):
        if self.food_coordinates is None or not self.ui.checkBox_showFood.isChecked():
            return

        painter = QPainter()
        painter.begin(image)

        penwidth = max(1, max(image.height(), image.width()) // 800)
        col = Qt.darkMagenta
        p = QPolygonF()
        for x,y in self.food_coordinates:
            p.append(QPointF(x,y))

        pen = QPen()
        pen.setWidth(penwidth)
        pen.setColor(col)
        painter.setPen(pen)

        painter.drawPolyline(p)
        painter.end()

class IntensityLabeler(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui)

        self.mean_intensity = None
        self.ui.intensity_label.setStyleSheet('') #avoid displaying color at the start of the programı

    def updateVideoFile(self, vfilename):
        super().updateVideoFile(vfilename)
        if self.fid is not None:
            #get mean intensity information.
            #Useful for the optogenetic experiments.
            try:
                mean_int = self.fid.get_node('/mean_intensity')[:]

                #calculate the intensity range and normalize the data.
                #I am ignoring any value less than 1. The viewer only works with uint8 data.

                dd = mean_int[mean_int>=1]
                if dd.size == 0:
                    raise ValueError

                bot = np.min(dd)
                top = np.max(dd)
                rr = top-bot

                # if the mean value change is less than 1 (likely continous image do nothing)
                if rr <= 1:
                    raise ValueError

                self.mean_intensity = (mean_int-bot)/(rr)

            except (tables.exceptions.NoSuchNodeError, ValueError):
                self.mean_intensity = None
                self.ui.intensity_label.setStyleSheet('')

    def display_intensity(self):
        if self.mean_intensity is not None and self.frame_number < self.mean_intensity.size:
            d = int(self.mean_intensity[self.frame_number]*255)
            self.ui.intensity_label.setStyleSheet('QLabel {background-color: rgb(%i, %i, %i);}' % (0, 0, d))

class BlobLabeler(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui)
        self.wlab = WLAB
        self.label_type = 'worm_label'

        self.ui.pushButton_U.clicked.connect(
            partial(self._h_tag_worm, self.wlab['U']))
        self.ui.pushButton_W.clicked.connect(
            partial(self._h_tag_worm, self.wlab['WORM']))
        self.ui.pushButton_WS.clicked.connect(
            partial(self._h_tag_worm, self.wlab['WORMS']))
        self.ui.pushButton_B.clicked.connect(
            partial(self._h_tag_worm, self.wlab['BAD']))

        self.ui.pushButton_W.setShortcut(QKeySequence(Qt.Key_W))
        self.ui.pushButton_U.setShortcut(QKeySequence(Qt.Key_U))
        self.ui.pushButton_WS.setShortcut(QKeySequence(Qt.Key_C))
        self.ui.pushButton_B.setShortcut(QKeySequence(Qt.Key_B))


    def enable_label_buttons(self, value):
        self.ui.pushButton_U.setEnabled(value)
        self.ui.pushButton_W.setEnabled(value)
        self.ui.pushButton_WS.setEnabled(value)
        self.ui.pushButton_B.setEnabled(value)


    def _h_tag_worm(self, label_ind):
        if not self.worm_index_type == 'worm_index_manual':
            return

        worm_ind = self.current_worm_index

        if self.frame_data is None:
            return

        if not worm_ind in self.frame_data['worm_index_manual'].values:
            QMessageBox.critical(
                self,
                'The selected worm is not in this frame.',
                'Select a worm in the current frame to label.',
                QMessageBox.Ok)
            return

        good = self.trajectories_data['worm_index_manual'] == worm_ind
        self.trajectories_data.loc[good, 'worm_label'] = label_ind
        self.updateImage()


class ROIWorm():
    def __init__(self, wormCanvas, comboBox_ROI, checkBox_ROI):
        self.worm_index = None
        self.wormCanvas = wormCanvas
        self.comboBox_ROI = comboBox_ROI
        self.checkBox_ROI = checkBox_ROI

        self.comboBox_ROI.activated.connect(self.selectROI)

    def selectROI(self, index):
        try:
            self.worm_index = int(self.comboBox_ROI.itemText(index))
        except ValueError:
            self.worm_index = None

    @property
    def isDrawSkel(self):
        return self.checkBox_ROI.isChecked()


class ROIManager(TrackerViewerAuxGUI):
    def __init__(self, ui):

        super().__init__(ui)
        self.rois = [
            ROIWorm(
                self.ui.wormCanvas1,
                self.ui.comboBox_ROI1,
                self.ui.checkBox_ROI1
                ),
            ROIWorm(
                self.ui.wormCanvas2,
                self.ui.comboBox_ROI2,
                self.ui.checkBox_ROI2
                )
        ]


        self.ui.radioButton_ROI1.setShortcut(QKeySequence(Qt.Key_Up))
        self.ui.radioButton_ROI2.setShortcut(QKeySequence(Qt.Key_Down))


        self.ui.checkBox_ROI1.stateChanged.connect(partial(self._updateROI, self.rois[0]))
        self.ui.checkBox_ROI2.stateChanged.connect(partial(self._updateROI, self.rois[1]))

        self.ui.comboBox_ROI1.activated.connect(partial(self._updateROI, self.rois[0]))
        self.ui.comboBox_ROI2.activated.connect(partial(self._updateROI, self.rois[1]))

        # flags for RW and FF
        self.RW, self.FF = 1, 2
        self.ui.pushButton_ROI1_RW.clicked.connect(partial(self.roiRWFF, self.RW, self.rois[0]))
        self.ui.pushButton_ROI1_FF.clicked.connect(partial(self.roiRWFF, self.FF, self.rois[0]))
        self.ui.pushButton_ROI2_RW.clicked.connect(partial(self.roiRWFF, self.RW, self.rois[1]))
        self.ui.pushButton_ROI2_FF.clicked.connect(partial(self.roiRWFF, self.FF, self.rois[1]))

    @property
    def current_roi(self):
        if self.ui.radioButton_ROI1.isChecked():
            return self.rois[0]
        elif self.ui.radioButton_ROI2.isChecked():
            return self.rois[1]
        else:
            raise ValueError("I shouldn't be here")

    @property
    def current_worm_index(self):
        return self.current_roi.worm_index

    def updateSkelFile(self, skeletons_file):
        for roi in self.rois:
            roi.worm_index = None
        super().updateSkelFile(skeletons_file)

    def keyPressEvent(self, event):
        #MORE SHORTCUTS
        # go the the start of end of a trajectory
        if event.key() == Qt.Key_BracketLeft:

            self.roiRWFF(self.RW, self.current_roi)

        elif event.key() == Qt.Key_BracketRight:

            self.roiRWFF(self.FF, self.current_roi)

        super().keyPressEvent(event)


    def updateROIcomboBox(self, roi):
        # update valid index for the comboBox
        roi.comboBox_ROI.clear()

        if roi.worm_index is not None:
            roi.comboBox_ROI.addItem(str(int(roi.worm_index)))


        for ind in self.frame_data[self.worm_index_type]:
            roi.comboBox_ROI.addItem(str(int(ind)))

        if roi.worm_index is None:
            w_ind = float(roi.comboBox_ROI.itemText(0))
            roi.worm_index = int(w_ind)

    # function that generalized the updating of the ROI
    def _updateROI(self, roi):

        if self.frame_data is None or not self.worm_index_type:
            # no trajectories data presented, nothing to do here
            roi.wormCanvas.clear()
            return

        self.updateROIcomboBox(roi)

        # extract individual worm ROI
        good = self.frame_data[self.worm_index_type] == roi.worm_index
        row_data = self.frame_data.loc[good].squeeze()

        if row_data.size == 0 or \
        np.isnan(row_data['coord_x']) or \
        np.isnan(row_data['coord_y']):
            # invalid data nothing to do here
            roi.wormCanvas.clear()
            return

        worm_img, roi_corner = getWormROI(self.frame_img,
                                        row_data['coord_x'],
                                        row_data['coord_y'],
                                        row_data['roi_size']
                                        )

        roi_ori_size = worm_img.shape
        worm_img = np.ascontiguousarray(worm_img)
        worm_qimg = self._convert2Qimg(worm_img)

        canvas_size = min(roi.wormCanvas.height(), roi.wormCanvas.width())
        worm_qimg = worm_qimg.scaled(
            canvas_size, canvas_size, Qt.KeepAspectRatio)

        worm_qimg = self.drawSkelResult(worm_img, worm_qimg, row_data, roi.isDrawSkel, roi_corner, read_center=False)

        pixmap = QPixmap.fromImage(worm_qimg)
        roi.wormCanvas.setPixmap(pixmap)

    def updateROIs(self):
        for roi in self.rois:
            self._updateROI(roi)

    def clearROIs(self):
        for roi in self.rois:
            roi.wormCanvas.clear()

    # move to the first or the last frames of a trajectory
    def roiRWFF(self, rwff, roi):

        if self.frame_data is None:
            return

        # use 1 for rewind RW or 2 of fast forward
        good = self.trajectories_data[self.worm_index_type] == roi.worm_index
        frames = self.trajectories_data.loc[good, 'frame_number']

        if frames.size == 0:
            return

        if rwff == self.RW:
            self.frame_number = frames.min()
        elif rwff == self.FF:
            self.frame_number = frames.max()
        else:
            raise ValueError('Invalid rwff value : {} '.format(rwff))

        self.ui.spinBox_frame.setValue(self.frame_number)

class TrajectoryEditor(ROIManager):
    def __init__(self, ui):
        super().__init__(ui)
        self.ui.pushButton_join.clicked.connect(self.joinTraj)
        self.ui.pushButton_split.clicked.connect(self.splitTraj)

        #SHORTCUTS
        self.ui.pushButton_join.setShortcut(QKeySequence(Qt.Key_J))
        self.ui.pushButton_split.setShortcut(QKeySequence(Qt.Key_S))

    def enable_trajectories_buttons(self, value):
        self.ui.pushButton_join.setEnabled(value)
        self.ui.pushButton_split.setEnabled(value)

    def joinTraj(self):
        if self.worm_index_type != 'worm_index_manual' \
        or self.frame_data is None:
            return

        worm_ind1 = self.rois[0].worm_index
        worm_ind2 = self.rois[1].worm_index

        if worm_ind1 == worm_ind2:
            QMessageBox.critical(
                self,
                'Cannot join the same trajectory with itself',
                'Cannot join the same trajectory with itself.',
                QMessageBox.Ok)
            return

        index1 = (self.trajectories_data[
                  'worm_index_manual'] == worm_ind1).values
        index2 = (self.trajectories_data[
                  'worm_index_manual'] == worm_ind2).values

        # if the trajectories do not overlap they shouldn't have frame_number
        # indexes in commun
        frame_number = self.trajectories_data.loc[
            index1 | index2, 'frame_number']

        if frame_number.size != np.unique(frame_number).size:
            QMessageBox.critical(
                self,
                'Cannot join overlaping trajectories',
                'Cannot join overlaping trajectories.',
                QMessageBox.Ok)
            return

        if not (worm_ind1 in self.frame_data[
                'worm_index_manual'].values or worm_ind2 in self.frame_data['worm_index_manual'].values):
            reply = QMessageBox.question(
                self,
                'Message',
                "The none of the selected worms to join is not in this frame. Are you sure to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)

            if reply == QMessageBox.No:
                return

        # get the first row for each segment to extract some data
        first_row1 = self.trajectories_data.loc[index1, :].iloc[0]
        first_row2 = self.trajectories_data.loc[index2, :].iloc[0]

        # join trajectories
        self.trajectories_data.loc[
            index2, 'worm_label'] = first_row1['worm_label']
        self.trajectories_data.loc[index2, 'worm_index_manual'] = worm_ind1

        self.rois[0].worm_index = worm_ind1
        self.rois[1].worm_index = worm_ind1

        #this might be too slow. I might need to change it
        self.traj_worm_index_grouped = self.trajectories_data.groupby(self.worm_index_type)

        self.updateImage()


    def splitTraj(self):
        if self.worm_index_type != 'worm_index_manual' \
        or self.frame_data is None:
            return

        worm_ind = self.current_worm_index

        if not worm_ind in self.frame_data['worm_index_manual'].data:
            QMessageBox.critical(
                self,
                'Worm index is not in the current frame.',
                'Worm index is not in the current frame. Select a valid index.',
                QMessageBox.Ok)
            return

        last_index = self.trajectories_data['worm_index_manual'].max()

        new_ind1 = last_index + 1
        new_ind2 = last_index + 2

        good = self.trajectories_data['worm_index_manual'] == worm_ind
        frames = self.trajectories_data.loc[good, 'frame_number']
        frames = frames.sort_values(inplace=False)

        good = frames < self.frame_number
        index1 = frames[good].index
        index2 = frames[~good].index
        self.trajectories_data.ix[index1, 'worm_index_manual'] = new_ind1
        self.trajectories_data.ix[index2, 'worm_index_manual'] = new_ind2

        self.rois[0].index = new_ind1
        self.rois[1].index = new_ind2

        #this might be too slow. I might need to change it
        self.traj_worm_index_grouped = self.trajectories_data.groupby(self.worm_index_type)

        self.updateImage()

class FeatureReaderBase(TrackerViewerAuxGUI):
    index_cols = ['worm_index', 'timestamp', 'motion_modes', 'skeleton_id', 'well_name']
    valid_fields = ['/timeseries_data', '/features_timeseries']

    def __init__(self, ui):
        self.timeseries_data = None
        self.feat_column = ''

        super().__init__(ui)

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        try:
            self.traj_colors = {}
            with pd.HDFStore(self.skeletons_file, 'r') as ske_file_id:
                for field in self.valid_fields:
                    if field in ske_file_id:
                        self.timeseries_data = ske_file_id[field]

                        if field == '/timeseries_data':
                            blob_features = ske_file_id['/blob_features']
                            blob_features.columns = ['blob_' + x for x in blob_features.columns]
                            self.timeseries_data = pd.concat((self.timeseries_data, blob_features), axis=1)
                        break
                else:
                    raise KeyError

            if not len(self.timeseries_data) != len(self.trajectories_data):
                ValueError('timeseries_data and trajectories_data does not match. You might be using an old version of featuresN.hdf5')


            self.valid_features = [x for x in self.timeseries_data.columns if x not in self.index_cols]


        except (TypeError, AttributeError, IOError, KeyError, tables.exceptions.HDF5ExtError):
            self.valid_features = None
            self.timeseries_data = None

class MarkersDrawer(FeatureReaderBase):
    def __init__(self, ui):
        super().__init__(ui)

        self.traj_colors = {}
        self.n_points_traj = 250
        self.n_colors = 256
        cmap = matplotlib.cm.get_cmap("bwr")
        palette = [cmap(x) for x in np.linspace(0, 1, self.n_colors)]
        #palette = sns.color_palette("RdBu_r", self.n_colors)
        palette = np.round(np.array(palette)*255).astype(np.int)
        self.palette = [QColor(*x) for x in palette]


        self.drawT = {x: self.ui.comboBox_drawType.findText(x , flags=Qt.MatchContains)
                                for x in ['boxes', 'traj', 'skel']}

        self.showT = {x: self.ui.comboBox_showLabels.findText(x , flags=Qt.MatchContains)
                                for x in ['hide', 'all', 'filter']}

        self.ui.comboBox_showLabels.setCurrentIndex(self.showT['all'])

        self.ui.comboBox_showLabels.currentIndexChanged.connect(self.updateImage)
        self.ui.comboBox_drawType.currentIndexChanged.connect(self.updateImage)

        self.ui.feature_column.currentIndexChanged.connect(self.change_feature)


        self.ui.feat_max_value.valueChanged.connect(self.updateImage)
        self.ui.feat_min_value.valueChanged.connect(self.updateImage)
        self.ui.is_color_features.stateChanged.connect(self.updateImage)

        self.enable_color_feats(False)


        self.ui.spinBox_step.valueChanged.connect(self.updateImage)

    def updateSkelFile(self, skeletons_file):
        self.ui.is_color_features.setChecked(False)

        super().updateSkelFile(skeletons_file)

        self.ui.feature_column.clear()
        if self.timeseries_data is None:
            #no feature data
            self.enable_color_feats(False)
        else:
            self.enable_color_feats(True)
            self.ui.feature_column.addItems(self.valid_features)
            self._h_find_feat_limits()

    def change_feature(self):
        self._h_find_feat_limits()
        self.updateImage()

    def _h_find_feat_limits(self):
        self.feat_column = str(self.ui.feature_column.currentText())
        print(self.feat_column)

        if self.feat_column and self.timeseries_data is not None:
            f_max = self.timeseries_data[self.feat_column].max()
            f_min = self.timeseries_data[self.feat_column].min()
            q1, q2 = self.timeseries_data[self.feat_column].quantile([0.02, 0.98])

        else:
            f_min, f_max, q1, q2  = 0,0,0,0

        self.ui.feat_max_value.setRange(f_min, f_max)
        self.ui.feat_min_value.setRange(f_min, f_max)
        self.ui.feat_min_value.setValue(q1)
        self.ui.feat_max_value.setValue(q2)


    def enable_color_feats(self, value):
        self.ui.feature_column.setEnabled(value)
        self.ui.feat_min_value.setEnabled(value)
        self.ui.feat_max_value.setEnabled(value)
        self.ui.is_color_features.setEnabled(value)


    def _h_assign_feat_color(self, irow):

        feat_val = self.timeseries_data.loc[irow, self.feat_column]

        if (feat_val != feat_val):
            return Qt.black

        #this function can and should be optimized
        f_min = self.ui.feat_min_value.value()
        f_max = self.ui.feat_max_value.value()

        if f_min == f_max: #dummy range in case all the values are the same
            f_min, f_max = -1, 1
        elif f_min > f_max:
            return Qt.black

        nn = np.clip((feat_val - f_min)/(f_max - f_min), 0, 1)
        ind = int(np.round(nn*(self.n_colors-1)))
        
        col = self.palette[ind]
        return col


    def draw_worm_markers(self, image):
        '''
        Draw traj worm trajectory.
        '''
        if not self.worm_index_type in self.frame_data or \
        self.ui.comboBox_showLabels.currentIndex() == self.showT['hide']:
            return

        if hasattr(self, 'current_worm_index'):
            current_index = self.current_worm_index
        else:
            current_index = -1

        painter = QPainter()
        painter.begin(image)

        self.fontsize = max(1, max(image.height(), image.width()) // 120)

        penwidth = max(1, max(image.height(), image.width()) // 800)
        self.penwidth = penwidth if penwidth % 2 == 1 else penwidth + 1

        if not self.label_type in self.frame_data:
            self.frame_data[self.label_type] = self.wlab['U']

        for row_id, row_data in self.frame_data.iterrows():
            # check if the coordinates are nan
            if np.isnan(row_data['coord_x']) or np.isnan(row_data['coord_y']):
                continue

            #if select between showing filtered index or not
            if self.ui.comboBox_showLabels.currentIndex() == self.showT['filter']:
                continue

            is_current_index =  current_index == int(row_data[self.worm_index_type])

            cb_ind = self.ui.comboBox_drawType.currentIndex()
            if cb_ind == self.drawT['boxes']:
                self.draw_boxes(painter, row_id, row_data, is_current_index)
            elif cb_ind == self.drawT['traj']:
                self.draw_trajectories(painter, row_data, is_current_index)
            elif cb_ind == self.drawT['skel']:
                self.draw_skeletons(painter, row_id, row_data, is_current_index)


        painter.end()

    def _h_get_trajectory(self, worm_index, current_frame):
        worm_data = self.traj_worm_index_grouped.get_group(worm_index)
        valid_index = worm_data.index[worm_data['frame_number']<= current_frame]

        ini = max(0, valid_index.size - self.frame_step*self.n_points_traj)
        traj_ind = valid_index.values[ini::self.frame_step]
        traj_data = worm_data.loc[traj_ind]
        return traj_data


    def draw_trajectories(self, painter, row_data, is_current_index):
        if self.traj_worm_index_grouped is None:
            return
        worm_index = int(row_data[self.worm_index_type])
        current_frame = row_data['frame_number']
        traj_data = self._h_get_trajectory(worm_index, current_frame)
        traj_data = traj_data.dropna(subset=['coord_x', 'coord_y'])

        x_v = traj_data['coord_x'].round()
        y_v = traj_data['coord_y'].round()
        points = [QPointF(*map(int, c)) for c in zip(x_v, y_v)]

        if self.ui.is_color_features.isChecked():

            vec_color = [self._h_assign_feat_color(x) for x in traj_data.index]

            pen = QPen()
            pen.setWidth(self.penwidth)
            for p1, p2, c in zip(points[1:], points[:-1], vec_color):
                pen.setColor(c)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
        else:
            pol = QPolygonF()
            for p in points:
                pol.append(p)

            if not worm_index in self.traj_colors:
                self.traj_colors[worm_index] = QColor(*np.random.randint(50, 230, 3))
            col = self.traj_colors[worm_index]

            pen = QPen()
            pen.setWidth(self.penwidth)
            pen.setColor(col)
            painter.setPen(pen)
            painter.drawPolyline(pol)

    def draw_boxes(self, painter, row_id, row_data,  is_current_index):
        '''
        Draw traj worm trajectory.
        '''
        worm_index = int(row_data[self.worm_index_type])
        x = int(round(row_data['coord_x']))
        y = int(round(row_data['coord_y']))

        label_color = self.wlabC[int(row_data[self.label_type])]
        if not self.ui.is_color_features.isChecked():
            label_color = self.wlabC[int(row_data[self.label_type])]
        else:
            label_color = self._h_assign_feat_color(row_id)


        pen = QPen()
        pen.setColor(label_color)
        pen.setWidth(self.penwidth)
        painter.setPen(pen)
        painter.setFont(QFont('Decorative', self.fontsize))

        painter.drawText(x, y, str(worm_index))

        bb = row_data['roi_size']
        painter.drawRect(x - bb / 2, y - bb / 2, bb, bb)

        if is_current_index:

            b_size = bb//5
            offset = bb/2 - b_size
            painter.fillRect(x + offset, y + offset, b_size, b_size, QBrush(label_color))

    def draw_skeletons(self, painter, roi_id, row_data, is_current_index):
        if self.traj_worm_index_grouped is None:
            return
        if self.coordinates_fields is None:
            return
        worm_index = int(row_data[self.worm_index_type])
        skel_id = int(row_data['skeleton_id'])
        if self.coordinates_fields is None or skel_id < 0:
            return

        skel_dat = {}
        with tables.File(self.skeletons_file, 'r') as skel_file_id:
            # print(self.coordinates_group)
            # print(self.coordinates_fields)
            for ff, tt in self.coordinates_fields.items():
                field = self.coordinates_group + ff
                if field in skel_file_id:
                    dat = skel_file_id.get_node(field)[skel_id]
                    dat /= self.microns_per_pixel

                    if self.stage_position_pix is not None and self.stage_position_pix.size > 0:
                        #subtract stage motion if necessary
                        dat -= self.stage_position_pix[self.frame_number]

                    #dat[:, 0] = (dat[:, 0] - roi_corner[0] + 0.5) * c_ratio_x
                    #dat[:, 1] = (dat[:, 1] - roi_corner[1] + 0.5) * c_ratio_y

                else:
                    dat = np.full((1,2), np.nan)

                skel_dat[tt] = dat


        if 'is_good_skel' in row_data and row_data['is_good_skel'] == 0:
            skel_colors = BAD_SKEL_COLOURS
        else:
            skel_colors = GOOD_SKEL_COLOURS

        qPlg = {}
        for tt, dat in skel_dat.items():
            qPlg[tt] = QPolygonF()
            for p in dat:
                #do not add point if it is nan
                if p[0] == p[0]:
                    qPlg[tt].append(QPointF(*p))

        if not qPlg or len(qPlg['skeleton']) == 0:
            #all nan skeleton nothing to do here...
            return

        pen = QPen()
        pen.setWidth(0.5)

        # pen.setColor(QColor(col))
        # painter.setPen(pen)
        # painter.drawPolyline(pol_v)

        for k, pol_v in qPlg.items():
            color = skel_colors[k]
            pen.setColor(QColor(*color))
            painter.setPen(pen)
            painter.drawPolyline(pol_v)

        pen.setColor(Qt.black)
        painter.setBrush(Qt.white)
        painter.setPen(pen)

        radius = 2
        painter.drawEllipse(qPlg['skeleton'][0], radius, radius)
        painter.drawEllipse(QPointF(0,0), radius, radius)


class PlotCommunicator(FeatureReaderBase, ROIManager):
    def __init__(self, ui=''):
        super().__init__(ui)
        self.ui.pushButton_plot.setEnabled(False)
        self.ui.pushButton_plot.clicked.connect(self.show_plot)
        self.plotter = None

    def closePrev(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        self.closePrev()
        if self.timeseries_data is None:
            self.ui.pushButton_plot.setEnabled(False)
        else:
            self.ui.pushButton_plot.setEnabled(True)

    def show_plot(self):
        self.closePrev()

        self.plotter = PlotFeatures(self.skeletons_file,
                                   self.timeseries_data,
                                   self.traj_worm_index_grouped,
                                   self.time_units,
                                   self.xy_units,
                                   self.fps,
                                   parent = self)

        self.plotter.setWindowFlags(self.plotter.windowFlags() | Qt.WindowStaysOnTopHint)

        self.plotter.show()
        self.update_plot()

    def update_plot(self):
        if self.plotter:
            self.plotter.plot(self.current_worm_index, self.feat_column)

class MWTrackerViewer_GUI( MarkersDrawer, PlotCommunicator,
    FoodContourDrawer, BlobLabeler, IntensityLabeler, TrajectoryEditor, WellsDrawer):

    def __init__(self, ui='', argv=''):
        if not ui:
            super().__init__(Ui_MWTrackerViewer())
        else:
            super().__init__(ui)

        self.setWindowTitle("Multi-Worm Viewer")

        self.vfilename = '' if len(argv) <= 1 else argv[1]
        self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"
        self.results_dir = ''
        self.skeletons_file = ''
        self.worm_index_type = 'worm_index_manual'
        self.frame_data = None


        self.ui.comboBox_labelType.currentIndexChanged.connect(self.selectWormIndexType)

        self.ui.pushButton_save.clicked.connect(self.saveData)

        # select worm ROI when doubleclick a worm
        self.mainImage._canvas.mouseDoubleClickEvent = self.selectWorm
        self.mainImage._canvas.mouseRightClickEvent = self.toggleWellStatus

        self.ui.comboBox_ROI1.activated.connect(self.update_plot)
        self.ui.comboBox_ROI2.activated.connect(self.update_plot)

    def saveData(self):
        '''save data from manual labelling. pytables saving format is more convenient than pandas'''

        if os.name == 'nt':
            # I Windows the paths return by QFileDialog use / as the file
            # separation character. We need to correct it.
            for field_name in ['vfilename', 'skeletons_file']:
                setattr(
                    self, field_name, getattr(
                        self, field_name).replace(
                        '/', os.sep))
        has_skeletons_file = ((self.skeletons_file is not None)
                              and (self.skeletons_file != ''))
        if has_skeletons_file:
            save_modified_table(self.skeletons_file,
                                self.trajectories_data,
                                'trajectories_data')

        if self.is_fov_tosplit:
            if has_skeletons_file:
                self.fovsplitter.write_fov_wells_to_file(self.skeletons_file)
            else:
                warnings.warn('No skeletons file. Saving wells info in masked video')
                self.fid.close()
                self.fovsplitter.write_fov_wells_to_file(self.vfilename)
                # self.fid = tables.File(self.vfilename, 'r')
                self.updateVideoFile(self.vfilename)

        if has_skeletons_file:
            self.updateSkelFile(self.skeletons_file)



    def updateVideoFile(self, vfilename):
        super().updateVideoFile(vfilename)
        self.updateImage()

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)

        if self.trajectories_data is None:
            #empty file nothing to do here
            self.updateImage()
            return

        #correct the `worm_index_N` to the actual name `worm_index_manual`
        if 'worm_index_N' in self.trajectories_data:
            self.trajectories_data = self.trajectories_data.rename(
                columns={'worm_index_N': 'worm_index_manual'})

        #if this is really a trajectories_data not (_features.hdf5) add `worm_index_manual` if it does not exists
        if not 'worm_index_manual' in self.trajectories_data and not self.is_estimated_trajectories_data:
            self.trajectories_data['worm_label'] = self.wlab['U']
            self.trajectories_data['worm_index_manual'] = self.trajectories_data['worm_index_joined']

        #deactiate the save option if we are dealing with estimated data...
        self.ui.pushButton_save.setEnabled(not self.is_estimated_trajectories_data)


        #add this column if it does not exist
        if not 'has_skeleton' in self.trajectories_data:
            self.trajectories_data['has_skeleton'] = (
                self.trajectories_data['skeleton_id'] >= 0).astype(np.uint8)

        self.updateWormIndexTypeMenu()
        self.updateImage()

    def updateWormIndexTypeMenu(self):
        possible_indexes = [x.replace('worm_index_', '') for x in self.trajectories_data.columns if x.startswith('worm_index_')]
        assert len(set(possible_indexes)) == len(possible_indexes) #all indexes ending must be different

        menu_names = sorted([x + ' index' for x in possible_indexes])
        self.ui.comboBox_labelType.clear()
        self.ui.comboBox_labelType.addItems(menu_names)
        if 'manual' in possible_indexes:
            dd = self.ui.comboBox_labelType.findText('manual index')
            self.ui.comboBox_labelType.setCurrentIndex(dd);

        self.selectWormIndexType()

    def selectWormIndexType(self):
        index_option = self.ui.comboBox_labelType.currentText()

        if not index_option:
            return
        assert index_option.endswith(' index')
        self.worm_index_type = 'worm_index_' + index_option.replace(' index', '')

        # select between automatic and manual worm indexing and label
        if self.worm_index_type == 'worm_index_manual':
            self.label_type = 'worm_label'
            self.enable_trajectories_buttons(True)
            self.enable_label_buttons(True)
        else:
            self.label_type = 'auto_label'
            self.enable_trajectories_buttons(False)
            self.enable_label_buttons(False)

        #recalculate the grouped indexes
        self.traj_worm_index_grouped = self.trajectories_data.groupby(self.worm_index_type)

        self.updateImage()


    # update image
    def updateImage(self):
        if (self.image_group is None) and (self.isimgstore is False):
            return

        super(TrackerViewerAuxGUI, self).readCurrentFrame()

        # Update the time display
        if self.fps > 0:  # Ensure FPS is valid
            time_in_seconds = int(self.frame_number / self.fps)
            self.ui.time_display.setText(f"{time_in_seconds}")

        # read the data of the particles that exists in the frame
        self.frame_data = self.getFrameData(self.frame_number)

        #draw extra info only if the worm_index_type is valid
        if self.frame_data is not None and \
        self.worm_index_type in self.frame_data:
            #filter any -1 index
            self.frame_data = self.frame_data[self.frame_data[self.worm_index_type]>=0]
            if self.frame_data.size > 0:
                self.draw_worm_markers(self.frame_qimg)
                self.draw_food_contour(self.frame_qimg)

                self.updateROIs()

        else:
            self.clearROIs()
        # plot wells
        self.draw_wells(self.frame_qimg)

        # create the pixmap for the label
        self.mainImage.setPixmap(self.frame_qimg)

        self.display_intensity()

    def selectWorm(self, event):

        x = event.pos().x()
        y = event.pos().y()
        print(x,y)

        if self.frame_data is None or self.frame_data.size == 0:
            return

        R = (x - self.frame_data['coord_x'])**2 + \
            (y - self.frame_data['coord_y'])**2

        ind = R.idxmin()

        good_row = self.frame_data.loc[ind]
        if np.sqrt(R.loc[ind]) < good_row['roi_size']:
            self.current_roi.worm_index = int(good_row[self.worm_index_type])
            self.update_plot()

        self.updateImage()

    def toggleWellStatus(self, event):
        # abort if not multifov
        if self.is_fov_tosplit != True:
            return
        # event is for sure a right click or this does not get called
        x = event.pos().x()
        y = event.pos().y()
        # this will always return something. n/a if clicking outside a well
        well_name = self.fovsplitter.find_well_of_xy(x, y)[0].decode('utf-8')
        idx = self.fovsplitter.wells['well_name'] == str(well_name)
        self.fovsplitter.wells.loc[idx, 'is_good_well'] = \
            np.mod(self.fovsplitter.wells.loc[idx, 'is_good_well']+1, 2)
#        print(self.fovsplitter.wells)
        self.updateImage()

    def joinTraj(self):
        super().joinTraj()
        self.update_plot()

    def splitTraj(self):
        super().splitTraj()
        self.update_plot()

    def change_feature(self):
        super().change_feature()
        self.update_plot()

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main = MWTrackerViewer_GUI(argv=sys.argv)

    #mask_file = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/mutliworm_example/BRC20067_worms10_food1-10_Set2_Pos5_Ch2_02062017_121709.hdf5'
    #mask_file = '/Volumes/rescomp1/data/WormData/screenings/Pratheeban/First_Set/MaskedVideos/Old_Adult/16_07_22/W3_ELA_1.0_Ch1_22072016_131149.hdf5'
    #mask_file = '/Users/avelinojaver/Documents/GitHub/tierpsy-tracker/tests/data/AVI_VIDEOS/MaskedVideos/AVI_VIDEOS_1.hdf5'
#    mask_file = '/Users/avelinojaver/Documents/GitHub/tierpsy-tracker/tests/data/WT2/MaskedVideos/WT2.hdf5'
    mask_file = '/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/MaskedVideos/20191205/syngenta_screen_run1_bluelight_20191205_151104.22956805/metadata.hdf5'
    main.updateVideoFile(mask_file)

    main.show()
    sys.exit(app.exec_())
