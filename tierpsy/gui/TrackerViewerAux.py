import json
import os
import re
import sys

import numpy as np
import pandas as pd
import tables
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import (
    QPixmap, QImage, QPolygonF, QPen, QPainter, QColor)
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QWidget, QPushButton)

from tierpsy.gui.HDF5VideoPlayer import HDF5VideoPlayerGUI, LineEditDragDrop
from tierpsy.gui.TrackerViewerAux_ui import Ui_TrackerViewerAux
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask
from tierpsy.analysis.ske_create.segWormPython.mainSegworm import getSkeleton
from tierpsy.helper.params import read_unit_conversions
from tierpsy.helper.misc import WLAB

BAD_SKEL_COLOURS = dict(
    skeleton = (102, 0, 0),
    contour_side1 = (102, 0, 0),
    contour_side2 = (102, 0, 0)
    )
GOOD_SKEL_COLOURS = dict(
    skeleton = (27, 158, 119),
    contour_side1 = (217, 95, 2),
    contour_side2 = (231, 41, 138)
    )

#%%
def _estimate_trajectories_data(ow_feat_file, timestamp, microns_per_pixel, stage_position_pix = None):
    '''
    I want to estimate the trajectorires_data table from the features_timeseries in the old features
    so I can used them with the viewer.
    '''

    stamp2frame = {f:i for i, f in list(enumerate(timestamp))[::-1]}



    with pd.HDFStore(ow_feat_file, 'r') as fid:
       features_timeseries = fid['features_timeseries']

    features_timeseries['frame_number'] = features_timeseries['timestamp'].map(stamp2frame)
    features_timeseries = features_timeseries.dropna(subset=['frame_number'])



    df = []
    with tables.File(ow_feat_file, 'r') as fid:
        skels = fid.get_node('/coordinates/skeletons')
        mid_ind = skels.shape[1]//2
        for w_ind, w_data in features_timeseries.groupby('worm_index'):
            midbody_coords = skels[w_data.index, mid_ind, :]

            new_data = pd.DataFrame(midbody_coords, columns=['coord_x', 'coord_y'], index=w_data.index)/microns_per_pixel
            new_data['worm_index_joined'] = w_ind
            new_data['frame_number'] = features_timeseries['frame_number']
            new_data['roi_size'] = w_data['length'].max()/microns_per_pixel
            new_data['threshold'] = 255
            new_data['area'] = w_data['area']/microns_per_pixel**2
            new_data['was_skeletonized'] = True
            new_data['skeleton_id'] = w_data.index

            new_data = new_data.drop_duplicates('frame_number')

            if stage_position_pix is not None:
                #subtract stage motion if necessary
                ss = stage_position_pix[new_data['frame_number'].astype(np.int)]
                new_data['coord_x'] -= ss[:, 0]
                new_data['coord_y'] -= ss[:, 1]



            df.append(new_data)
    trajectories_data = pd.concat(df)

    return trajectories_data
#%%

class TrackerViewerAuxGUI(HDF5VideoPlayerGUI):

    def __init__(self, ui=''):
        if not ui:
            super().__init__(Ui_TrackerViewerAux())
        else:
            super().__init__(ui)

        self.results_dir = ''
        self.skeletons_file = ''
        self.trajectories_data = None
        self.traj_time_grouped = None
        self.traj_worm_index_grouped = None

        self.skel_dat = {}

        self.is_estimated_trajectories_data = False
        self.frame_save_interval = 1

        self.microns_per_pixel = 1.
        self.fps = 1.
        self.time_units = ''
        self.xy_units = ''

        self.stage_position_pix = None
        self.coordinates_group = '/'
        self.coordinates_fields = None

        self.ui.pushButton_skel.clicked.connect(self.getSkelFile)
        LineEditDragDrop(
            self.ui.lineEdit_skel,
            self.newSkelFileEvent,
            os.path.isfile)

    def getSkelFile(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self, 'Select file with the worm skeletons and/or features', self.results_dir, "Results files (*_featuresN.hdf5 *features.hdf5 *_skeletons.hdf5);; All files (*)")

        if not os.path.exists(selected_file):
            return

        self.newSkelFileEvent(selected_file)

    def newSkelFileEvent(self, selected_file):
        """
        Given a new skeletons file name, take decisions on what to do based
        also on the current value of the masked video field
        new results, no mask => find mask, open both
        new results, already a mask, they match => open new results
        new results, already a mask, they don't match => ask user
            - do nothing
            - find matching masked video
            - just open new skeletons
        """

        # both if there is or if there isn't an opened masked video
        # I'll need to find the matching one
        matching_maskname = results2maskedvideo(selected_file)
        matching_imgstorename = results2imgstore(selected_file)

        # skel, but no mask:
        if (self.fid is None) and (self.isimgstore is False):
            # if the masked file, or the raw video, exist, open them
            self.updateSkelAndVideoFileFromSkelName(selected_file)

        # already an opened masked videos file
        elif self.vfilename in [matching_imgstorename, matching_maskname]:
            # check if open video (could be raw video) matches results
            # print("results match opened video, load results")
            self.updateSkelFile(selected_file)

        # masked video does not match
        # ask user if to ignore mismatch, or reload video as well
        else:
            # print("results does not match opened video")
            msgbox = QMessageBox(self)
            msgbox.setIcon(QMessageBox.Question)
            msgbox.setText(
                ("The results file does not match the open video file\n"
                    "Select an option:")
                )
            # cancel: do nothing
            # No: just load the results
            # Yes: reload both
            open_results_anyway_msg = 'Open results anyway'
            find_vid_open_both_msg = 'Open with matching video'
            msgbox.addButton(QMessageBox.Cancel)
            msgbox.addButton(
                QPushButton(open_results_anyway_msg), QMessageBox.NoRole)
            msgbox.addButton(
                QPushButton(find_vid_open_both_msg), QMessageBox.YesRole)
            msgbox.setDefaultButton(QMessageBox.Cancel)
            msgbox.exec_()
            qres = msgbox.clickedButton()
            if (qres is None) or (qres.text() == 'Cancel'):
                # print("button is None or Cancel, do nothing")
                pass
            elif qres.text() == open_results_anyway_msg:
                # print("ignore mismatch, open results")
                self.updateSkelFile(selected_file)
            elif qres.text() == find_vid_open_both_msg:
                # print("find matching video, open both")
                self.updateSkelAndVideoFileFromSkelName(selected_file)
            else:
                print(qres)
                print(qres.text())
                print('This should never be triggered')

        return

    def updateSkelAndVideoFileFromSkelName(self, selected_skelfile):
        # find the matching masked and raw video
        matching_maskname = results2maskedvideo(selected_skelfile)
        matching_imgstorename = results2imgstore(selected_skelfile)
        # select the one that exists. if both exist, use masked
        matching_vfilename = None
        for fname in [matching_imgstorename, matching_maskname]:
            if os.path.exists(fname):
                matching_vfilename = fname
        # open the existing video and results
        if matching_vfilename is not None:
            self.updateVideoFile(matching_vfilename)
            self.updateSkelFile(selected_skelfile)
        else:  # otherwise error out
            err_msg = (
                "Could not find matching video at\n"
                f"{matching_maskname}\nnor at\n{matching_imgstorename}\n"
                "Try selecting a video file instead"
                )
            QMessageBox.critical(self, '', err_msg, QMessageBox.Ok)

        return

    def updateSkelFile(self, selected_file, dflt_skel_size = 5):
        self.trajectories_data = None
        self.traj_time_grouped = None
        self.traj_worm_index_grouped = None
        self.skel_dat = {}

        self.skeletons_file = selected_file
        self.ui.lineEdit_skel.setText(self.skeletons_file)
        try:
            #find were to read the skeletons and pixel2microns transforming factor
            self.stage_position_pix =  None

            #read units data
            fps_out, microns_per_pixel_out, _ = read_unit_conversions(self.skeletons_file)
            self.fps, _, self.time_units = fps_out
            self.microns_per_pixel, self.xy_units = microns_per_pixel_out


            with tables.File(self.skeletons_file, 'r') as skel_file_id:

                if '/coordinates' in skel_file_id:

                    self.coordinates_group = '/coordinates/'
                    self.coordinates_fields = {
                        'dorsal_contours':'contour_side2',
                        'skeletons':'skeleton',
                        'ventral_contours':'contour_side1'
                    }
                    if self.fid is not None:
                        if (not self.isimgstore) and ('/stage_position_pix' in self.fid):
                            self.stage_position_pix = self.fid.get_node('/stage_position_pix')[:]

                else:
                    self.microns_per_pixel = 1.
                    self.coordinates_group = '/'

                    self.coordinates_fields = {
                        'contour_side1':'contour_side1',
                        'skeleton':'skeleton',
                        'contour_side2':'contour_side2'
                    }
                self.coordinates_fields = {k:v for k,v in self.coordinates_fields.items() if (self.coordinates_group + k) in skel_file_id}

            #read trajectories data, and other useful factors
            with pd.HDFStore(self.skeletons_file, 'r') as ske_file_id:
                if 'trajectories_data' in ske_file_id:
                    self.trajectories_data = ske_file_id['/trajectories_data']
                    self.is_estimated_trajectories_data = False
                else:
                    timestamp = [np.nan]
                    if '/timestamp/raw' in self.fid:
                        timestamp = self.fid.get_node('/timestamp/raw')[:]


                    if np.any(np.isnan(timestamp)):
                        tot = self.fid.get_node('/mask').shape[0]
                        timestamp = np.arange(tot)


                    self.trajectories_data = _estimate_trajectories_data(self.skeletons_file, timestamp, self.microns_per_pixel, self.stage_position_pix)
                    self.is_estimated_trajectories_data = True

                #group data
                self.traj_time_grouped = self.trajectories_data.groupby('frame_number')
                self.traj_worm_index_grouped = self.trajectories_data.groupby('worm_index_joined')

                # read the size of the structural element used in to calculate
                # the mask
                if '/provenance_tracking/int_ske_orient' in ske_file_id:
                    prov_str = ske_file_id.get_node(
                        '/provenance_tracking/ske_create').read()
                    func_arg_str = json.loads(
                        prov_str.decode("utf-8"))['func_arguments']
                    strel_size = json.loads(func_arg_str)['strel_size']
                    if isinstance(strel_size, (list, tuple)):
                        strel_size = strel_size[0]

                    self.strel_size = strel_size
                else:
                    # use default
                    self.strel_size = dflt_skel_size



        except (IOError, KeyError, tables.exceptions.HDF5ExtError, tables.exceptions.NoSuchNodeError):
            self.trajectories_data = None
            self.traj_time_grouped = None
            self.skel_dat = {}

        #here i am updating the image
        self.ui.spinBox_frame.setValue(0)

    def updateImGroup(self, h5path):
        super().updateImGroup(h5path)
        try:
            self.frame_save_interval = int(self.image_group._v_attrs['save_interval'])
        except:
            self.frame_save_interval = 1

    def updateVideoFile(self, vfilename, possible_ext = ['_featuresN.hdf5', '_features.hdf5', '_skeletons.hdf5']):
        super().updateVideoFile(vfilename)
        if (self.image_group is None) and (self.isimgstore is False):
            return

        #find if it is a fluorescence image
        if (self.isimgstore is True) or ('is_light_background' not in self.image_group._v_attrs):
            self.is_light_background = 1
        else:
            self.is_light_background = self.image_group._v_attrs['is_light_background']

        if (self.fid is not None) and ('/trajectories_data' in self.fid):
            self.skeletons_file = vfilename
        else:
            videos_dir, basename = os.path.split(vfilename)
            basename = os.path.splitext(basename)[0]

            self.skeletons_file = ''
            self.results_dir = ''

            possible_dirs = [
                videos_dir, videos_dir.replace(
                    'MaskedVideos', 'Results').replace(
                        'RawVideos', 'Results'), os.path.join(
                            videos_dir, 'Results')]

            for new_dir in possible_dirs:
                for ext_p in possible_ext:
                    new_skel_file = os.path.join(new_dir, basename + ext_p)
                    if os.path.exists(new_skel_file):
                        self.skeletons_file = new_skel_file
                        self.results_dir = new_dir
                        break

        self.updateSkelFile(self.skeletons_file)

    def getFrameData(self, frame_number):
        try:
            if not isinstance(self.traj_time_grouped, pd.core.groupby.DataFrameGroupBy):
                raise KeyError

            frame_data = self.traj_time_grouped.get_group(self.frame_save_interval*frame_number)
            return frame_data

        except KeyError:
            return None

    def drawSkelResult(self, img, qimg, row_data, isDrawSkel,
        roi_corner=(0,0), read_center=True):


        if isDrawSkel and isinstance(row_data, pd.Series):
            if 'has_skeleton' in row_data and row_data['has_skeleton'] == 0:
                self.drawThreshMask(
                    img,
                    qimg,
                    row_data,
                    read_center=read_center)
            else:
                self.drawSkel(
                    img,
                    qimg,
                    row_data,
                    roi_corner = roi_corner
                    )

        return qimg

    def drawSkel(self, worm_img, worm_qimg, row_data, roi_corner=(0, 0)):
        if not self.skeletons_file or \
        self.trajectories_data is None or \
        worm_img.size == 0 or \
        worm_qimg is None or \
        self.coordinates_fields is None:
            return

        c_ratio_y = worm_qimg.width() / worm_img.shape[1]
        c_ratio_x = worm_qimg.height() / worm_img.shape[0]

        skel_id = int(row_data['skeleton_id'])
        skel_dat = {}


        with tables.File(self.skeletons_file, 'r') as ske_file_id:
            for ff, tt in self.coordinates_fields.items():
                field = self.coordinates_group + ff
                if field in ske_file_id:
                    dat = ske_file_id.get_node(field)[skel_id]
                    dat /= self.microns_per_pixel

                    if self.stage_position_pix is not None and self.stage_position_pix.size > 0:
                        #subtract stage motion if necessary
                        dat -= self.stage_position_pix[self.frame_number]

                    dat[:, 0] = (dat[:, 0] - roi_corner[0] + 0.5) * c_ratio_x
                    dat[:, 1] = (dat[:, 1] - roi_corner[1] + 0.5) * c_ratio_y

                else:
                    dat = np.full((1,2), np.nan)

                skel_dat[tt] = dat

        if 'is_good_skel' in row_data and row_data['is_good_skel'] == 0:
            skel_colors = BAD_SKEL_COLOURS
        else:
            skel_colors = GOOD_SKEL_COLOURS

        self._drawSkel(worm_qimg, skel_dat, skel_colors = skel_colors)


    def _drawSkel(self, worm_qimg, skel_dat, skel_colors = GOOD_SKEL_COLOURS):

        qPlg = {}
        for tt, dat in skel_dat.items():
            qPlg[tt] = QPolygonF()
            for p in dat:
                #do not add point if it is nan
                if p[0] == p[0]:
                    qPlg[tt].append(QPointF(*p))


        if not qPlg or len(qPlg['skeleton']) == 0:
            return

        pen = QPen()
        pen.setWidth(1)

        painter = QPainter()
        painter.begin(worm_qimg)

        for k, pol_v in qPlg.items():
            color = skel_colors[k]
            pen.setColor(QColor(*color))
            painter.setPen(pen)
            painter.drawPolyline(pol_v)

        pen.setColor(Qt.black)
        painter.setBrush(Qt.white)
        painter.setPen(pen)

        radius = 3
        painter.drawEllipse(qPlg['skeleton'][0], radius, radius)
        painter.drawEllipse(QPointF(0,0), radius, radius)

        painter.end()

    def drawThreshMask(self, worm_img, worm_qimg, row_data, read_center=True):
        #in very old versions of the tracker I didn't save the area in trajectories table,
        #let's assign a default value to deal with this cases
        if 'area' in row_data:
            min_blob_area = row_data['area'] / 2
        else:
            min_blob_area = 10

        c1, c2 = (row_data['coord_x'], row_data[
                  'coord_y']) if read_center else (-1, -1)

        worm_mask, worm_cnt, _ = getWormMask(worm_img, row_data['threshold'], strel_size=self.strel_size,
                                      roi_center_x=c1, roi_center_y=c2, min_blob_area=min_blob_area,
                                      is_light_background = self.is_light_background)


        worm_mask = QImage(
            worm_mask.data,
            worm_mask.shape[1],
            worm_mask.shape[0],
            worm_mask.strides[0],
            QImage.Format_Indexed8)
        worm_mask = worm_mask.convertToFormat(
            QImage.Format_RGB32, Qt.AutoColor)
        worm_mask = QPixmap.fromImage(worm_mask)

        worm_mask = worm_mask.createMaskFromColor(Qt.black)
        p = QPainter(worm_qimg)
        p.setPen(QColor(0, 204, 102))
        p.drawPixmap(worm_qimg.rect(), worm_mask, worm_mask.rect())

        if False:
            #test skeletonization
            skeleton, ske_len, cnt_side1, cnt_side2, cnt_widths, cnt_area = \
                getSkeleton(worm_cnt, np.zeros(0), 49)
            for cnt in skeleton, cnt_side1, cnt_side2:
                p.setPen(Qt.black)
                polyline = QPolygonF()
                for point in cnt:
                    polyline.append(QPointF(*point))
                p.drawPolyline(polyline)

        p.end()


def results2maskedvideo(results_path):
    ending_regexp = r"\_(?:featuresN|features|skeletons)\.hdf5$"
    maskedvideo_path = re.sub(ending_regexp, '.hdf5', results_path)
    subfolder_regexp = r"(.*)/Results[^/]*/"
    maskedvideo_path = re.sub(
        subfolder_regexp, r'\1/MaskedVideos/', maskedvideo_path)
    return maskedvideo_path

def results2imgstore(results_path):
    ending_regexp = r"\_(?:featuresN|features|skeletons)\.hdf5$"
    maskedvideo_path = re.sub(ending_regexp, '.yaml', results_path)
    subfolder_regexp = r"(.*)/Results[^/]*/"
    maskedvideo_path = re.sub(
        subfolder_regexp, r'\1/RawVideos/', maskedvideo_path)
    return maskedvideo_path


def test_results2maskedvideo():
    assert results2maskedvideo('/evgeny/Results/20190808/Results/metadata_featuresN.hdf5') == '/evgeny/Results/20190808/MaskedVideos/metadata.hdf5'
    assert results2maskedvideo('/evgeny/Results/20190808/metadata_featuresN.hdf5') == '/evgeny/MaskedVideos/20190808/metadata.hdf5'
    assert results2maskedvideo('/evgeny/Results_NN/20190808/metadata_featuresN.hdf5') == '/evgeny/MaskedVideos/20190808/metadata.hdf5'
    assert results2maskedvideo('/evgeny/Results/20190808/Results/metadata_skeletons.hdf5') == '/evgeny/Results/20190808/MaskedVideos/metadata.hdf5'
    assert results2maskedvideo('/evgeny/Results/20190808/metadata_skeletons.hdf5') == '/evgeny/MaskedVideos/20190808/metadata.hdf5'
    assert results2maskedvideo('/evgeny/Results_NN/20190808/metadata_skeletons.hdf5') == '/evgeny/MaskedVideos/20190808/metadata.hdf5'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = TrackerViewerAuxGUI()
    ui.show()
    sys.exit(app.exec_())
    # test_results2video()

