import json
import os
from functools import partial

import cv2
import numpy as np
from tierpsy.analysis.compress.compressVideo import getROIMask, selectVideoReader, reduceBuffer
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, \
QFileDialog, QMessageBox

from tierpsy.gui.GetAllParameters import GetAllParameters, ParamWidgetMapper, save_params_json
from tierpsy.gui.GetMaskParams_ui import Ui_GetMaskParams
from tierpsy.gui.HDF5VideoPlayer import LineEditDragDrop, ViewsWithZoom, setChildrenFocusPolicy

from tierpsy.analysis.compress.BackgroundSubtractor import BackgroundSubtractorVideo, BackgroundSubtractorMasked
from tierpsy.helper.params.tracker_param import TrackerParams, default_param



class twoViewsWithZoom():
    '''
    Object useful to sincronize the pan/zoom of two images i.e. original and masked
    '''
    def __init__(self, view_full, view_mask):
        self.view_full = ViewsWithZoom(view_full)
        self.view_mask = ViewsWithZoom(view_mask)

        self.view_full._view.wheelEvent = self.zoomWheelEvent
        self.view_mask._view.wheelEvent = self.zoomWheelEvent

        # link scrollbars in the graphics view so if one changes the other
        # changes too
        self.hscroll_full = self.view_full._view.horizontalScrollBar()
        self.vscroll_full = self.view_full._view.verticalScrollBar()
        self.hscroll_mask = self.view_mask._view.horizontalScrollBar()
        self.vscroll_mask = self.view_mask._view.verticalScrollBar()
        dd = partial(self.chgScroll, self.hscroll_mask, self.hscroll_full)
        self.hscroll_mask.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.vscroll_mask, self.vscroll_full)
        self.vscroll_mask.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.hscroll_full, self.hscroll_mask)
        self.hscroll_full.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.vscroll_full, self.vscroll_mask)
        self.vscroll_full.valueChanged.connect(dd)

    def chgScroll(self, scroll_changed, scroll2change):
        scroll2change.setValue(scroll_changed.value())

    def zoomWheelEvent(self, event):
        self.view_full.zoomWheelEvent(event)
        self.view_mask.zoomWheelEvent(event)

    def zoom(self, zoom_direction):
        self.view_full.zoom(zoom_direction)
        self.view_mask.zoom(zoom_direction)

    def zoomFitInView(self):
        self.view_full.zoomFitInView()
        self.view_mask.zoomFitInView()

    def setPixmap(self, qimage_full, qimage_mask):
        # for both maps to have the same size
        v_height = min(
            self.view_full._view.height(),
            self.view_mask._view.height())
        v_width = min(
            self.view_full._view.width(),
            self.view_mask._view.width())

        self.view_full._view.resize(v_width, v_height)
        self.view_mask._view.resize(v_width, v_height)

        self.view_full.setPixmap(qimage_full)
        self.view_mask.setPixmap(qimage_mask)

    def cleanCanvas(self):
        self.view_full.cleanCanvas()
        self.view_mask.cleanCanvas()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mousePressed = True
            self._dragPos = event.pos()
            event.accept()

            self.view_full._view.setCursor(Qt.ClosedHandCursor)
            self.view_mask._view.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mousePressed:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos

            dx = self.view_full._view.horizontalScrollBar().value() - diff.x()
            dy = self.view_full._view.verticalScrollBar().value() - diff.y()
            self.view_full._view.horizontalScrollBar().setValue(dx)
            self.view_full._view.verticalScrollBar().setValue(dy)

            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.view_full._view.setCursor(Qt.OpenHandCursor)
            self.view_mask._view.setCursor(Qt.OpenHandCursor)
            self._mousePressed = False

class ParamsGUI(QMainWindow):
    def __init__(self, default_videos_dir='', scripts_dir=''):
        self.json_file = ''
        self.json_param = default_param.copy()
        super().__init__()
        
        # Set up the user interface from Designer.
        self.ui = Ui_GetMaskParams()
        self.ui.setupUi(self)
        self._link_slider_spinbox()
        self.mapper = ParamWidgetMapper(self.ui)

        self.ui.pushButton_saveParam.clicked.connect(self.saveParamFile)
        self.ui.pushButton_paramFile.clicked.connect(self.getParamFile)
        
        LineEditDragDrop(self.ui.lineEdit_paramFile, self.updateParamFile, os.path.isfile)

        self.ui.pushButton_moreParams.clicked.connect(self.getMoreParams)


    def _link_slider_spinbox(self):
        '''
        Link a given slider to a spinbox so when the value of one changes the other changes too.
        '''
        def _single_link(slider, spinbox, connect_func):
            slider.sliderReleased.connect(connect_func)
            spinbox.editingFinished.connect(connect_func)
            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)

        for field in ['mask_min_area', 'mask_max_area', 'thresh_block_size', 'thresh_C', 'dilation_size']:
            slider = getattr(self.ui, 'horizontalSlider_' + field)
            spinbox = getattr(self.ui, 'p_' + field)
            _single_link(slider, spinbox, self.updateMask)

    def saveParamFile(self):
        self.json_file = self.ui.lineEdit_paramFile.text()
        
        if not self.json_file:
            QMessageBox.critical(
                self,
                'No parameter file name given.',
                'No parameter file name given. Please select name using the "Parameters File" button',
                QMessageBox.Ok)
            return

        if os.path.exists(self.json_file):

            reply = QMessageBox.question(
                self,
                'Message',
                '''The parameters file already exists. Do you want to overwrite it?
            If No the parameters in the existing file will be used instead of the values displayed.''',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)

            if reply == QMessageBox.No:
                return

        dname = os.path.dirname(self.json_file)
        if not os.path.isdir(dname):
            QMessageBox.critical(
                self,
                'Invalid directory.',
                'The directory in the file path name does not exists. Please be sure the path name is correct.',
                QMessageBox.Ok)
            return


        # save data into the json file
        #update before saving
        self._update_json_params()

        try:
            save_params_json(self.json_file, self.json_param)
        except OSError:
            QMessageBox.critical(
                self,
                'Invalid file name.',
                'I was not able to save the file. Please be sure the path name is correct.',
                QMessageBox.Ok)
            return


    def _update_json_params(self):
        '''
        update json_params field from the gui vaues.
        '''
        for param_name in self.mapper:
            assert param_name in self.json_param
            self.json_param[param_name] = self.mapper[param_name]

    def _update_gui_params(self, json_param):
        checkit = (json_param['mask_bgnd_buff_size']>0) and (json_param['mask_bgnd_frame_gap']>0)

        self.ui.checkBox_is_bgnd_subtraction.setChecked(checkit)
        # set correct widgets to the values given in the json file
        for param_name in json_param:
            if param_name in self.mapper:
               self.mapper[param_name] = json_param[param_name]

    def getMoreParams(self):
        self._update_json_params()
        allparamGUI = GetAllParameters(self.json_param)
        
        #the gui must have updated the dictioary in json_param 
        allparamGUI.exec_()
        self._update_gui_params(self.json_param)

    # file dialog to the the hdf5 file
    def getParamFile(self):
        json_dir = os.path.dirname(self.ui.lineEdit_paramFile.text())
        json_file, _ = QFileDialog.getOpenFileName(self, "Find parameters file", json_dir, "JSON files (*.json);; All (*)")
        if json_file:
            self.updateParamFile(json_file)

    def updateParamFile(self, json_file):
        # set the widgets with the default parameters, in case the parameters are not given
        # by the json file.
        if os.path.exists(json_file):
            try:
                params = TrackerParams(json_file)
                json_param = params.p_dict

            except (OSError, UnicodeDecodeError, json.decoder.JSONDecodeError):
                QMessageBox.critical(
                    self,
                    'Cannot read parameters file.',
                    "Cannot read parameters file. Try another file",
                    QMessageBox.Ok)
                return
        else:
            json_param = default_param.copy()

        

        self._update_gui_params(json_param)
        self.json_file = json_file
        self.json_param = json_param
        self.ui.lineEdit_paramFile.setText(self.json_file)
        
    
class GetMaskParams_GUI(ParamsGUI):

    def __init__(self, default_videos_dir='', scripts_dir=''):
        super().__init__()
        
        self.video_file = ''
        
        self.Ibuff = np.zeros(0)
        self.Ifull = np.zeros(0)
        self.IsubtrB = np.zeros(0)
        self.bgnd_subtractor = None
        self.vid = None
        self.frame_number = 0


        #remove tabs for the moment. I need to fix this it later
        self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.indexOf(self.ui.tab_mask))
        self.tab_keys = dict(mask=self.ui.tabWidget.indexOf(self.ui.tab_mask),
                         bgnd=self.ui.tabWidget.indexOf(self.ui.tab_bgnd))
        
        self.ui.p_keep_border_data.stateChanged.connect(self.updateMask)
        self.ui.p_is_light_background.stateChanged.connect(self.updateBgnd)
        self.ui.pushButton_update_bgnd.clicked.connect(self.updateBgnd)

        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.pushButton_next.clicked.connect(self.getNextChunk)

        #self.ui.checkBox_subtractBackground.clicked.connect(self.updateMask)
        self.ui.checkBox_is_bgnd_subtraction.stateChanged.connect(self.updateCheckedBgndSubtr)
        self.updateCheckedBgndSubtr()
        self.ui.tabWidget.currentChanged.connect(self.updateROIs)

        self.videos_dir = default_videos_dir
        if not os.path.exists(self.videos_dir):
            self.videos_dir = ''
        

        # setup image view as a zoom
        self.twoViews = twoViewsWithZoom(
            self.ui.graphicsView_full,
            self.ui.graphicsView_mask)

        LineEditDragDrop(self.ui.lineEdit_video, self.updateVideoFile, os.path.isfile)
        
        # make sure the childrenfocus policy is none in order to be able to use
        # the arrow keys
        setChildrenFocusPolicy(self, Qt.ClickFocus)
        
        self.is_play = False
        self.ui.pushButton_play.clicked.connect(self.playVideo)
        self.timer = QTimer()
        self.timer.timeout.connect(self.getNextChunk)

    
    def keyPressEvent(self, event):
        if self.vid is not None:
            # break no file open, nothing to do here
            return
        key = event.key()
        if key == Qt.Key_Minus:
            self.twoViews.zoom(-1)
        elif key == Qt.Key_Plus:
            self.twoViews.zoom(1)

        else:
            QMainWindow.keyPressEvent(self, event)

    def playVideo(self):
        if self.vid is None:
            return
        if not self.is_play:
            self.startPlay()
        else:
            self.stopPlay()

    def startPlay(self):
        
        fps = self.mapper['expected_fps']
        compression_buff = self.mapper['compression_buff']
        if fps> 0 and compression_buff > 0:
            fps_n = fps/compression_buff
            freq = round(1000 / fps_n)
        else:
            freq = 30

        self.timer.start()
        self.is_play = True
        self.ui.pushButton_play.setText('Stop')

    def stopPlay(self):
        self.timer.stop()
        self.is_play = False
        self.ui.pushButton_play.setText('Play')


    def closeEvent(self, event):
        if self.vid is not None:
            self.vid.release()
        super().closeEvent(event)

    # update image if the GUI is resized event
    def resizeEvent(self, event):
        self.updateROIs()
        self.twoViews.zoomFitInView()

    
    # file dialog to the the hdf5 file
    def getVideoFile(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Find video file", self.videos_dir, "All files (*)")
        self.updateVideoFile(video_file)

    def _start_video(self, video_file):
        vid = selectVideoReader(video_file)
        if vid.width == 0 or vid.height == 0:
            raise ValueError
        else:
            if self.vid is not None:
                self.vid.release()
            self.vid, self.im_width, self.im_height = vid, vid.width, vid.height
            self.bgnd_subtractor = None #make sure this get restarted when a new file is initialized
            self.frame_number = 0
        
    def updateVideoFile(self, video_file):
        if video_file and os.path.exists(video_file):
            try:
                self._start_video(video_file)
            except (OSError, ValueError, IOError):
                QMessageBox.critical(
                    self,
                    'Cannot read video file.',
                    "Cannot read video file. Try another file",
                    QMessageBox.Ok)
                return

            self.video_file = video_file
            self.videos_dir = os.path.split(self.video_file)[0]
            self.ui.lineEdit_video.setText(self.video_file)

            # read json file
            json_file = self.video_file.rpartition('.')[0] + '.json'
            self.updateParamFile(json_file)

            # get next chuck
            self.getNextChunk()

            # fit the image to the canvas size
            self.twoViews.zoomFitInView()

            
            #set the valid limits for the block size
            max_block = min(self.Ifull.shape)
            min_block = 3
            self.ui.p_thresh_block_size.setRange(min_block, max_block)

    def updateReducedBuff(self):
        if self.Ibuff.size > 0:
            #update the image used to create the mask
            
            is_light_background = self.mapper['is_light_background']
            if self.ui.checkBox_is_bgnd_subtraction.isChecked() and self.bgnd_subtractor is not None:
                Ibuff_b = self.bgnd_subtractor.apply(self.Ibuff, self.frame_number)
            else:
                Ibuff_b = self.Ibuff
                
            self.Imin = reduceBuffer(Ibuff_b, is_light_background)
            
            self.updateMask()

    def getNextChunk(self):
        if self.vid is not None:
            # read the buffsize before getting the next chunk
            self.buffer_size =  self.mapper['compression_buff']
            if self.buffer_size <= 0:
                self.buffer_size  = 1

            self.Ibuff = np.zeros(
                (self.buffer_size,
                 self.im_height,
                 self.im_width),
                dtype=np.uint8)

            
            tot = 0
            for ii in range(self.buffer_size):
                # get video frame, stop program when no frame is retrive (end
                # of file)
                ret, image = self.vid.read()
                if ret == 0:
                    if ii == 0:
                        #restart video
                        self._start_video(self.video_file) 
                        ret, image = self.vid.read() #try to read again, if you cannot again just quit
                        if ret == 0: 
                            break
                    else:
                        break
                
                
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                image = np.ascontiguousarray(image)
                self.Ibuff[ii] = image

                tot += 1
            if tot == 0:
                return
            elif tot < self.buffer_size:
                self.Ibuff = self.Ibuff[:tot]

            self.frame_number += tot
            self.ui.spinBox_frame_number.setValue(self.frame_number)
            self.Ifull = self.Ibuff[0].copy()
            
            
            self.updateBgnd()
            

    def _numpy2qimage(self, im_ori):
        return QImage(im_ori.data, im_ori.shape[1], im_ori.shape[0],
                      im_ori.data.strides[0], QImage.Format_Indexed8)

    def updateROIs(self):
        #useful for resizing events
        if self.Ifull.size == 0:
            self.twoViews.cleanCanvas()
        else:
            cur = self.ui.tabWidget.currentIndex()
            if cur == self.tab_keys['mask']:
                I1, I2 = self.Ifull, self.Imask
            elif cur == self.tab_keys['bgnd']:
                I1 = self.Ifull
                I2 = np.zeros_like(self.IsubtrB)
                cv2.normalize(self.IsubtrB,I2,0,255,cv2.NORM_MINMAX)
            else:
                I1, I2 = self.Ifull, self.Ifull

            qimage_roi1 = self._numpy2qimage(I1)
            qimage_roi2 = self._numpy2qimage(I2)
            self.twoViews.setPixmap(qimage_roi1, qimage_roi2)

    def updateMask(self):
        if self.Ifull.size == 0:
            return
        # read parameters used to calculate the mask
        roi_mask_params_str = ['mask_max_area', 'mask_min_area', 'thresh_block_size', 'thresh_C', 
        'dilation_size', 'keep_border_data', 'is_light_background']
        mask_param = {x:self.mapper[x] for x in roi_mask_params_str}
        
        mask_param['max_area'] = mask_param.pop('mask_max_area')
        mask_param['min_area'] = mask_param.pop('mask_min_area')

        mask = getROIMask(self.Imin.copy(), **mask_param)
        self.Imask = mask * self.Ifull
        self.updateROIs()
        
    def _updateBgnd(self):
        if self.vid is None:
            return 

        if self.bgnd_subtractor is None:
            #try to calculate the background if the parameters are corrected
            keys = ['is_light_background', 'mask_bgnd_buff_size', 'mask_bgnd_frame_gap']
            kwargs = {x.replace('mask_bgnd_', ''):self.mapper[x] for x in keys}

            if kwargs['buff_size'] >0 and kwargs['frame_gap']>0:
                if not self.video_file.endswith('.hdf5'):
                    self.bgnd_subtractor = BackgroundSubtractorVideo(self.video_file, **kwargs)
                else:
                   self.bgnd_subtractor = BackgroundSubtractorMasked(self.video_file, **kwargs)

        #if the background substraction is checked and it was calculated correctly update the background
        if self.ui.checkBox_is_bgnd_subtraction.isChecked() and \
        self.Ifull.size > 0 and self.bgnd_subtractor is not None:
            self.IsubtrB = self.bgnd_subtractor.subtract_bgnd(self.Ifull)
            #debug!!!
            #self.IsubtrB = self.bgnd_subtractor.bgnd.astype(np.uint8)
        else:
            self.IsubtrB = self.Ifull

        self.updateReducedBuff()

    def updateBgnd(self):
        self.bgnd_subtractor = None
        self._updateBgnd()

    def updateCheckedBgndSubtr(self):
        valid = self.ui.checkBox_is_bgnd_subtraction.isChecked()
        self.ui.p_mask_bgnd_frame_gap.setEnabled(valid)
        self.ui.p_mask_bgnd_buff_size.setEnabled(valid)
        self._updateBgnd()
    
    
    def updateParamFile(self, json_file):
        super().updateParamFile(json_file)
        self.updateMask()

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    ui = GetMaskParams_GUI()
    ui.show()

    sys.exit(app.exec_())
