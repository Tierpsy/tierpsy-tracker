from tierpsy.gui.TrackerViewerAux import TrackerViewerAuxGUI
from tierpsy.helper.misc import remove_ext

from collections import OrderedDict 
import tables
import pandas as pd

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#plt.style.use(['default', 'fast'])
#plt.style.use(['ggplot', 'fast'])
#plt.style.use(['fivethirtyeight', 'fast'])
plt.style.use(['seaborn', 'fast'])

class PlotFeatures(QDialog):
    def __init__(self, 
                features_file = '',
                timeseries_data = None,
                traj_worm_index_grouped = None,
                time_units = None,
                xy_units = None,
                fps = None,
                parent = None):
        
        super().__init__(parent)
        
        self.plot_funcs = OrderedDict ([
            ('Single Trajectory, Time Series', self._plot_single_timeseries),
            ('All Trajectories, Time Series', self._plot_all_timeseries),
            ('Single Trajectory, Histogram', self._plot_single_histogram),
            ('All Trajectories, Histogram', self._plot_all_histogram)
        ])

        self.worm_index = None
        self.feature = None
        self.ts_bin = 5

        self.timeseries_data = timeseries_data
        self.traj_worm_index_grouped = traj_worm_index_grouped
        self.time_units = time_units
        self.xy_units = xy_units
        self.fps = fps
        self.root_file = remove_ext(features_file)

        self.df2save = pd.DataFrame([])
        self.save_postfix = ''

        self.button_save_csv = QPushButton('Write to csv')
        self.button_save_fig = QPushButton('Save Figure')
        self.combobox_plot_types = QComboBox()
        self.combobox_plot_types.addItems(self.plot_funcs.keys())
        self.combobox_plot_types.currentIndexChanged.connect(lambda x : self.plot())


        self.button_save_csv.clicked.connect(self.save_csv)
        self.button_save_fig.clicked.connect(self.save_fig)

        # a figure instance to plot on
        self.figure = Figure(figsize=(6, 3))

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self._ax = self.canvas.figure.subplots()

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        layout.addWidget(self.combobox_plot_types)
        
        layout_menu = QHBoxLayout()
        layout_menu.addWidget(self.button_save_csv)
        layout_menu.addWidget(self.button_save_fig)
        
        layout.addLayout(layout_menu)

        self.setLayout(layout)

    def _get_save_name(self, ext):
        fullname = '{}_{}{}'.format(self.root_file, self.save_postfix ,ext)

        dialog = QFileDialog()
        
        dialog.selectFile(fullname)
        dialog.setOptions(QFileDialog.DontUseNativeDialog)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilters(['*' + ext])
        ret = dialog.exec();
        if (ret == QDialog.Accepted):
            fullname = dialog.selectedFiles()[0]
        return fullname

    def save_csv(self):
        fullname = self._get_save_name('.csv')
        self.df2save.to_csv(fullname, index=False)

    def save_fig(self):
        fullname = self._get_save_name('.pdf')
        self.figure.savefig(fullname)

    def plot(self, worm_index = None, feature = None):
        
        if worm_index is None:
            worm_index = self.worm_index
        else:
            self.worm_index = worm_index

        if feature is None:
            feature = self.feature
        else:
            self.feature = feature

        key = self.combobox_plot_types.currentText()
        func = self.plot_funcs[key]

        if 'All' in key:
            func(feature)
        else:
            func(worm_index, feature)

        
    def feature_label(self, feature):
        lab =  feature.replace('_' , ' ').title()
        return lab

    def _plot_single_timeseries(self, worm_index, feature):
        worm_data = self.traj_worm_index_grouped.get_group(worm_index)
        feat_val = self.timeseries_data.loc[worm_data.index]

        self._ax.clear()

        self._ax.set_xlabel('Time [{}]'.format(self.time_units))
        self._ax.set_ylabel(self.feature_label(feature))
        self._ax.set_title('W: {}'.format(worm_index))

        tt = feat_val['timestamp']/self.fps
        self._ax.plot(tt, feat_val[feature])
        

        self.figure.tight_layout()

        self._ax.figure.canvas.draw()


        self.df2save = pd.DataFrame({'time':tt, feature:feat_val[feature]})
        self.save_postfix = 'TS_W{}_{}'.format(worm_index, feature)

    def _plot_all_timeseries(self, feature):
        self._ax.clear()

        self._ax.set_xlabel('Time [{}]'.format(self.time_units))
        self._ax.set_ylabel(self.feature_label(feature))
        self._ax.set_title('All Trajectories')


        # compute derived time columns in one operation to avoid DataFrame fragmentation
        ts = self.timeseries_data['timestamp'] / self.fps
        tb = (ts / self.ts_bin).round()
        # assign both columns at once and rebind the DataFrame (avoids many small inserts)
        self.timeseries_data = self.timeseries_data.assign(timestamp_s=ts, timestamp_binned=tb)
        #self._ax.plot(feat_val['timestamp'], feat_val[feature])
        for _, worm_data in self.traj_worm_index_grouped:
            feat_val = self.timeseries_data.loc[worm_data.index]

            self._ax.plot(feat_val['timestamp_s'], feat_val[feature], alpha=0.4)


        agg_data = self.timeseries_data[['timestamp_binned', feature]].groupby('timestamp_binned').agg('median')[feature]
        xx = agg_data.index*self.ts_bin 
        yy = agg_data.values

        self._ax.plot(xx, yy, '-', lw=2, color='black')

        self.figure.tight_layout()

        self._ax.figure.canvas.draw()

        self.df2save = pd.DataFrame({'time_bin':xx, 'median_' + feature : yy })
        self.save_postfix = 'TS_ALL_{}'.format(feature)
    
    def _plot_single_histogram(self, worm_index, feature):
        worm_data = self.traj_worm_index_grouped.get_group(worm_index)
        feat_val = self.timeseries_data.loc[worm_data.index]

        self._ax.clear()

        
        self._ax.set_xlabel(self.feature_label(feature))
        self._ax.set_ylabel('Counts')
        self._ax.set_title('W: {}'.format(worm_index))

        counts, edges, _ = self._ax.hist(feat_val[feature].dropna())
        bins = edges[:-1] + (edges[1] - edges[0])/2

        
        self.figure.tight_layout()

        self._ax.figure.canvas.draw()

        self.df2save = pd.DataFrame({feature + '_bin': bins, 'counts': counts })
        self.save_postfix = 'HIST_W{}_{}'.format(worm_index, feature)

    def _plot_all_histogram(self, feature):
        self._ax.clear()

        self._ax.set_xlabel(self.feature_label(feature))
        self._ax.set_ylabel('Counts')
        self._ax.set_title('All Trajectories')

        counts, edges, _ = self._ax.hist(self.timeseries_data[feature].dropna(), 100)
        bins = edges[:-1] + (edges[1] - edges[0])/2

        self.figure.tight_layout()

        self._ax.figure.canvas.draw()

        self.df2save = pd.DataFrame({feature + '_bin': bins, 'counts': counts })
        self.save_postfix = 'HIST_ALL_{}'.format(feature)



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main = FeatureReaderBase(ui='')

    #skel_file = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/mutliworm_example/BRC20067_worms10_food1-10_Set2_Pos5_Ch2_02062017_121709_featuresN.hdf5'
    mask_file = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/mutliworm_example/BRC20067_worms10_food1-10_Set2_Pos5_Ch2_02062017_121709.hdf5'
    main.updateVideoFile(mask_file)

    plotter = PlotFeatures(main.skeletons_file,
        main.timeseries_data,
                                   main.traj_worm_index_grouped,
                                   main.time_units,
                                   main.xy_units,
                                   main.fps)
    plotter.show()
    plotter.plot(1, 'length')
    

    sys.exit(plotter.exec_())