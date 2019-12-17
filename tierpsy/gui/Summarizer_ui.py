# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Summarizer.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Summarizer(object):
    def setupUi(self, Summarizer):
        Summarizer.setObjectName("Summarizer")
        Summarizer.resize(632, 537)
        self.centralwidget = QtWidgets.QWidget(Summarizer)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 1)
        self.p_feature_type = QtWidgets.QComboBox(self.centralwidget)
        self.p_feature_type.setObjectName("p_feature_type")
        self.gridLayout_2.addWidget(self.p_feature_type, 1, 1, 1, 1)
        self.p_root_dir = QtWidgets.QLineEdit(self.centralwidget)
        self.p_root_dir.setObjectName("p_root_dir")
        self.gridLayout_2.addWidget(self.p_root_dir, 0, 1, 1, 2)
        self.p_is_manual_index = QtWidgets.QCheckBox(self.centralwidget)
        self.p_is_manual_index.setObjectName("p_is_manual_index")
        self.gridLayout_2.addWidget(self.p_is_manual_index, 1, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.pushButton_rootdir = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_rootdir.sizePolicy().hasHeightForWidth())
        self.pushButton_rootdir.setSizePolicy(sizePolicy)
        self.pushButton_rootdir.setObjectName("pushButton_rootdir")
        self.gridLayout_2.addWidget(self.pushButton_rootdir, 0, 0, 1, 1)
        self.p_summary_type = QtWidgets.QComboBox(self.centralwidget)
        self.p_summary_type.setObjectName("p_summary_type")
        self.gridLayout_2.addWidget(self.p_summary_type, 2, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_2)
        self.FoldArgs = QtWidgets.QWidget(self.centralwidget)
        self.FoldArgs.setObjectName("FoldArgs")
        self.gridLayout = QtWidgets.QGridLayout(self.FoldArgs)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.FoldArgs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.p_n_folds = QtWidgets.QSpinBox(self.FoldArgs)
        self.p_n_folds.setObjectName("p_n_folds")
        self.gridLayout.addWidget(self.p_n_folds, 0, 0, 1, 1)
        self.p_time_sample_seconds = QtWidgets.QDoubleSpinBox(self.FoldArgs)
        self.p_time_sample_seconds.setMaximum(100000000000.0)
        self.p_time_sample_seconds.setObjectName("p_time_sample_seconds")
        self.gridLayout.addWidget(self.p_time_sample_seconds, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.FoldArgs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.FoldArgs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.p_frac_worms_to_keep = QtWidgets.QDoubleSpinBox(self.FoldArgs)
        self.p_frac_worms_to_keep.setMaximum(1.0)
        self.p_frac_worms_to_keep.setSingleStep(0.01)
        self.p_frac_worms_to_keep.setObjectName("p_frac_worms_to_keep")
        self.gridLayout.addWidget(self.p_frac_worms_to_keep, 0, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.FoldArgs)
        self.TimeWindows = QtWidgets.QWidget(self.centralwidget)
        self.TimeWindows.setObjectName("TimeWindows")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.TimeWindows)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.p_select_feat = QtWidgets.QComboBox(self.TimeWindows)
        self.p_select_feat.setObjectName("p_select_feat")
        self.gridLayout_3.addWidget(self.p_select_feat, 1, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.TimeWindows)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 1, 1, 1, 1)
        self.p_time_units = QtWidgets.QComboBox(self.TimeWindows)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.p_time_units.sizePolicy().hasHeightForWidth())
        self.p_time_units.setSizePolicy(sizePolicy)
        self.p_time_units.setObjectName("p_time_units")
        self.p_time_units.addItem("")
        self.p_time_units.addItem("")
        self.gridLayout_3.addWidget(self.p_time_units, 0, 4, 1, 1)
        self.p_time_windows = QtWidgets.QLineEdit(self.TimeWindows)
        self.p_time_windows.setObjectName("p_time_windows")
        self.gridLayout_3.addWidget(self.p_time_windows, 0, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.TimeWindows)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 0, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.TimeWindows)
        self.SelectByKeywords = QtWidgets.QWidget(self.centralwidget)
        self.SelectByKeywords.setObjectName("SelectByKeywords")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.SelectByKeywords)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_9 = QtWidgets.QLabel(self.SelectByKeywords)
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 2, 0, 1, 1)
        self.p_keywords_include = QtWidgets.QLineEdit(self.SelectByKeywords)
        self.p_keywords_include.setObjectName("p_keywords_include")
        self.gridLayout_4.addWidget(self.p_keywords_include, 1, 1, 1, 1)
        self.p_keywords_exclude = QtWidgets.QLineEdit(self.SelectByKeywords)
        self.p_keywords_exclude.setObjectName("p_keywords_exclude")
        self.gridLayout_4.addWidget(self.p_keywords_exclude, 2, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.SelectByKeywords)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.SelectByKeywords)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 0, 0, 1, 2)
        self.verticalLayout_3.addWidget(self.SelectByKeywords)
        self.p_abbreviate_features = QtWidgets.QCheckBox(self.centralwidget)
        self.p_abbreviate_features.setObjectName("p_abbreviate_features")
        self.verticalLayout_3.addWidget(self.p_abbreviate_features)
        self.p_dorsal_side_known = QtWidgets.QCheckBox(self.centralwidget)
        self.p_dorsal_side_known.setObjectName("p_dorsal_side_known")
        self.verticalLayout_3.addWidget(self.p_dorsal_side_known)
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setObjectName("pushButton_start")
        self.verticalLayout_3.addWidget(self.pushButton_start)
        Summarizer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Summarizer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 632, 22))
        self.menubar.setObjectName("menubar")
        Summarizer.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Summarizer)
        self.statusbar.setObjectName("statusbar")
        Summarizer.setStatusBar(self.statusbar)

        self.retranslateUi(Summarizer)
        QtCore.QMetaObject.connectSlotsByName(Summarizer)

    def retranslateUi(self, Summarizer):
        _translate = QtCore.QCoreApplication.translate
        Summarizer.setWindowTitle(_translate("Summarizer", "MainWindow"))
        self.label_5.setText(_translate("Summarizer", "Summary Type"))
        self.p_is_manual_index.setText(_translate("Summarizer", "Use manually \n"
"edited trajectories?"))
        self.label_4.setText(_translate("Summarizer", "Features Type"))
        self.pushButton_rootdir.setText(_translate("Summarizer", "Root  Directory"))
        self.label_2.setText(_translate("Summarizer", "Fraction of trajectories\n"
" to sample"))
        self.label.setText(_translate("Summarizer", "Number of Folds"))
        self.label_3.setText(_translate("Summarizer", "Time (seconds)\n"
" to sample"))
        self.label_7.setText(_translate("Summarizer", "Select features"))
        self.p_time_units.setItemText(0, _translate("Summarizer", "frame_numbers"))
        self.p_time_units.setItemText(1, _translate("Summarizer", "seconds"))
        self.p_time_windows.setText(_translate("Summarizer", "0:end"))
        self.label_6.setText(_translate("Summarizer", "Time windows"))
        self.label_9.setText(_translate("Summarizer", "Exclude keywords"))
        self.label_8.setText(_translate("Summarizer", "Include keywords"))
        self.label_10.setText(_translate("Summarizer", "Select features by keywords :"))
        self.p_abbreviate_features.setText(_translate("Summarizer", "Abbreviate feature names"))
        self.p_dorsal_side_known.setText(_translate("Summarizer", "Dorsal side annotated"))
        self.pushButton_start.setText(_translate("Summarizer", "START"))

