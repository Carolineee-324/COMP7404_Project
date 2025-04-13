# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.choose = QtWidgets.QPushButton(self.centralwidget)
        self.choose.setGeometry(QtCore.QRect(600, 100, 130, 100))
        font = QtGui.QFont()
        font.setFamily("STKaiti")
        font.setPointSize(16)
        self.choose.setFont(font)
        self.choose.setObjectName("choose")
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(600, 220, 130, 80))
        font = QtGui.QFont()
        font.setFamily("STKaiti")
        font.setPointSize(16)
        self.predict.setFont(font)
        self.predict.setObjectName("predict")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(600, 320, 100, 50))
        font = QtGui.QFont()
        font.setFamily("STKaiti")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(275, 40, 101, 51))
        font = QtGui.QFont()
        font.setFamily("STKaiti")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.choosepic = QtWidgets.QLabel(self.centralwidget)
        self.choosepic.setEnabled(True)
        self.choosepic.setGeometry(QtCore.QRect(50, 100, 450, 321))
        self.choosepic.setInputMethodHints(QtCore.Qt.ImhNone)
        self.choosepic.setFrameShape(QtWidgets.QFrame.Box)
        self.choosepic.setText("")
        self.choosepic.setTextFormat(QtCore.Qt.PlainText)
        self.choosepic.setScaledContents(False)
        self.choosepic.setWordWrap(True)
        self.choosepic.setOpenExternalLinks(False)
        self.choosepic.setObjectName("choosepic")
        font = QtGui.QFont()
        font.setFamily("STKaiti")
        font.setPointSize(16)
        self.predict_2 = QtWidgets.QLabel(self.centralwidget)
        self.predict_2.setFont(font)
        self.predict_2.setEnabled(True)
        self.predict_2.setGeometry(QtCore.QRect(615, 380, 100, 50))
        self.predict_2.setInputMethodHints(QtCore.Qt.ImhNone)
        self.predict_2.setFrameShape(QtWidgets.QFrame.Box)
        self.predict_2.setText("")
        self.predict_2.setTextFormat(QtCore.Qt.PlainText)
        self.predict_2.setScaledContents(False)
        self.predict_2.setWordWrap(True)
        self.predict_2.setOpenExternalLinks(False)
        self.predict_2.setObjectName("predict_2")

        # 新增摄像头按钮
        self.start_camera = QtWidgets.QPushButton(self.centralwidget)
        self.start_camera.setGeometry(QtCore.QRect(600, 500, 150, 50))
        font.setPointSize(14)
        self.start_camera.setFont(font)
        self.start_camera.setObjectName("start_camera")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 460, 300, 41))
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.choose.clicked.connect(MainWindow.showpic)
        self.predict.clicked.connect(MainWindow.showpred)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.choose.setText(_translate("MainWindow", "SELECT\nIMAGE"))
        self.predict.setText(_translate("MainWindow", "PREDICTION"))
        self.label.setText(_translate("MainWindow", "RESULT"))
        self.label_2.setText(_translate("MainWindow", "IMAGE"))
        self.start_camera.setText(_translate("MainWindow", "START CAMERA"))  # 摄像头按钮文本


