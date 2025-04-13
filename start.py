# start.py

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from gui import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QFileDialog
import predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Ui_GUI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Ui_GUI, self).__init__()
        self.setupUi(self)
        self.choose.clicked.connect(self.showpic)
        self.predict.clicked.connect(self.showpred)
        self.start_camera.clicked.connect(self.open_camera_window)  # 新增摄像头窗口按钮事件

    def showpic(self):
        imgName, _ = QFileDialog.getOpenFileName()
        if imgName:
            jpg = QtGui.QPixmap(imgName).scaled(self.choosepic.width(), self.choosepic.height())
            self.choosepic.setPixmap(jpg)
            self.label_3.setText(str(imgName))

    def showpred(self):
        imgpath = self.label_3.text()
        self.predict_2.setText(str(predict.main(imgpath)))

    def open_camera_window(self):
        self.camera_window = CameraWindow()  # 创建新的摄像头窗口实例
        self.camera_window.show()  # 显示摄像头窗口


class CameraWindow(QtWidgets.QWidget):
    def __init__(self):
        super(CameraWindow, self).__init__()
        self.setWindowTitle("Camera Stream")
        self.setGeometry(100, 100, 800, 600)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(10, 10, 780, 580)  # 设置显示区域大小

        self.cap = cv2.VideoCapture(0)  # 初始化摄像头
        self.model = tf.keras.models.load_model('best_model_64.h5')  # 加载模型
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # 开始更新帧
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = cv2.merge([face, face, face])  # 将灰度图像转换为三通道
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)

                # 进行情感预测
                predictions = self.model.predict(face)
                emotion_index = np.argmax(predictions[0])
                emotion = self.emotion_labels[emotion_index]

                # 在图像上绘制面部框和情感标签
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            # 将 BGR 转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 在 QLabel 中显示摄像头画面
            self.label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))

        # 定时更新摄像头画面
        QtCore.QTimer.singleShot(20, self.update_frame)

    def closeEvent(self, event):
        """关闭窗口时释放摄像头"""
        self.cap.release()  # 释放摄像头资源
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_GUI()
    ui.show()
    sys.exit(app.exec_())