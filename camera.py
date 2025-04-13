import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# 加载训练好的模型（假设模型已保存为 'model.h5'）
model = tf.keras.models.load_model('best_model_64.h5')

# 标签映射（根据你的模型训练标签进行调整）
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 创建主窗口
root = tk.Tk()
root.title("实时表情识别")

# 创建一个标签用于显示摄像头画面
label = Label(root)
label.pack()

# 使用 Haar 级联分类器检测面部
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict():
    # 捕获视频流中的每一帧
    ret, frame = cap.read()
    if not ret:
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测面部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 遍历检测到的每个面部
    for (x, y, w, h) in faces:
        # 提取面部区域
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # 假设模型输入为 48x48

        # 将灰度图像转换为三通道图像
        face = cv2.merge([face, face, face])  # 将灰度图像转换为三通道

        face = face.astype('float32') / 255.0  # 归一化
        face = np.expand_dims(face, axis=0)  # 增加批次维度

        # 进行预测
        predictions = model.predict(face)
        emotion_index = np.argmax(predictions[0])
        emotion = emotion_labels[emotion_index]

        # 在图像上绘制面部框和情感标签
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 将 BGR 转换为 RGB 格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 将图像转换为 PIL 格式
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)

    # 更新标签上的图像
    label.img = img
    label.config(image=img)

    # 每 10 毫秒调用一次该函数
    label.after(10, detect_and_predict)

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 启动实时检测
detect_and_predict()

# 启动 GUI 主循环
root.mainloop()

# 释放摄像头
cap.release()