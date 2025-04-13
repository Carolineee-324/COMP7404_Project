import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

test_img = np.zeros([1, 48 * 48], dtype=np.uint8)
cascade = cv2.CascadeClassifier("/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/haarcascade_frontalface_alt.xml")
cascade.load("/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/haarcascade_frontalface_alt.xml")



def main(imgpath):
    test_image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    print('bb')
    # rects = detect(test_image, cascade)
    # for x1, y1, x2, y2 in rects:
    #     cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    #     # 调整截取脸部区域大小
    #     img_roi = np.uint8([y2 - y1, x2 - x1])
    #     roi = test_image[y1:y2, x1:x2]
    #     img_roi = roi
    #     re_roi = cv2.resize(img_roi, (48, 48))
    #     # global test_img
    #     test_img[0][0:48 * 48] = np.ndarray.flatten(re_roi)
    # data_img = np.array(test_img)
    # Face_data = np.zeros((1, 48 * 48))
    # x = data_img[0]
    # Face_data[0] = x
    # test_x = Face_data[:]
    # test_x = cv2.resize(test_x, (48, 48)).astype(np.uint8)
    # test_x = cv2.cvtColor(test_x, cv2.COLOR_GRAY2RGB)  # 转换为 RGB 图像
    # test_x = np.expand_dims(test_x, axis=0)   

    # 使用 OpenCV 读取图像
    # img = cv2.imread(img_path)
    # 将图像转换为 RGB 格式（OpenCV 默认是 BGR）
    img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # 调整图像大小（根据模型输入大小进行调整，例如 (48, 48)）
    img = cv2.resize(img, (48, 48))
    # 将图像数据归一化到 [0, 1]
    img = img.astype('float32') / 255.0
    # 增加一个维度以匹配模型输入 (1, height, width, channels)
    img = np.expand_dims(img, axis=0)

    model = load_model('/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/best_model_64.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    predictions = model.predict(img)
    accuracy = np.argmax(predictions, axis=1)
    print(predictions)
    print(accuracy)
    if accuracy == 0:
        result = "anger"
    elif accuracy == 1:
        result = "disgust"
    elif accuracy == 2:
        result = "fear"
    elif accuracy == 3:
        result = "happy"
    elif accuracy == 4:
        result = "sad"
    elif accuracy == 5:
        result = "surprised"
    elif accuracy == 6:
        result = "normal"
    return result



def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3,
                                     minSize=(30, 30))  # ,flags=cv2.CASCADE_SCALE_IMAGE
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


if __name__ == '__main__':
    main()