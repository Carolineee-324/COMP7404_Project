#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf,optimizers
from tensorflow.keras import layers, models
import csv

import cv2

import pickle
from keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Conv2D
import seaborn as sns

# In[2]:


database_path = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/'    
csv_file = os.path.join(database_path, 'fer2013.csv')
df = pd.read_csv(csv_file)
df.head()


# In[3]:


# load dataset
# database_path = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/'     # dataset_path
datasets_path = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/'     # output_path
csv_file = os.path.join(database_path, 'fer2013.csv')   # fer2013_csv
train_csv = os.path.join(datasets_path, 'train.csv')    # training_data
val_csv = os.path.join(datasets_path, 'val.csv')        # validation_data
test_csv = os.path.join(datasets_path, 'test.csv')      # tes_data
 
# seperate training,validation,test data
with open(csv_file) as f:
    csvr = csv.reader(f)    # read data by rows
    header = next(csvr)     # get header of each row
    rows = [row for row in csvr]     # search each row
 
    # seperate data by target at last column
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
    print("number of train_data：", len(trn))
 
    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
    print("number of valid_data：", len(val))
 
    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
    print("number of test_data：", len(tst))


# In[4]:


label_list = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'normal'}


# In[5]:


 
# transform into gred_pic and divide into different emotion set
datasets_path = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/'
train_csv = os.path.join(datasets_path, 'train.csv')    # load data
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')
train_set = os.path.join(datasets_path, 'train')        # output data
val_set = os.path.join(datasets_path, 'val')
test_set = os.path.join(datasets_path, 'test')
label_list = {'0':'anger','1':'disgust','2':'fear','3':'happy','4':'sad','5':'surprised','6':'normal'}
 
for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
    if not os.path.exists(save_path):           
        os.makedirs(save_path)
 
    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
 
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label_list[label])
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            # transform into RGB pic,and convert into 8 level gred-pic，L=R*299/1000+G*587/1000+B*114/1000
            img = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            print(image_name)
            img.save(image_name)


# In[9]:
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
train_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/train'
val_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/val'
test_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/test'
 
 
datagen = ImageDataGenerator( rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             zoom_range=0.1,
                             rescale=1./255)

# batch_size = 128
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)
validation_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)
# batch_size = 64
train_generatorc_64 = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)
validation_generator_64 = datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)
test_generator_64 = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    shuffle=True,
    class_mode='categorical'
)
 
# create network
model = Sequential()
# first part
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))      # first max_pooling
model.add(BatchNormalization())
model.add(Dropout(0.4))     # drop 40% network randomly, avoid overfitting
# second part
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
# third part
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
 
model.add(Flatten())                                  
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))             
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))             # classifier output layer
model.summary()


# In[82]:


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0001),  # Adam optimizar
            # optimizer=optimizers.RMSprop(learning_rate=0.0001),  # rmsprop optimizar
              metrics=['accuracy'])




# In[76]:

# using batch_size=128
# checkpoint avoid overfitting
from keras.callbacks import ModelCheckpoint,EarlyStopping
checkpointer = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, 
                              restore_best_weights=True,mode="max",patience = 10),
                ModelCheckpoint('best_model.h5',monitor="val_accuracy",verbose=1,
                                save_best_only=True,mode="max")]

history = model.fit(train_generator,
                    epochs=50,
                    batch_size=128,   
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=validation_generator)


# In[79]:


plt.plot(history.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


# In[78]:


plt.plot(history.history["loss"],'r', label="Training Loss")
plt.plot(history.history["val_loss"],'b', label="Validation Loss")
plt.legend()

# In[85]:

# using batch_size=64
checkpointer_64 = [EarlyStopping(monitor = 'val_accuracy', verbose = 1, 
                              restore_best_weights=True,mode="max",patience = 10),
                ModelCheckpoint('best_model_64.h5',monitor="val_accuracy",verbose=1,
                                save_best_only=True,mode="max")]

history_64 = model.fit(train_generatorc_64,
                    epochs=50,
                    batch_size=64,   
                    verbose=1,
                    callbacks=[checkpointer_64],
                    validation_data=validation_generator_64)


# history_non = model.fit(train_generator,
#                     epochs=50,
#                     batch_size=64,   
#                     verbose=1,
#                     callbacks=[checkpointer])



# In[86]:


plt.plot(history_64.history["accuracy"],'r',label="Training Accuracy")
plt.plot(history_64.history["val_accuracy"],'b',label="Validation Accuracy")
plt.legend()


# In[87]:

# loss
loss = model.evaluate(test_generator_64) 
print("Test Acc: " + str(loss[1]))


# In[7]:
# load_model

from tensorflow.keras.models import load_model
loaded_model = load_model('best_model.h5')

# model structure
loaded_model.summary()

# predict by loaded_model
# predictions = loaded_model.predict(X_test)


loss = loaded_model.evaluate(test_generator) 
print("Test Acc: " + str(loss[1]))

# load model_64
# loaded_model_64 = load_model('best_model_64.h5')

# loss_64 = loaded_model.evaluate(test_generator_64) 
# print("Test Acc: " + str(loss_64[1]))


# In[13]:


# get data and label
x_test = []
y_test = []

for i in range(len(test_generator)):
    batch_x, batch_y = test_generator[i]
    x_test.append(batch_x)
    y_test.append(batch_y)

# transform into NumPy
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# In[20]:
# predict test_data

preds = loaded_model.predict(x_test)
y_pred = np.argmax(preds , axis = 1 )


# In[31]:

# predict loaded_pic

# load and preprocessing pic
def load_and_preprocess_image(img_path):
    # load pic
    img = cv2.imread(img_path)
    # transform BGR into RGB（OpenCV default is BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # modify the size of pic
    img = cv2.resize(img, (48, 48))
    # normalization
    img = img.astype('float32') / 255.0
    # add a new dimension (1, height, width, channels)
    img = np.expand_dims(img, axis=0)
    return img

# load_pic
img_path = '/Users/gyd/Downloads/sad.jpeg' 
img = load_and_preprocess_image(img_path)


# In[36]:

# predict load_pic
loaded_model.predict(img)


# In[27]:

# confusion matrix

cm_=classification_report(np.argmax(y_test, axis = 1 ),y_pred,digits=3)
categories = [f'Class {i+1}' for i in range(7)]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_, annot=True, fmt="d"ww, xticklabels=categories, yticklabels=categories)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()


