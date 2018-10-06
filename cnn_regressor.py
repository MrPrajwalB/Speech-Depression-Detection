import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread

dev=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\dev_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)
train=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\train_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)
test=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\test_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)

dev_target=np.delete(dev,(24,25), axis=0)
train_target=np.delete(train,(9,12,25,40), axis=0)
test_target=np.delete(test,(), axis=0)

Y_test=test_target[:,1]
Y_dev=dev_target[:,1]

y_dev=np.load('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\y_dev.npy')
y_train=np.load('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\y_train.npy')
y_test=np.load('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\y_test.npy')
length_dev=np.cumsum(np.load('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\dev_img.npy'))
length_test=np.cumsum(np.load('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\test_img.npy'))

dev_images, train_images, test_images = [],[],[]
for image_path in os.listdir('I:\\database_CNN\\dev'):
    path='I:\\database_CNN\\dev'+ '\\' + image_path
    img = imread(path)
    img = resize(img, (174, 66, 3))
    dev_images.append(img)
for image_path in os.listdir('I:\\database_CNN\\train'):
    path='I:\\database_CNN\\train'+ '\\' + image_path
    img = imread(path)
    img = resize(img, (174, 66, 3))
    train_images.append(img)
for image_path in os.listdir('I:\\database_CNN\\test'):
    path='I:\\database_CNN\\test'+ '\\' + image_path
    img = imread(path)
    img = resize(img, (174, 66, 3))
    test_images.append(img)  
  
X_dev = np.array(dev_images)
X_train=np.array(train_images)
X_test=np.array(test_images)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, SpatialDropout2D
from keras import backend
 
def rmse(y_test, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_test), axis=-1))

# Initialising the CNN
regressor = Sequential()

# Step 1 - Convolution
regressor.add(Conv2D(32, (3, 3), input_shape = (174, 66, 3), activation = 'relu'))

# Step 2 - Pooling
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(SpatialDropout2D(0.3))
# Adding a second convolutional layer
regressor.add(Conv2D(32, (3, 3), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2, 2)))
regressor.add(SpatialDropout2D(0.3))
# Step 3 - Flattening
regressor.add(Flatten())

# Step 4 - Full connection
regressor.add(Dense(units = 64, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))

# Compiling the CNN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae',rmse])
regressor.fit(X_train, y_train,validation_data=(X_dev,y_dev), batch_size =256, epochs =10)

y_pred_test = regressor.predict(X_test)
y_pred_dev = regressor.predict(X_dev)

p=0
MEAN_DEV=[]
for m in range(len(length_dev)):
    mean=np.mean(y_pred_dev[p:length_dev[m]])
    MEAN_DEV.append(mean)
    p=length_dev[m]
    
q=0
MEAN_TEST=[]
for m in range(len(length_test)):
    mean=np.mean(y_pred_test[q:length_test[m]])
    MEAN_TEST.append(mean)
    p=length_test[m]
    
Y_pred_dev=np.asarray(MEAN_DEV)
Y_pred_test=np.asarray(MEAN_TEST)
    
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math  

mae_dev=mean_absolute_error(Y_dev,Y_pred_dev)
rmse_dev= math.sqrt(mean_squared_error(Y_dev, Y_pred_dev))
print('MAE_DEV=',mae_dev)
print('RMSE_DEV=',rmse_dev)
mae_test=mean_absolute_error(Y_test,Y_pred_test)
rmse_test= math.sqrt(mean_squared_error(Y_test, Y_pred_test))
print('MAE_TEST=',mae_test)
print('RMSE_TEST=',rmse_test)
