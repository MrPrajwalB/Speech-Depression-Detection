import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


data=[]
Target=[]
length=[]

dev=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\dev_split_Depression_AVEC2017.csv'))[:,[0,2,3]].astype(int)
train=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\train_split_Depression_AVEC2017.csv'))[:,[0,2,3]].astype(int)
test=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\test_split_Depression_AVEC2017.csv'))[:,:].astype(int)

Y_test=test[:,1]
Y_dev=dev[:,1]

p=[train,dev,test]
for k in range(len(p)):
    path1=[]
    path2=[]
    path3=[]
    ffeatures=[]
    foutput=[]
    start_f=[]
    stop_f=[]
    diff_f=[]
    for i in range(len(p[k])):
        s1= 'C:\\Users\\Prajwal\\Downloads\\transcript\\transcript\\' + str(p[k][i][0]) + '_TRANSCRIPT.csv'
        path1.append(s1)
        s2= 'C:\\Users\\Prajwal\\Downloads\\wwwdaicwoz\\wwwdaicwoz\\' + str(p[k][i][0]) + '_P/' + str(p[k][i][0]) +'_COVAREP.csv'
        path2.append(s2)
        s3= 'C:\\Users\\Prajwal\\Downloads\\wwwdaicwoz\\wwwdaicwoz\\' + str(p[k][i][0]) + '_P/' + str(p[k][i][0]) +'_FORMANT.csv'
        path3.append(s3)

    for j in range(len(path1)): 
        frames=pd.read_csv(path1[j], sep='\t')
        features= pd.read_csv(path2[j], sep= ',', header= None)
        formant= pd.read_csv(path3[j], sep= ',', header= None)
        gender=p[k][j][2]
        score=p[k][j][1]
        frames1=frames[frames['speaker'].str.match('Participant')]
        frames2= frames1.iloc[:,0:2]
        frames3= frames2.values
        frames4= frames3[:,:]*100
        c=[]
        d=[]
        if((gender==0) and (score>10)): 
            g=5
        else:
            g=4
        for m in range(len(frames4)):
            start_f.append(int(frames4[m,0]))
            stop_f.append(int(frames4[m,1]))
            diff_f.append(int(frames4[m,1])-int(frames4[m,0]))
            
        start=np.asarray(start_f).reshape(-1,1)
        stop=np.asarray(stop_f).reshape(-1,1)
        diff=np.asarray(diff_f).reshape(-1,1)
        lis_frame=np.concatenate((start,stop,diff), axis =1)            
        lis_frame=lis_frame[-lis_frame[:,2].argsort()] # sorting as per length of frame
        
        for f in range(len(lis_frame[:g,:])): #selecting longest frames and extracting corresponding features
            start_frame=lis_frame[f][0]
            stop_frame=lis_frame[f][1]
            a=features.iloc[start_frame:stop_frame, :]
            b=formant.iloc[start_frame:stop_frame, :]
            c.append(a)
            d.append(b)
        final_feats=pd.concat(c)
        final_formant=pd.concat(d)
        final_features= pd.concat([final_feats,final_formant], axis =1)
        l=len(final_features)
        length.append(l)
        ffeatures.append(final_features)
        arr_ft=np.full([len(final_features), 1], p[k][j][1])
        foutput.append(arr_ft)
            
    ffeatures=pd.concat(ffeatures)
    ffeatures= np.asarray(ffeatures)
    foutput=np.concatenate(foutput)
    ffeatures= MinMaxScaler().fit_transform(ffeatures)
    data.append(ffeatures)
    Target.append(foutput)

X_train= np.expand_dims(data[0], axis=2)
y_train= Target[0]
X_dev= np.expand_dims(data[1], axis =2)
y_dev= Target[1]
X_test=np.expand_dims(data[2], axis=2)
y_test=Target[2]


length=np.asarray(length)
length_dev=np.cumsum(length[107:142])
length_test=np.cumsum(length[142:])
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, SpatialDropout1D, Flatten, BatchNormalization
from keras import backend
from keras import regularizers
 
def rmse(y_test, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_test), axis=-1))
# Initialising the ANN
regressor = Sequential()

regressor.add(Conv1D(input_shape=(79,1),filters=64, kernel_size=3, activation='relu'))
regressor.add(MaxPooling1D())
regressor.add(SpatialDropout1D(0.4))

regressor.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
regressor.add(MaxPooling1D())
regressor.add(SpatialDropout1D(0.4))
regressor.add(Flatten())

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2()))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.4))
regressor.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', kernel_regularizer=regularizers.l2()))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.4))
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', rmse])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train,validation_data=(X_dev,y_dev), batch_size =128, epochs =30)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
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
