import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

l= np.arange(300, 493, 1)
l=np.delete(l, [18, 21, 41, 42, 62, 94, 98, 151, 158, 160, 180])

path1=[]
path2=[]
ffeatures=[]
foutput=[]

outputs=pd.read_csv('/home/prajwal/UMD_FT/PHQ_Patient.csv', sep=',', header=None)
outputs=outputs.iloc[:,1]
output=np.asarray(outputs)

for i in range(len(l)):
    s1= '/home/prajwal/UMD_FT/transcript/' + str(l[i]) + '_TRANSCRIPT.csv'
    path1.append(s1)
    s2= '/home/prajwal/UMD_FT/wwwdaicwoz/' + str(l[i]) + '_P/' + str(l[i]) +'_COVAREP.csv'
    path2.append(s2)
    
for j in range(len(path1)): 
    frames=pd.read_csv(path1[j], sep='\t')
    features= pd.read_csv(path2[j], sep= ',', header= None)
    frames1=frames[frames['speaker'].str.match('Participant')]
    frames2= frames1.iloc[:,0:2]
    frames3= frames2.values
    frames4= frames3[:,:]*100
    c=[] 
    for i in range(len(frames4)):
        start_frame= int(frames4[i,0])
        stop_frame= int(frames4[i,1])
        a=features.iloc[start_frame:stop_frame, :]
        c.append(a)
    final_features=pd.concat(c)
    ffeatures.append(final_features)
    arr_ft=np.full([len(final_features), 1], output[j])
    foutput.append(arr_ft)
    
ffeatures=pd.concat(ffeatures)
foutput=np.concatenate(foutput)
foutput1=foutput.astype(int)

for k in range(len(foutput)):
   if(foutput[k][0]>10):
       foutput1[k][0]= 1
   else:
       foutput1[k][0]= 0 

ffeatures= StandardScaler().fit_transform(ffeatures)
ffeatures= pd.DataFrame(data=ffeatures)
target1=pd.DataFrame(data=foutput1, columns=['Target'])
finalDf = pd.concat([ffeatures, target1], axis = 1)

depressed= finalDf[(finalDf['Target']==1)]
control= finalDf[(finalDf['Target']==0)]

depressed1=depressed.sample(n=10000)
control1= control.sample(n=10000)
finalDF=pd.concat([depressed1,control1], axis=0)
finalDF1=finalDF.drop(['Target'], axis=1)
    
X_train, X_test, y_train, y_test = train_test_split(finalDF1, finalDF['Target'], test_size=0.25)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
 
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 74))

# Adding the second hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
#classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units =8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size =32, epochs =50)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math  
mae=mean_absolute_error(y_test,y_pred)
rmse= math.sqrt(mean_squared_error(y_test, y_pred))
print(mae)
print(rmse)

#confusion matrix
y_pred= (y_pred>0.5)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc= (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print('The accuracy obtained =', acc)
    
