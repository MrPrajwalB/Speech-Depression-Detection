import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

l= np.arange(300, 493, 1)
l=np.delete(l, [18, 21, 41, 42, 62, 94, 98, 151, 158, 160, 180])

path1=[]
path2=[]
#path3=[]
ffeatures=[]
foutput=[]

outputs=pd.read_csv('PHQ_Patient.csv', sep=',', header=None)
outputs=outputs.iloc[:,1]
output=np.asarray(outputs)

for i in range(len(l)):
    s1= 'C:\\Users\\Prajwal\\Downloads\\transcript\\transcript\\' + str(l[i]) + '_TRANSCRIPT.csv'
    path1.append(s1)
    s2= 'C:\\Users\\Prajwal\\Downloads\\wwwdaicwoz\\wwwdaicwoz\\' + str(l[i]) + '_P\\' + str(l[i]) +'_COVAREP.csv'
    path2.append(s2)
    #s3= 'C:\\Users\\Prajwal\\Downloads\\wwwdaicwoz\\wwwdaicwoz\\' + str(l[i]) + '_P\\' + str(l[i]) +'_FORMANT.csv'
    #path3.append(s3)

    
for j in range(len(path1)): 
    frames=pd.read_csv(path1[j], sep='\t')
    features= pd.read_csv(path2[j], sep= ',', header= None)
    #formant= pd.read_csv(path3[j], sep= ',', header= None)

    frames1=frames[frames['speaker'].str.match('Participant')]
    frames2= frames1.iloc[:,0:2]
    frames3= frames2.values
    frames4= frames3[:,:]*100
    c=[] 
    #d=[]
    for i in range(len(frames4)):
        start_frame= int(frames4[i,0])
        stop_frame= int(frames4[i,1])
        a=features.iloc[start_frame:stop_frame, :]
        #b=formant.iloc[start_frame:stop_frame, :]
        c.append(a)
        #d.append(b)
    #name1= str(l[j])+ '_final_features'
    #name2= str(l[j]) + '_final_formants'
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
#foutput=np.ravel(foutput)

ffeatures= StandardScaler().fit_transform(ffeatures)
ffeatures= pd.DataFrame(data=ffeatures)
target1=pd.DataFrame(data=foutput1, columns=['Target'])
finalDf = pd.concat([ffeatures, target1], axis = 1)

depressed= finalDf[(finalDf['Target']==1)]
control= finalDf[(finalDf['Target']==0)]

depressed1=depressed.sample(n=5000)
control1= control.sample(n=5000)
finalDF=pd.concat([depressed1,control1], axis=0)
finalDF1=finalDF.drop(['Target'], axis=1)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(ffeatures)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

#tSNE
import time
from sklearn.manifold import TSNE
time_start = time.time()
tsne = TSNE()
tsne_results = tsne.fit_transform(finalDF1)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

tsne_res=pd.DataFrame(data=tsne_results, columns=['principal component 1', 'principal component 2'])
target2=np.asarray(finalDF['Target'])
target2=pd.DataFrame(data=target2, columns=['Target'])
finDf=pd.concat([tsne_res, target2 ], axis=1)

# Scatter plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('x-tSNE', fontsize = 15)
ax.set_ylabel('y-tSNE', fontsize = 15)
ax.set_title('tSNE plot', fontsize = 20)
targets = ['depressed', 'control']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finDf['Target'] == target
    ax.scatter(finDf.loc[indicesToKeep, 'principal component 1'], finDf.loc[indicesToKeep, 'principal component 2'], c = color)
ax.legend(targets)
ax.grid()

    
X_train, X_test, y_train, y_test = train_test_split(finalDF1, finalDF['Target'], test_size=0.25)

model = RandomForestRegressor()
model.fit(X_train, y_train)
preds=model.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

mae=mean_absolute_error(y_test, preds)
rmse= math.sqrt(mean_squared_error(y_test, preds))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 74))

# Adding the second hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 25)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


    