import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

ffeatures= StandardScaler().fit_transform(ffeatures)
ffeatures= pd.DataFrame(data=ffeatures)
target1=pd.DataFrame(data=foutput, columns=['Target'])
finalDf = pd.concat([ffeatures, target1], axis = 1)

depressed= finalDf[(finalDf['Target']>10)]
control= finalDf[(finalDf['Target']<=10)]

depressed1=depressed.sample(n=5000)
control1= control.sample(n=5000)
finalDF=pd.concat([depressed1,control1], axis=0)
finalDF1=finalDF.drop(['Target'], axis=1)

###Buildng the model
X_train, X_test, y_train, y_test = train_test_split(finalDF1, finalDF['Target'], test_size=0.25)

model = RandomForestRegressor()
model.fit(X_train, y_train)
preds=model.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

mae=mean_absolute_error(y_test, preds)
rmse= math.sqrt(mean_squared_error(y_test, preds))
