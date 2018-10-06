import numpy as np
import pandas as pd
import librosa 
import librosa.display
import os


dev=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\dev_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)
train=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\train_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)
test=np.asarray(pd.read_csv('C:\\Users\\Prajwal\\Desktop\\UMD_FT\\test_split_Depression_AVEC2017.csv'))[:,[0,2]].astype(int)

folder=['I:\\dev', 'I:\\train', 'I:\\test']
no_data=np.array([318,321,341,342,362,394,398,451,458,460,480])
dev_names=np.setdiff1d(dev[:,0],no_data)
train_names=np.setdiff1d(train[:,0],no_data)
test_names=np.setdiff1d(test[:,0], no_data)
no_image=[]
dev_output,train_output,test_output=[],[],[]
p=0

dev_paths, train_paths, test_paths=[],[],[]
for i in range(len(dev_names)):
    path='C:\\Users\\Prajwal\\Desktop\\UMD_FT\\audio_2\\' + str(dev_names[i]) + '_P'
    dev_paths.append(path)
for i in range(len(train_names)):
    path='C:\\Users\\Prajwal\\Desktop\\UMD_FT\\audio_2\\' + str(train_names[i]) + '_P'
    train_paths.append(path)
for i in range(len(test_names)):
    path='C:\\Users\\Prajwal\\Desktop\\UMD_FT\\audio_2\\' + str(test_names[i]) + '_P'
    test_paths.append(path)

data=[dev_paths,train_paths,test_paths]
for q in range(3):
    for j in range(len(data[q])):
        target=[]
        for filename in sorted(os.listdir(data[q][j])):
            file=data[q][j] + '\\' +filename
            y, sr= librosa.load(file,sr=None)
            dur= librosa.get_duration(filename=file)
            print(sr, dur, file)
            l=len(y)
            name= filename.replace(".wav", "")
            if dur >= 5 :
                ind= np.arange(sr*5,l,sr*5)
                lis= np.split(y,ind)
                del lis[len(lis)-1]
                target.append(len(lis))
                """i=0
                for x in lis:
                    i=i+1
                    name1= name + '_' + str(i)
                    S= librosa.feature.melspectrogram(y=x, sr= sr)
                    plt.figure(figsize=(10, 4),frameon=False)
                    librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
                    plt.tight_layout()
                    plt.savefig(folder[p] + '\\' +name + '.jpg',bbox_inches='tight', pad_inches=0)
                    plt.close()"""
        
        no_image.append(np.sum(np.asarray(target)))
    p=p+1
    
dev_image=np.asarray(no_image[:len(dev_names)]).astype(int)
np.save('dev_img',dev_image)
train_image=np.asarray(no_image[len(dev_names):len(dev_names)+len(train_names)]).astype(int)
test_image=np.asarray(no_image[len(dev_names)+len(train_names):]).astype(int)
np.save('test_img',test_image)
dev_target=np.delete(dev,(24,25), axis=0)
train_target=np.delete(train,(9,12,25,40), axis=0)
test_target=np.delete(test,(), axis=0)

for i in range(len(dev_names)):
    arr=np.full((dev_image[i],1),dev_target[i][1])
    dev_output.append(arr)
for i in range(len(train_names)):
    arr=np.full((train_image[i],1),train_target[i][1])
    train_output.append(arr)
for i in range(len(test_names)):
    arr=np.full((test_image[i],1),test_target[i][1])
    test_output.append(arr)
    
y_dev=np.concatenate(dev_output)
y_train=np.concatenate(train_output)
y_test=np.concatenate(test_output)

np.save('y_dev',y_dev)
np.save('y_train',y_train)
np.save('y_test',y_test)