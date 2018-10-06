import numpy as np
import librosa 
import matplotlib.pyplot as plt
import os
import glob

p = []
q = []

path=  'C:\\Users\\Prajwal\\Desktop\\UMD project\\audio_2\\492_P'

for filename in glob.glob(os.path.join(path, '*.wav')):
    dur= librosa.get_duration(filename=filename)
    p.append(filename)
    q.append(dur)
    
final=np.empty([len(p), 2])
array_name=np.asarray(p)
array_dur=np.asarray(q)
final=np.concatenate((array_name, array_dur))
plt.hist(array)
plt.xlabel('Duration in seconds')
plt.ylabel('No. of Samples')
plt.title('Speech duration analysis for P_492')
plt.savefig('P_492.jpg')