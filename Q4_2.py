# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:05:47 2018

@author: anny
"""


import seaborn
import numpy as np, scipy
import librosa, librosa.display
import  matplotlib.pyplot as plt, IPython.display as ipd
plt.rcParams['figure.figsize'] = (13, 5)
from os import listdir 
from os.path import isfile, join 

path = 'data1/BallroomData/'
out = open('Q4_2.txt','w')
for i in ['ChaChaCha','Jive','Quickstep','Rumba','Samba','Tango','VienneseWaltz','Waltz']:
    mypath = join(path,i)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    bpmPath = 'data2/BallroomAnnotations/ballroomGroundTruth/'
    
    average_P1 = 0
    average_P2 = 0
    out.write('Genres: '+i+'\n')
    print("Genres: ",i)
    for f in files:
        tmp = f.split('.')
        if tmp[1]=='wav':
            bpmFile = tmp[0] + '.bpm'
        
        #y, sr = librosa.load('data1/BallroomData/ChaChaCha/Media-1034{0:02}.wav'.format(idx))
        #ans = int(open('data2/BallroomAnnotations/ballroomGroundTruth/Media-1034{0:02}.bpm'.format(idx),'r').read())
        y, sr = librosa.load(join(mypath,f))
        ans = int(open(join(bpmPath,bpmFile),'r').read())
        hop_length = 220
        # novelty curve
        onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=2048)
        
        # calculate tempogram
        #S = librosa.stft(onset_env, hop_length=1, n_fft=1200)
        ACF = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                      hop_length=hop_length,win_length = int(30*sr/hop_length))
        #librosa.display.specshow(ACF, sr=sr, hop_length=hop_length, x_axis='time')
        # method 1 to get bpm (not good)
        bpms = librosa.core.tempo_frequencies(ACF.shape[0], hop_length=hop_length, sr=sr)


        ACF = np.mean(ACF, axis=1)
        test = np.array([ACF[y] for y in range(ACF.shape[0])])
        
        max_idx = np.argmax(bpms < 300)
        test[:max_idx] = 0
        
        
        best_period = np.argmax(test)
        t1 = bpms[best_period]
        ftmp1 = ACF[best_period]
        test[best_period]=0;
        second = np.argmax(test);
        t2 = bpms[second]
        ftmp2 = ACF[second]
        #print(ftmp2)
        f1 = ftmp1
        f2 = ftmp2
        while(abs(t1-t2)<0.08*t1 or abs(t1-t2)<0.08*t2):
            second = np.argmax(test);
            t2 = bpms[second]
            f2 = ACF[second]
            test[second]=0;
        if(t1 > t2):
            tmp = t1
            t1 = t2
            t2 = tmp
            tmp = ftmp1
            f1 = ftmp2
            f2 = tmp
        
        
        
        # tempo using librosa
        #b=librosa.beat.tempo(y, sr=sr)
        
        t21 = t2/t1
        t1g = t1/ans
        t2g = t2/ans
        out.write('Estimate: '+str(f)+'\n')
        out.write('Estimate tempo slow: '+str(t1)+'\n')
        out.write('Estimate tempo fast: '+str(t2)+'\n')
        out.write('Ans:                 '+str(ans)+'\n')
        out.write('T2/T1:               '+str(t21)+'\n')
        out.write('T1/G:               '+str(t1g)+'\n')
        out.write('T2/G:               '+str(t2g)+'\n')
        print('Estimate: ',f)
        print('Estimate tempo slow: ',t1)
        print('Estimate tempo fast: ',t2)
        print('Ans:                 ',ans)
        print('T2/T1:               ',t21)
        print('T1/G:                ',t1g)
        print('T2/G:                ',t2g)
       
    
    
    out.write("---------------------------------------------\n")
    
    print("---------------------------------------------\n")
    out.write('\n')
    
out.close()
