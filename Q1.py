# -*- coding: utf-8 -*-
"""
Created on Tue May 22 20:40:20 2018

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
out = open('Q1.txt','w')
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
        S = librosa.stft(onset_env, hop_length=1, n_fft=1200)
        fourier_tempogram = np.absolute(S)
        #librosa.display.specshow(fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')
        # method 1 to get bpm (not good)
        
        bpms = librosa.core.tempo_frequencies(fourier_tempogram.shape[0], hop_length=hop_length, sr=sr)
        #calculate mean at each time
        fourier_tempogram[0] = [ 0 for i in range(0, fourier_tempogram.shape[1])]
        fourier_tempogram[1] = [ 0 for i in range(0, fourier_tempogram.shape[1])]
        max_idx = np.argmax(bpms < 320)
        fourier_tempogram[:max_idx] = 0
        max_idx = np.argmax(bpms < 50)
        fourier_tempogram[max_idx:] = 0
        fourier_tempogram = np.mean(fourier_tempogram, axis=1, keepdims=True)
        # choose the maximum
        a = np.argmax(fourier_tempogram)
    
        #ftmp2 = fourier_tempogram[a]
        
        #translate bin to bpm freauency
       
        #t2 = bpms[a]
    
        #set the probability of each bpms to normal distribution
        #prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(120)) /1.0)**2)
    
        
        # get the two largest one
        test = fourier_tempogram[:]# * prior[:, np.newaxis];
        best_period = np.argmax(test)
        t1 = bpms[best_period]
        ftmp1 = fourier_tempogram[best_period][0]
        test[best_period]=0;
        second = np.argmax(test);
        t2 = bpms[second]
        ftmp2 = fourier_tempogram[second][0]
        test[second]=0;
        #print(ftmp2)
        f1 = ftmp1
        f2 = ftmp2

        while(abs(t1-t2)<0.08*t1 or abs(t1-t2)<0.08*t2):
            second = np.argmax(test);
            t2 = bpms[second]
            f2 = fourier_tempogram[second][0]
            test[second]=0;
        if(t1 > t2):
            tmp = t1
            t1 = t2
            t2 = tmp
            tmp = ftmp1
            f1 = ftmp2
            f2 = tmp
        
        s1 = f1/(f1+f2)
        
    
        
        tt1 = 0 
        if(abs(ans-t1)/ans<=0.08):
            tt1 = 1
        tt2 = 0
        if(abs(ans-t2)/ans<=0.08):
            tt2 = 1
        p1 = s1*tt1+(1-s1)*tt2
        p2 = 0
        if(abs(ans-t1)/ans<=0.08 or abs(ans-t2)/ans<=0.08):
            p2=1
        
        average_P1 += p1
        average_P2 += p2
        
        # tempo using librosa
        #b=librosa.beat.tempo(y, sr=sr)
        
        out.write('Estimate: '+str(f)+'\n')
        out.write('Estimate tempo slow: '+str(t1)+'\n')
        out.write('Estimate tempo fast: '+str(t2)+'\n')
        out.write('Ans:                 '+str(ans)+'\n')
        out.write('Relative P:          '+str(p1)+'\n')
        out.write('ALOTC:               '+str(p2)+'\n')
        
        print('Estimate: ',f)
        print('Estimate tempo slow: ',t1)
        print('Estimate tempo fast: ',t2)
        print('Ans:                 ',ans)
        print('Relative P:          ',p1)
        print('ALOTC:               ',p2)
    average_P1 /= len(files)
    average_P2 /= len(files)   
    
    print("---------------------------------------------")
    
    out.write("---------------------------------------------\n")
    out.write("Average P-scores:        "+str(average_P1)+'\n')
    out.write("Average ALOCT:           "+str(average_P2)+'\n')
    out.write("---------------------------------------------\n")
    
    print("Average P-scores:        ",average_P1)
    print("Average ALOCT:           ",average_P2)
    
    print("---------------------------------------------")
    out.write('\n')
    
out.close()




