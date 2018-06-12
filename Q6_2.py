# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:24:06 2018

@author: anny
"""

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
out = open('Q6.txt','w')
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
        ta1 = bpms[best_period]
        atmp1 = ACF[best_period]
        test[best_period]=0;
        second = np.argmax(test);
        ta2 = bpms[second]
        atmp2 = ACF[second]
        #print(ftmp2)
        a1 = atmp1
        a2 = atmp2
        while(abs(ta1-ta2)<0.08*ta1 or abs(ta1-ta2)<0.08*ta2):
            second = np.argmax(test);
            ta2 = bpms[second]
            a2 = ACF[second]
            test[second]=0;
        if(ta1 > ta2):
            tmp = ta1
            ta1 = ta2
            ta2 = tmp
            tmp = a1
            a1 = a2
            a2 = tmp
        
        
        #########################################Fourier
        
        S = librosa.stft(onset_env, hop_length=1, n_fft=1200)
        fourier_tempogram = np.absolute(S)
        #librosa.display.specshow(fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')
        # method 1 to get bpm (not good)
        
        bpms = librosa.core.tempo_frequencies(fourier_tempogram.shape[0], hop_length=hop_length, sr=sr)
        #calculate mean at each time
        fourier_tempogram = np.mean(fourier_tempogram, axis=1)
        test = np.array([fourier_tempogram[y] for y in range(fourier_tempogram.shape[0])])
        
        max_idx = np.argmax(bpms < 1000)
        test[:max_idx] = 0
        max_idx = np.argmax(bpms < 50)
        test[max_idx:] = 0
        
        best_period = np.argmax(test)
        t1 = bpms[best_period]
        ftmp1 = fourier_tempogram[best_period]
        test[best_period]=0;
        second = np.argmax(test);
        t2 = bpms[second]
        ftmp2 = fourier_tempogram[second]
        test[second]=0;
        #print(ftmp2)
        f1 = ftmp1
        f2 = ftmp2

        while(abs(t1-t2)<0.08*t1 or abs(t1-t2)<0.08*t2):
            second = np.argmax(test);
            t2 = bpms[second]
            f2 = fourier_tempogram[second]
            test[second]=0;
        if(t1 > t2):
            tmp = t1
            t1 = t2
            t2 = tmp
            tmp = ftmp1
            f1 = ftmp2
            f2 = tmp
        
        
        
        ######
        if(abs(ta1 - t2)<=0.08*ta1):
            tempo_fast = ta1
            tempo_slow = t1
        elif(abs(ta2 - t2)<=0.08*ta2):
            tempo_fast = ta2
            tempo_slow = ta1
        elif(abs(ta1 - t1)<=0.08*ta1):
            tempo_fast = ta2
            tempo_slow = ta1
        else:
            tempo_fast = ta2
            tempo_slow = t2
        #tempo_fast = ta2
        """
        score1 = a2
        score2 = f1
        if(abs(tempo_fast-tempo_slow)<0.08*tempo_fast or abs(tempo_fast-tempo_slow)<0.08*tempo_slow ):
            tempo_slow = t2
            score2 =f2
        """
        if(tempo_fast < tempo_slow):
            tmp = tempo_fast
            tempo_fast = tempo_slow
            tempo_slow = tmp
        score1 = fourier_tempogram[int(np.rint(60*sr/(hop_length*tempo_slow)))]
        score2 = fourier_tempogram[int(np.rint(60*sr/(hop_length*tempo_fast)))]
        stmp1 = score1/(score1+score2)
        score1 = ACF[int(np.rint(60*sr/(hop_length*tempo_slow)))]
        score2 = ACF[int(np.rint(60*sr/(hop_length*tempo_fast)))]
        stmp2 = score1/(score1+score2)
    
        s1 = (stmp1+stmp2)/2
        
        tt1 = 0 
        if(abs(ans-tempo_slow)/ans<=0.08):
            tt1 = 1
        tt2 = 0
        if(abs(ans-tempo_fast)/ans<=0.08):
            tt2 = 1
        p1 = s1*tt1+(1-s1)*tt2
        p2 = 0
        if(abs(ans-tempo_slow)/ans<=0.08 or abs(ans-tempo_fast)/ans<=0.08):
            p2=1
        
        average_P1 += p1
        average_P2 += p2
        
        # tempo using librosa
        #b=librosa.beat.tempo(y, sr=sr)
        
        out.write('Estimate: '+str(f)+'\n')
        out.write('Estimate tempo slow: '+str(tempo_slow)+'\n')
        out.write('Estimate tempo fast: '+str(tempo_fast)+'\n')
        out.write('Ans:                 '+str(ans)+'\n')
        out.write('Relative P:          '+str(p1)+'\n')
        out.write('ALOTC:               '+str(p2)+'\n')
        
        print('Estimate: ',f)
        print('Estimate tempo slow: ',tempo_slow)
        print('Estimate tempo fast: ',tempo_fast)
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
