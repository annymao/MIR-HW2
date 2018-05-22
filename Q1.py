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
for idx in range (1,2):
    #y, sr = librosa.load('data1/BallroomData/ChaChaCha/Media-1034{0:02}.wav'.format(idx))
    #ans = int(open('data2/BallroomAnnotations/ballroomGroundTruth/Media-1034{0:02}.bpm'.format(idx),'r').read())
    y, sr = librosa.load('data1/BallroomData/Rumba-International/Albums-Cafe_Paradiso-09.wav')
    ans = int(open('data2/BallroomAnnotations/ballroomGroundTruth/Albums-Cafe_Paradiso-09.bpm','r').read())
    hop_length = 220
    # novelty curve
    onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=2048)
    # calculate tempogram
    S = librosa.stft(onset_env, hop_length=1, n_fft=1200)
    fourier_tempogram = np.absolute(S)
    #librosa.display.specshow(fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')
    # method 1 to get bpm (not good)
    m = np.max(fourier_tempogram, axis=0)
    
    #calculate mean at each time
    fourier_tempogram[0] = [ 0 for i in range(0, fourier_tempogram.shape[1])]
    fourier_tempogram[1] = [ 0 for i in range(0, fourier_tempogram.shape[1])]
    
    fourier_tempogram = np.mean(fourier_tempogram, axis=1, keepdims=True)
    
    # choose the maximum
    a = np.argmax(fourier_tempogram)
    #translate bin to bpm freauency
    bpms = librosa.core.tempo_frequencies(fourier_tempogram.shape[0], hop_length=hop_length, sr=sr)
    t1 = bpms[a]

    #set the probability of each bpms to normal distribution
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(120)) /1.0)**2)
    
    # cut off too large bpm
    max_idx = np.argmax(bpms < 320)
    prior[:max_idx] = 0
    fourier_tempogram[:max_idx] = 0
    
    # get the two largest one
    test = fourier_tempogram * prior[:, np.newaxis];
    best_period = np.argmax(test, axis=0)
    test[best_period]=0;
    second = np.argmax(test, axis=0);
    #take t2 be the avg of the two
    t2 = (bpms[best_period]+bpms[second])/2
    
    # tempo using librosa
    b=librosa.beat.tempo(y, sr=sr)
    print(np.mean(m),bpms[best_period],bpms[second],t2,t1,b,ans)
    
    





