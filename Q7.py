# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:10:13 2018

@author: anny
"""
import seaborn
import numpy as np, scipy
import librosa, librosa.display
import  matplotlib.pyplot as plt, IPython.display as ipd
plt.rcParams['figure.figsize'] = (13, 5)
from os import listdir 
from os.path import isfile, join 

def beat_track(spectral,bpm,fft_res,tight,trim):
    period = round(60.0 * fft_res / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore = local_score(spectral, period)
    """
    print(localscore)
    frames = range(len(localscore))
    t = librosa.frames_to_time(frames, sr=fft_res*220, hop_length=220)
    plt.plot(t, localscore)
    plt.xlim(0, t.max())
    plt.ylim(0)
    plt.xlabel('Time (sec)')
    plt.title('Novelty Function')
    """
    # run the DP
    backlink, cumscore = DP(spectral, period, tight)

    # get the position of the last beat
    beats = [last_beat(cumscore)]

    # Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    # Put the beats in ascending order
    # Convert into an array of frame numbers
    beats = np.array(beats[::-1], dtype=int)

    # Discard spurious trailing beats
    #beats = trim_beats(localscore, beats, trim)

    return beats
def normalize_spectral(spectral):
    norm = spectral.std(ddof=1)
    if norm > 0:
        spectral = spectral / norm
    return spectral

def local_score(spectral,period):
    window = np.exp(-0.5 * (np.arange(-period, period+1)*32.0/period)**2)
    return scipy.signal.convolve(normalize_spectral(spectral),
                                 window,'same')
def DP(local_score,period,tight): 
    backlink = np.zeros_like(local_score, dtype=int)
    cumscore = np.zeros_like(local_score)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    txwt = -tight * (np.log(-window / period) ** 2)

    first_beat = True
    for i, score_i in enumerate(local_score):

        z_pad = np.maximum(0, min(- window[0], len(window)))

        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * local_score.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore
def last_beat(cumscore):
    """Get the last beat from the cumulative score array"""
    maxes = librosa.beat.util.localmax(cumscore)
    med_score = np.median(cumscore[np.argwhere(maxes)])

    # The last of these is the last beat (since score generally increases)
    return np.argwhere((cumscore * maxes * 2 > med_score)).max()

 
if __name__ == '__main__':
    path = 'data1/BallroomData/'
    out = open('Q7.txt','w')
    for i in ['ChaChaCha']:#,'Jive','Quickstep','Rumba','Samba','Tango','VienneseWaltz','Waltz']:
        mypath = join(path,i)
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        
        bpmPath = 'BallroomAnnotations/'
        
        average_P1 = 0
        average_P2 = 0
        out.write('Genres: '+i+'\n')
        print("Genres: ",i)
        for f in files[0:1]:
            print(f)
            tmp = f.split('.')
            if tmp[1]=='wav':
                bpmFile = tmp[0] + '.beats'

            y, sr = librosa.load(join(mypath,f))
            with open(join(bpmPath,bpmFile)) as f:
                ans = f.readlines()
            #ans = open(join(bpmPath,bpmFile),'r').read()
            ans=[float(i.rstrip().split(' ')[0]) for i in ans]
            print(ans)
            hop_length = 220
            # novelty curve
            onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=2048)
            bpm = librosa.beat.tempo(onset_envelope=onset_env,
                    sr=sr,
                    hop_length=hop_length)[0]
            beats = beat_track(onset_env,
                           bpm,
                           float(sr) / hop_length,
                           100,
                           True)
            frames = range(len(onset_env))
            """
            t = librosa.frames_to_time(frames, sr=sr, hop_length=220)
            plt.plot(t, onset_env)
            plt.xlim(0, t.max())
            plt.ylim(0)
            
            plt.xlabel('Time (sec)')
            plt.title('Novelty Function')
            """
            beats = librosa.frames_to_time(beats, hop_length=hop_length, sr=sr)
            
            print (beats)
            ntp=0
            nfp=0
            for i in range(0,beats.shape[0]):
                for j in ans:
                    if(abs(j-beats[i])<=0.07):
                        ntp+=1
                        break;
                    if(j - beats[i]>0.07):
                        nfp+=1
                        break;
            print(ntp,nfp)
            out.write(str(beats))
    out.close()