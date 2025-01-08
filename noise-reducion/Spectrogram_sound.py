# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:16:01 2024

@author: turunenj
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
#there are another toolboxes for reading the audio this is just one of them


Fs, data = wav.read('Kuusi.wav')

t=np.linspace(0,len(data)/Fs,len(data))

data = data / 32767 # 2**15 - 1
#to get the data between -1...1

NFFT=1024 #window size
noverlap=128 #overlapping between windows
mode='psd'
scale='dB'

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t, data)
ax1.set_ylabel('time series')
ax1.set_title('/Kuusi/')
Pxx, freqs, bins, im = ax2.specgram(data, NFFT=NFFT, Fs=Fs, noverlap=noverlap, mode=mode, scale=scale)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the .image.AxesImage instance representing the data in the plot
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Spectrogram of /Kuusi/')
plt.show()


