# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:16:01 2024

@author: turunenj
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.0005
max_time=3.0
NFFT = 1024  # the length of the windowing segments
Fs = 1/dt  # the sampling frequency

#time series components
t = np.arange(0.0, max_time, dt)
s1 = np.sin(2 * np.pi * 940 * t)
s2 = 2 * np.sin(2 * np.pi * 630 * t)

# add some noise into the mix
nse = np.random.normal(0,0.1, size=len(t))

combo = s1 + s2 + nse  # the combined time series

#fft of the time series
fft_combo=np.fft.fft(combo)
freq_t=np.linspace(0,Fs,len(fft_combo)) #frequency vector

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
ax1.plot(t[0:300], combo[0:300])
ax1.set_ylabel('Sinusoidal time series')

ax2.plot(freq_t,np.abs(fft_combo))
ax2.set_ylabel('FFT of Sinusoidals')
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(t, combo)
ax1.set_ylabel('time series')

Pxx, freqs, bins, im = ax2.specgram(combo, NFFT=NFFT, Fs=Fs)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the .image.AxesImage instance representing the data in the plot
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlim(0, max_time)

plt.show()


