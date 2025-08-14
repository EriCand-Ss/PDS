import time
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import sys
import FFT
import matplotlib
matplotlib.use('TkAgg')  # força backend interativo

SAMPLERATE = 44.1e3
DURATION = 0.05
FRAMES = int(DURATION*SAMPLERATE)

stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32', blocksize=int(FRAMES))
stream.start()


# ==== FUNÇÃO DE ATUALIZAÇÃO ====
def update():
    audio, _ = stream.read(FRAMES)
    audio = audio.flatten()
    
    # FFT com janela de Hanning
    windowed = audio * np.hanning(len(audio))
    X = FFT.fft_completa(windowed)

    freqs = FFT.frequencia(SAMPLERATE,X)
    # threshold = 5 # ajuste conforme necessário (teste!)
    magnitude = np.abs(X[0:len(X)//2])

    return audio, magnitude, freqs

plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6))
line_audio, = ax1.plot(np.zeros(FRAMES))
ax1.set_ylim(-1,1)

line_spectral, = ax1.plot(np.zeros(FRAMES))
ax1.set_ylim(-1,1)

plt.show()

while True:
    audio, magnitude, freqs = update()
    time.sleep(DURATION)
    line_audio.set_ydata(audio)

    # line_spectral.set_ydata(freqs,magnitude)
    plt.pause(0.01)
