import FFT
from scipy.fft import fft
import matplotlib
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sounddevice as sd

matplotlib.use('TkAgg')  # backend interativo

##############################################################################################

def achar_x_e_y_maximo_em_um_intervalo(funcao, x, xmin, xmax):
    
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(funcao)
    x_pico = x[peaks]
    y_pico = funcao[peaks]
    for j in range (len(x_pico)):
        if(x_pico[j]< xmax and x_pico[j]>xmin):
            return x_pico[j], y_pico[j]
    return 0,0
##############################################################################################

SAMPLERATE = 44.1e3
DURATION = 0.1
FRAMES = int(DURATION*SAMPLERATE)

# Criar streams de input e output
input_stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32', blocksize=FRAMES)
output_stream = sd.OutputStream(samplerate=SAMPLERATE, channels=1, dtype='float32', blocksize=FRAMES)

input_stream.start()
output_stream.start()


# ==== FUNCAO DE ATUALIZACAO ====
def update():
    audio, _ = input_stream.read(FRAMES)
    audio = audio.flatten()
    
    windowed = audio * np.hanning(len(audio))
    X = FFT.fft_completa(windowed)

    freqs = FFT.frequencia(SAMPLERATE,X)
    magnitude = np.abs(X[0:len(X)//2])

    return audio, magnitude, freqs


plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
line_audio, = ax1.plot(np.zeros(FRAMES), color = 'b', linewidth = 0.5)
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(0,FRAMES)
ax1.set_xlabel("$k$")
ax1.set_ylabel("$Audio(k)$")
ax1.grid(True)

line_spectral, = ax2.plot(np.zeros( 2**mt.floor(np.log2(FRAMES)) ),color = 'r', linewidth = 0.5)

ax2.axvline(x = 82.40, color = 'k', linewidth = 0.5)
ax2.axvline(x = 110.00, color = 'k', linewidth = 0.5)
ax2.axvline(x = 146.83, color = 'k', linewidth = 0.5)
ax2.axvline(x = 195.99, color = 'k', linewidth = 0.5)
ax2.axvline(x = 246.94, color = 'k', linewidth = 0.5)
ax2.axvline(x = 329.63, color = 'k', linewidth = 0.5)

ax2.set_ylim(-1,70)
ax2.set_xlim(0,2000)
ax2.set_xlabel("$f (Hz)$")
ax2.set_ylabel("$|FFT|$")
ax2.grid(True)

ax2.xaxis.set_major_locator(MultipleLocator(100)) 

plt.show()

while True:
    audio, magnitude, freqs = update()
    line_audio.set_ydata(audio)

    line_spectral.set_ydata(magnitude)
    line_spectral.set_xdata(freqs)

    plt.pause(0.01)
