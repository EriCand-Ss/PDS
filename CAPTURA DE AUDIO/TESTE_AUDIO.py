import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys
import FFT

# ==== CONFIGURAÇÃO ====
SAMPLERATE = 44100
DURATION = 0.1  # 50 ms
FRAMES = int(SAMPLERATE * DURATION)

# ==== APP E JANELA ====
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="Visualizador de Áudio em Tempo Real")
win.resize(2000, 1200)

# ==== PLOT TEMPO ====
plot_time = win.addPlot(title="Sinal no Tempo")
curve_time = plot_time.plot(pen='y')
plot_time.setYRange(-1, 1)  # faixa mais ampla para melhor visualização
plot_time.setLabel('left', 'Amplitude')
plot_time.setLabel('bottom', 'Tempo', units='amostras')

# ==== PLOT FREQUÊNCIA ====
win.nextRow()
plot_freq = win.addPlot(title="Espectro de Frequência (FFT)")
curve_freq = plot_freq.plot(pen='c')
plot_freq.setYRange(0, 100)  # Ajustável dependendo do sinal
plot_freq.setXRange(0, 5000)  # até 1 kHz
plot_freq.setLabel('left', 'Magnitude')
plot_freq.setLabel('bottom', 'Frequência', units='Hz')

win.show()

# ==== VARIÁVEL PARA ÁUDIO ATUAL ====
last_audio = np.zeros(FRAMES, dtype='float32')

# ==== CALLBACK DUPLEX ====
def audio_callback(indata, outdata, frames, time, status):
    global last_audio
    if status:
        print(status)
    last_audio = indata[:, 0].copy()  # salva para plotar
    outdata[:] = indata  # manda o áudio para saída (monitoramento)

# ==== STREAM DUPLEX ====
stream = sd.Stream(samplerate=SAMPLERATE,
                   blocksize=FRAMES,
                   channels=1,
                   dtype='float32',
                   callback=audio_callback)
stream.start()

# ==== FUNÇÃO DE ATUALIZAÇÃO ====
def update():
    audio = last_audio

    # Atualiza forma de onda no tempo
    curve_time.setData(audio)

    # FFT com janela de Hanning
    windowed = audio * np.hanning(len(audio))
    X = FFT.fft_completa(windowed)

    freqs = FFT.frequencia(SAMPLERATE,X)

    magnitude = np.abs(X[0:len(X)//2])

    curve_freq.setData(freqs, magnitude)

# ==== TIMER PARA ATUALIZAÇÃO ====
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(DURATION * 1000))  # 50 ms

# ==== EXECUTA APLICAÇÃO ====
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()

