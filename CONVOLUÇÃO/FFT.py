# Defino a exponencial complexa
import numpy as np
import math as mt

def Wn(N):
    wn = np.exp(-1j*2*np.pi/N)
    return wn

def Wn_inverso(N):
    wn = np.exp(1j*2*np.pi/N)
    return wn

def reordenar (y, N = None):

    aux = 2**mt.ceil(np.log2(len(y)))
    zero_padding = aux - len(y) # quantidade de zeros que deve ser acrescentado para y ficar do mesmo tamanho que aux
    if (N != None):
        zero_padding = zero_padding + N - aux # quantidade de zeros que deve ser acrescentado para ficar do tamanho que o usuario quiser
    y = np.append(y,np.zeros(zero_padding))
    y_reordenado = []

    for i in range(len(y)):
        #para a inversao de bits, tem que se saber quantos zeros deve ser adicionado antes.
        #Ex: Se N=16, o indice 5 em binario sera 101, porem, N=16 exige representacao com 4 bits, 
        #logo um zero deve ser adicionado a esquerda, ficando 0101 

        qtd_zeros_binario = int(np.log2(len(y))) - len(bin(i)[2:]) 

        indice_em_binario = bin(i)[2:]

        for j in range(int(qtd_zeros_binario)):
            #Preenche com zeros a esquerda
            indice_em_binario = '0' + indice_em_binario

        #Aqui eh feita a inversao dos bits, ou seja, 0101 (5) vira 1010 (10) e depois é convertido para decimal. 
        y_reordenado.append(y[int(indice_em_binario[::-1],2)])
    return y_reordenado

def functionFFT(y):
    
    #eh preciso saber o valor de N para que tenha fim na recursividade quando N=1.
    N = int(len(y))

    X = np.zeros(N, dtype= complex)
    
    #Quando N=1 retorno o proprio sinal
    if(N==1):
        return y
    
    # print("y = ",y,"\n")

    #y ja eh reordenado, ou seja, a primeira metade do vetor sao os pares e a outra metade os impares
    par = y[0:N//2]
    impar = y[N//2::]

    # print("y pares = ", par,"\n")
    
    #aqui eh feita a recursividade
    Ak = functionFFT(par)
    Bk = functionFFT(impar)
    wn = Wn(N)

    # print(wn)
    for k in range(N//2):
        p = Bk[k]*(wn)**k
        X[k] = Ak[k] + p
        X[k+N//2] = Ak[k] - p
        # print(k,"A= ", Ak[k],"B = ", Bk[k], f"X({k}) = ", X[k], f"X({k+N//2}) = ", X[k+N//2])
    return X

def fft_inversa(y):    
    #eh preciso saber o valor de N para que tenha fim na recursividade quando N=1.
    N = int(len(y))

    X = np.zeros(N, dtype= complex)
    
    #Quando N=1 retorno o proprio sinal
    if(N==1):
        return y
    
    # print("y = ",y,"\n")

    #y ja eh reordenado, ou seja, a primeira metade do vetor sao os pares e a outra metade os impares
    par = y[0:N//2]
    impar = y[N//2::]

    # print("y pares = ", par,"\n")
    
    #aqui eh feita a recursividade
    Ak = fft_inversa(par)
    Bk = fft_inversa(impar)
    wn = Wn_inverso(N)

    # print(wn)
    for k in range(N//2):
        p = Bk[k]*(wn)**k
        X[k] = Ak[k] + p
        X[k+N//2] = Ak[k] - p
        # print(k,"A= ", Ak[k],"B = ", Bk[k], f"X({k}) = ", X[k], f"X({k+N//2}) = ", X[k+N//2])
    return X

def fft_completa(y, N = None):
    y_reordenado = reordenar(y, N)
    FFT = functionFFT(y_reordenado)
    return FFT

def fft_inversa_completa(y):
    y_reordenado = reordenar(y)
    FFT_RESULTADO = 1/(len(y_reordenado))*fft_inversa(y_reordenado)
    return FFT_RESULTADO

def convolucao(x1,x2):
    M = len(x1)+len(x2)-1
    zeros_x1 = 2**mt.ceil(np.log2(M)) - len(x1)
    zeros_x2 = 2**mt.ceil(np.log2(M)) - len(x2)

    x1 = np.append(x1, np.zeros(zeros_x1))
    x2 = np.append(x2, np.zeros(zeros_x2))

    FFT_x1 = fft_completa(x1)
    FFT_x2 = fft_completa(x2)

    convolucao = fft_inversa_completa(FFT_x1 * FFT_x2)[0:M]

    return convolucao
    
def overlap(M, sinal_longo, x2):
    #determina o tamanho de cada secao do sinal longo
    Tamanho_dos_blocos = M 

    #tamanho do sinal resultante da convolucao
    Tamanho_da_convolucao = len(sinal_longo) + len(x2) - 1
    #determina quantas secoes cabem no tamanho da convolucao estipulada, ou seja, quantos passos serao realizados
    qtd_de_passos = mt.ceil(len(sinal_longo)/Tamanho_dos_blocos)
    #vetor overlap do tamanho resultante da convolucao
    overlap = np.zeros(Tamanho_da_convolucao, dtype = complex)

    for i in range(qtd_de_passos):
        indice = i*Tamanho_dos_blocos
        #calculo de convolucao  do sinal longo truncado em M com o sinal x2
        x1_conv_x2 = convolucao(sinal_longo[indice:indice+Tamanho_dos_blocos],x2)
        #preencher o vetor overlap
        overlap[indice:indice + len(x1_conv_x2)] += x1_conv_x2 
    
    return overlap

def frequencia(fs, x_fft):
    N = len(x_fft)
    frequencias = np.zeros(N//2)
    for i in range(N//2):
        frequencias[i] = fs/N * i
    return frequencias