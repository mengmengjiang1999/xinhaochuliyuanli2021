from typing import final
import librosa
import numpy as np

import soundfile as sf

sourcename = ['./source/0.wav','./source/1.wav','./source/2.wav','./source/3.wav']

filename = ['./wav/0.wav','./wav/1.wav','./wav/2.wav','./wav/3.wav']

N = 1

# 一共30s语音，每一帧有Ns，语音分成T段
T = int(30/N)

# 对音频进行预处理(已完成)
# 原始音频：'./source/*.wav'
# 处理后：'./wav/*.wav'
def pre():
    for i in range(4):
        y, sr = librosa.load(sourcename[i])
        sr2 = 8000
        y_8k = librosa.resample(y, sr, sr2)

        # librosa.output.write_wav('test.wav', y_8k, 8000)
        start = 0
        duration = 30
        sf.write('./wav/%d.wav'%i, y_8k[start*sr2:duration*sr2], sr2, 'PCM_24')

def pin():
    SR = 8000

    # list, 有4份语音
    fileorigin = []
    for i in range(4):
        y, sr = librosa.load(filename[i], sr=SR)
        fileorigin.append(y)

    # T帧，每帧有4节，fft后的结果
    filecuts = []
    for j in range(T):
        fft_result = []
        # 每个f里面有4个
        for i in range(4):
            # 先取出来第j帧
            fft_result.append(fileorigin[i][j*SR*N:(j+1)*SR*N])
            fft_result[i] = np.fft.fft(fft_result[i],N*SR)
        filecuts.append(fft_result)

    pinqilais = []
    for j in range(T):
        pinqilai = []
        fft_result = filecuts[j]
        for i in range(4):
            pinqilai.extend(fft_result[i][0:int(N*SR/2)])
        for i in range(4-1,-1,-1):
            pinqilai.extend(fft_result[i][int(N*SR/2):N*SR])
        pinqilai = np.fft.ifft(pinqilai, 4*N*SR)
        pinqilais.extend(pinqilai)
        # sf.write('pin%d.wav'%j, np.real(pinqilai), 48000, 'PCM_24')

    sf.write('./wav/pin.wav', np.real(pinqilais), 48000, 'PCM_24')

def fen():
    SR = 8000

    y, sr = librosa.load('./wav/pin.wav', sr=48000)

    pinqilais = []
    for j in range(T):
        pinqilais.append(y[j * 4 * N * SR: (j+1) * 4 * N * SR])
        pinqilais[j]=np.fft.fft(pinqilais[j], 4*N*SR)

    # 按照频率分段取出来，得到T帧，每帧里面有4段
    # ifft
    results = []
    for j in range(T):
        result = []
        for i in range(4):
            result.append([])
        for i in range(4):
            result[i].extend(pinqilais[j][i*int(N*SR/2):(i+1)*int(N*SR/2)])
        for i in range(4-1,-1,-1):
            result[i].extend(pinqilais[j][2*N*SR+(4-1-i)*int(N*SR/2):2*N*SR+(4-i)*int(N*SR)])
        # fft
        for i in range(4):
            result[i]=np.fft.ifft(result[i],N*SR)
        results.append(result)
    
    # 得到4段音频
    merge_results = [] 
    for i in range(4):
        ifft_result = []
        for j in range(T):
            ifft_result.extend(results[j][i])
        merge_results.append(ifft_result)

    for i in range(4):
        sf.write('./result/%d.wav'%i, np.real(merge_results[i]), SR, 'PCM_24')
    
pin()
fen()

def fftest():
    for i in range(4):
        y, sr = librosa.load(filename[i],sr=8000)
        result = np.fft.fft(y,240000)
        iresult = np.fft.ifft(result, 240000)
        print(iresult[0:100])
        sf.write('./result/%d_test.wav'%i, np.real(iresult), 8000, 'PCM_24')
