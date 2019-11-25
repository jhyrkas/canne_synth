import synth_architecture as sa
import numpy as np
import soundfile as sf
import os
import time

def parse_params(params_raw) :
    lengths = params_raw[:,0]
    params_raw = params_raw[:,1:]
    params = np.zeros((0,8))
    audio_len = 0.0
    for i in range(1, params_raw.shape[0]) :
        length = lengths[i-1]
        num_frames = arch.get_num_frames(length)
        temp = np.zeros((num_frames, 8))
        for j in range(8) :
            temp[:,j] = np.linspace(params_raw[i-1,j], params_raw[i, j], num_frames)
        params = np.append(params, temp, axis=0)
        audio_len += length
    # final segment
    if lengths[-1] > 0.0 :
        length = lengths[-1]
        num_frames = arch.get_num_frames(length)
        temp = np.zeros((num_frames, 8))
        for j in range(8) :
            temp[:,j] = params_raw[-1,j]
        params = np.append(params, temp, axis=0)
        audio_len += length

    return params, audio_len

# TODO perhaps i can parameterize this
basepath = '/Users/hyrkas/eclipse-workspace/AlgoMusicHW3/scratch/' 
param_path = basepath + 'params.csv'
wav_path = basepath + 'audio.wav'
arch = sa.Architecture('root', np.zeros(8) + 1.0)

# very hacky
while True :
    if os.path.exists(param_path) :
        params_raw = np.genfromtxt(param_path, delimiter=',')
        params, audio_len = parse_params(params_raw)
        arch.update_params('root', params)
        audio = arch.generate_audio(audio_len)[0]
        sf.write(wav_path, audio, 44100)
        os.remove(param_path)
    time.sleep(0.1)

