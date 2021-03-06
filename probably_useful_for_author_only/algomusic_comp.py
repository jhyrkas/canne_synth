import synth_architecture as sa
import numpy as np
import soundfile as sf
import os
import time

def parse_params(params_raw) :
    # first line is a dummy line to get the pitch information
    pitch_shift = params_raw[0,-1]
    lengths = params_raw[1:,0]
    params_raw = params_raw[1:,1:]
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

    return params, audio_len, pitch_shift

basepath = '/Users/hyrkas/eclipse-workspace/AlgoMusicFinal/scratch/' 
param_path = basepath + 'params.csv'
wav_path = basepath + 'audio.wav'
done_path = basepath + 'done'
arch = sa.Architecture('root', np.zeros(8) + 1.0)

# very hacky
while True :
    if os.path.exists(param_path) :
        params_raw = np.genfromtxt(param_path, delimiter=',')
        params, audio_len, pitch_shift = parse_params(params_raw)
        arch.update_params('root', params)
        audio = arch.generate_audio(audio_len, pitch_shift)[0]
        sf.write(wav_path, audio, 44100)
        f = open(done_path, 'w')
        f.close()
        os.remove(param_path)
    time.sleep(0.1)

