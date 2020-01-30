import numpy as np
import synth_architecture as sa
import sounddevice as sd
import soundfile as sf
import librosa
import os
import scipy

np.random.seed(20855144)
#np.random.seed(212)
path = 'comp3_audio/'
if not os.path.exists(path) :
    os.mkdir(path)

# ARCHITECTURE
a = sa.Architecture('root', np.zeros(8))
a.add_network('channel1_carr', 'root')
a.add_network('channel2_pred', 'root', predictive_feedback_mode=True)
a.add_network('channel3_carr', 'root')
a.add_network('channel3_pred', 'channel3_carr', predictive_feedback_mode=True)
a.add_network('channel4_pred', 'root', predictive_feedback_mode=True)
a.add_network('channel4_car', 'channel4_pred')
a.add_network('channel5_pass', 'root', passthrough_network=True)

carr_names = ['channel1_carr', 'channel3_carr', 'channel4_car']

# NOTES

def gen_params(time, frames, low, high, osc, freq, phase) :
    ohm = np.linspace(0, time, frames) * 2.0 * np.pi + phase
    if osc == 'sin' :
        sig = np.sin(ohm)
    elif osc == 'square' :
        sig = scipy.signal.square(ohm)
    elif osc == 'saw' :
        sig = scipy.signal.sawtooth(ohm)
    else :
        sig = scipy.signal.sawtooth(ohm, width=0.5)

    return ((sig + 1.0) / 2.0) * (high - low) + low

osces = ['sin', 'square', 'saw', 'triangle']
audio_length = 30
nframes = a.get_num_frames(audio_length)
params = np.zeros((nframes, 8))
pitches = np.linspace(-24, 12, 12) 

for i in range(1, 11) :
    # root params
    for j in range(8) :
        lo,hi = np.random.random(2) * (3.9 + i / 10)
        if lo > hi :
            lo,hi = hi,lo
        osc = osces[np.random.randint(4)]
        freq = np.random.random()
        phase = np.random.random() * 2.0 * np.pi
        params[:,j] = gen_params(audio_length, nframes, lo, hi, osc, freq, phase)
    a.update_params('root', params)
    
    # carrier params
    for carrier in carr_names :
        for j in range(8) :
            lo,hi = np.random.random(2) *  (8.0 + i * 2.0)
            if lo > hi :
                lo,hi = hi,lo
            osc = osces[np.random.randint(4)]
            freq = np.random.random()
            phase = np.random.random() * 2.0 * np.pi
            params[:,j] = gen_params(audio_length, nframes, lo, hi, osc, freq, phase)
        a.update_params(carrier, params)
    channels = a.generate_audio(audio_length, pitches[i])
    assert(len(channels) == 5)
    for channel in range(5) :
        sf.write(path + 'channel' + str(channel+1) + '_' + str(i) + '.wav', channels[channel], 44100)


# ONE LAST DRONE
a.update_params('root', np.random.random(8) * 4.0)
sf.write(path + 'drone.wav', a.generate_audio(300, -18)[1], 44100)
