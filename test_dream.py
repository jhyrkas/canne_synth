import numpy as np
import dream_lstm
#import soundfile as sf
import wave
from struct import pack, unpack
from random import shuffle

# STEREO TEST
wf_read = wave.open( 'minicomp.wav', 'rb')
fs = wf_read.getframerate()
size = wf_read.getnframes()
tmp_sig = [[0] * size, [0] * size]
for i in range(size) :
    a,b = unpack('hh', wf_read.readframes(1))
    tmp_sig[0][i] = a
    tmp_sig[1][i] = b

sig = np.array(tmp_sig)
sig = sig / 2**15
sig_mono = np.mean(sig, axis=0)
wf_read.close()

# MONO TEST
#wf_read = wave.open( 'second_input_16bit.wav', 'rb')
#fs = wf_read.getframerate()
#size = wf_read.getnframes()
#tmp_sig = unpack('h'*size, wf_read.readframes(size))

#sig = np.array(tmp_sig)
#sig_mono = sig / 2**15
#wf_read.close()

# SIN WAVE TEST
#sig_mono = np.sin(2*np.pi*440.0*np.array(range(44100*15))/44100)

#[sig, fs] = sf.read('minicomp.wav')
#sig_mono = np.mean(sig, axis=1)

print(sig_mono.shape)

temp = dream_lstm.TimeDomainLSTM(sig_mono[0:44100*15], 1024, 0.1, 0.05)

print('training!')
# MEMORY SAVING TECHNIQUE
for i in range(5) :
    # 15 second segments
    num_segments - np.floor(len(sig_mono) / 44100 / 15)
    seg_starts = shuffle(range(num_segments))
    for j in range(num_segments) :
        temp.update_training(sig_mono[seg_starts[j]*44100*15:(seg_starts[j]+1)*44100*15])
        temp.train_model(5)

print('dreaming!')
#samples = temp.dream(temp.x[0,:], 44100*10)
samples = temp.dream(temp.x[0,:], 44100*10)
samples = samples / np.max(np.abs(samples))

wf_write = wave.open('dream.wav', 'w')
wf_write.setnchannels(1)      # one channel (mono)
wf_write.setsampwidth(2)      # two bytes per sample
wf_write.setframerate(44100)   # samples per second
out_sig_tmp = np.clip(samples * (2**15), -(2**15), 2**15 - 1).astype(int).tolist()
wf_write.writeframes(pack('h' * len(out_sig_tmp), *out_sig_tmp))
wf_write.close()
#sf.write('dream.wav', samples, fs)
print('done!')
