import numpy as np
import dream_lstm
#import soundfile as sf
import wave
from struct import pack, unpack

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
#sig_mono = np.mean(sig, axis=0)
wf_read.close()
#sig_mono = np.sin(2*np.pi*440.0*np.array(range(88200))/44100)
#[sig, fs] = sf.read('minicomp.wav')
sig_mono = np.mean(sig, axis=1)
temp = dream_lstm.TimeDomainLSTM(sig_mono, 1024, 0.1, 0.05)
print('training!')
temp.train_model(10)
print('dreaming!')
samples = temp.dream(temp.x[0,:], 44100*10)
samples = samples / np.max(np.abs(samples))

wf_write = wave.open('dream.wav', 'w')
wf_write.setnchannels(1)      # one channel (mono)
wf_write.setsampwidth(2)      # two bytes per sample
wf_write.setframerate(44100)   # samples per second
out_sig_tmp = np.clip(samples * (12**15), -(2**15), 2**15 - 1).astype(int).tolist()
wf_write.writeframes(pack('h' * len(out_sig_tmp), *out_sig_tmp))
wf_write.close()
#sf.write('dream.wav', samples, fs)
print('done!')
