import numpy as np
import dream_lstm
import soundfile as sf

#sig = np.sin(2*np.pi*440.0*np.array(range(88200))/44100)
[sig, fs] = sf.read('minicomp.wav')
sig_mono = np.mean(sig, axis=1)
temp = dream_lstm.TimeDomainLSTM(sig_mono[:44100*30], 1024, 0.1, 0.05)
print('training!')
temp.train_model(5)
print('dreaming!')
samples = temp.dream(temp.x[0,:], 44100*5)
samples = samples / np.max(np.abs(samples))
sf.write('dream.wav', samples, fs)
print('done!')
