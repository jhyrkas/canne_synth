import synth_architecture as sa
import numpy as np
import sounddevice as sd
import time

p = np.zeros(8) + 1.0
a = sa.Architecture('root', p)
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

a.add_network('c1', 'root')
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

a.update_envelope('exp')
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

a.update_envelope(None)
p[2] = 1.5
p[7] = 2.0
a.update_params('root', p)
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

p2 = np.zeros(8) + 0.2
a.update_params('c1', p2)
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

p3 = np.zeros((800, 8))
p3[:,2] = np.linspace(0, 6.0, 800)
a.update_params('c1', p3)
audio = a.generate_audio(5)
assert (len(audio) == 1)
sd.play(audio[0], 44100)
time.sleep(6)

p[0] = 2.7
p[1] = 0.3
a.update_params('root', p)
a.add_network('c2', 'root', p2, 0.5)
audio = a.generate_audio(5)
assert(len(audio) == 2)
a2 = np.vstack((np.array(audio[0]),np.array(audio[1])))
sd.play(a2.transpose() , 44100)
time.sleep(6)


