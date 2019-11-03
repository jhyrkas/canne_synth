import sounddevice as sd
import soundfile as sf
import compose_utility
import numpy as np
import time

def drunk_walk(n, minn, maxx, max_step, init_val=None) :

    outp = np.zeros(n)
    if init_val is None :
        outp[0] = np.random.randint(minn, maxx)
    else :
        outp[0] = init_val

    for i in range(1, n) :
        mult = np.random.choice([-1, 1])
        step = mult * np.random.random() * max_step
        outp[i] = max(min(maxx, outp[i-1] + step), minn)

    return outp

cu = compose_utility.ComposeUtility()
np.random.seed(69420)

# SECTION 1: GENERATE MELODY

outsig = np.zeros((44100*16*8,2))
# would be good to envelope these
note1 = np.array([1.4, 2.3, 3.2, 2.8, 1.4, 1.3, 1.2, 1.9])
note2 = np.array([1.4, 1.5, 3.2, 2.8, 1.4, 2.3, 1.2, 1.9])
note3 = np.array([1.4, 1.5, 1.8, 2.5, 1.,  0.8, 1.,  1.])
num_frames = cu.get_num_frames_depr(4)
params = np.zeros((8, num_frames))
for i in range(num_frames) :
    if i <= (3/8*num_frames) :
        params[:,i] = note1
    elif i <= (3/4*num_frames) :
        params[:,i] = note2
    else :
        params[:,i] = note3

# TODO: change this so it's all one for the whole song, avoids clicking
# this might involve fixing the note length bug
melody = cu.make_note_osc(np.hstack((params, params, params, params)).transpose(), 16)
stems = []
stems.append(melody.copy())

for i in range(6) :
    outsig[i*len(melody):(i+1)*len(melody), 0] += melody
    outsig[i*len(melody):(i+1)*len(melody), 1] += melody

# quick fade in for first melody
tmp = int(2.5*44100)
outsig[:tmp, 0] *= np.linspace(0, 1, tmp)
outsig[:tmp, 1] *= np.linspace(0, 1, tmp)

# SECTION 2: FADE IN PREDICTED AUDIO ON RIGHT SIDE

simple_pred = cu.predict_audio(melody, None)[0]
fade_in = np.linspace(0, 1.5, len(simple_pred))
outsig[1*len(simple_pred):2*len(simple_pred), 0] += 0.33 * simple_pred * fade_in
outsig[1*len(simple_pred):2*len(simple_pred), 1] += 0.66 * simple_pred * fade_in
outsig[2*len(simple_pred):3*len(simple_pred), 0] += 0.33 * simple_pred * 1.5
outsig[2*len(simple_pred):3*len(simple_pred), 1] += 0.66 * simple_pred * 1.5
stems.append(simple_pred.copy())

# SECTION 3 
params2 = np.zeros((8, num_frames*8))
for i in range(8) :
    params2[i,:] = drunk_walk(num_frames*8, 0, 4, .1, note1[i])
melody_walk = cu.make_note_osc(params2.transpose(), 32)
outsig[2*len(melody):2*len(melody) + len(melody_walk), 0] += 0.66 * melody_walk
outsig[2*len(melody):2*len(melody) + len(melody_walk), 1] += 0.33 * melody_walk
stems.append(melody_walk.copy())

# SECTION 4
params3 = np.zeros((8, num_frames*4))
for i in range(8) :
    params3[i,:] = drunk_walk(num_frames*4, -10, 10, .1, note1[i])
pred_walk = cu.predict_audio(melody, params3.transpose())[0]
outsig[3*len(melody):4*len(melody), 0] += 0.33 * pred_walk
outsig[3*len(melody):4*len(melody), 1] += 0.66 * pred_walk
stems.append(pred_walk.copy())

# SECTION 5
feedback_predl = cu.predict_audio(outsig[3*len(melody):4*len(melody), 0], feedback_rate = 0.5)[0]
feedback_predr = cu.predict_audio(outsig[3*len(melody):4*len(melody), 1], feedback_rate = 0.5)[0]
outsig[4*len(melody):5*len(melody), 0] += 0.75 * feedback_predl
outsig[4*len(melody):5*len(melody), 1] += 0.75 * feedback_predr
stems.append(feedback_predl.copy())
stems.append(feedback_predr.copy())

# SECTION 6
params4 = np.zeros((8, num_frames*4))
for i in range(8) :
    params4[i,:] = drunk_walk(num_frames*4, 0, 4, .1, note1[i])
melody_walk = cu.make_note_osc(params4.transpose(), 16)
for i in range(8) :
    params4[i,:] = drunk_walk(num_frames*4, -10, 10, .1, note1[i])
pred_walk = cu.predict_audio(melody_walk, params4.transpose(), feedback_rate = 0.25)[0]
fade_in = np.linspace(1, 2.5, len(melody_walk))
outsig[5*len(melody):6*len(melody), 0] += fade_in * melody_walk
outsig[5*len(melody):6*len(melody), 1] += fade_in * pred_walk
stems.append(melody_walk.copy())
stems.append(pred_walk.copy())

# SECTION 7-8
params5 = np.zeros((8, num_frames*8))
for i in range(8) :
    params5[i,:] = drunk_walk(num_frames*8, 0, 4, .1, note1[i])
melody_walk = cu.make_note_osc(params5.transpose(), 32)
for i in range(8) :
    params5[i,:] = drunk_walk(num_frames*8, -10, 10, .1, note1[i])
pred_walk = cu.predict_audio(melody_walk, params5.transpose(), feedback_rate = 0.25)[0]

outsig[6*len(melody):6*len(melody)+len(melody_walk), 0] += melody_walk
outsig[6*len(melody):6*len(melody)+len(melody_walk), 1] += pred_walk

stems.append(melody_walk.copy())
stems.append(melody_walk.copy())

for i in range(8) :
    params5[i,:] = drunk_walk(num_frames*8, 0, 4, .1, note1[i])
melody_walk = cu.make_note_osc(params5.transpose(), 32)
outsig[6*len(melody):6*len(melody)+len(melody_walk), 0] += 0.66 * melody_walk
outsig[6*len(melody):6*len(melody)+len(melody_walk), 1] += 0.33 * melody_walk
stems.append(melody_walk.copy())

simple_pred = cu.predict_audio(melody, feedback_rate = 0.1)[0]
outsig[6*len(melody):6*len(melody)+len(simple_pred), 0] += 0.33 * simple_pred
outsig[6*len(melody):6*len(melody)+len(simple_pred), 1] += 0.66 * simple_pred
stems.append(simple_pred.copy())

final_pred = cu.predict_audio(melody, feedback_rate = 0.5)[0]
final_pred *= np.linspace(0.5, 2, len(final_pred))
outsig[7*len(melody):7*len(melody)+len(final_pred), 0] += 0.33 * final_pred
outsig[7*len(melody):7*len(melody)+len(final_pred), 1] += 0.66 * final_pred
stems.append(final_pred.copy())

# SECTION 0: INTRO & OUTRO
intro_sig = np.zeros((44100*20,2))
num_frames = cu.get_num_frames_depr(20)
intro_params = np.zeros((8, num_frames))
for i in range(8) :
    intro_params[i,:] = drunk_walk(num_frames, 0, 4, .07, np.random.random()*4.0)
intro_walk = cu.make_note_osc(intro_params.transpose(), 20)
intro_pred = cu.predict_audio(intro_walk)[0]
intro_pred *= np.hstack((
    np.linspace(0, 0.75, 3*len(intro_walk)//4), np.linspace(0.75, 0, len(intro_pred)//4)))
intro_sig[:len(intro_pred), 0] += intro_pred
intro_sig[:len(intro_pred), 1] += intro_pred

stems.append(intro_pred.copy())

outsig[:len(intro_walk)//4,0] += intro_pred[3*len(intro_walk)//4:]
outsig[:len(intro_walk)//4,1] += intro_pred[3*len(intro_walk)//4:]

outro_sig = np.zeros((44100*5,2))
fake_signal = cu.make_note_osc(note1, 5)
outro_pred = cu.predict_audio(fake_signal, feedback_rate=0.9, use_prev_frame=True)[0]
outro_sig[:len(outro_pred),0] = 0.33*outro_pred*np.linspace(2, 0, len(outro_pred))
outro_sig[:len(outro_pred),1] = 0.66*outro_pred*np.linspace(2, 0, len(outro_pred))
stems.append(outro_sig.copy())

# NOTE: this part is ugly because times are not working out exactly as i hoped...might be able to fix later
j = outsig.shape[0] - 1
counter = 0
while outsig[j,0] == 0.0 and outsig[j,1] == 0.0 :
    counter += 1
    j -= 1
print('off by ' + str(counter) + ' samples')

outsig = np.vstack((intro_sig[:3*len(intro_walk)//4,:], outsig[:j+1,:], outro_sig))

# normalize
mv = np.max(outsig)
outsig /= mv

# write to files

for i in range(len(stems)) :
    sf.write('minicomp_stems/stem' + str(i) + '.wav', stems[i], 44100, subtype='PCM_16')

sf.write('minicomp_stems/reference_mix.wav', outsig, 44100, subtype='PCM_16')

# play
#sd.play(outsig, 44100)

#time.sleep(139)
