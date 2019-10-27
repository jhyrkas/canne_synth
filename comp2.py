# currently unnamed compositional system
# jeremy hyrkas, 2019

import queue
import random
from collections import Counter
import threading
import numpy as np
import scipy
import synth_architecture as sa
from scipy.signal import sawtooth
import sounddevice as sd
import soundfile as sf
import librosa
import time

# english letter frequency according to http://letterfrequency.org/
alphabet = 'abcdefghijklmnopqrstuvwxyz'
letter_freq = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']
letter_dict = {}
for i in range(26) :
    letter_dict[alphabet[i]] = i+1
pitch_dict = {}
for i in range(25) :
    pitch_dict[alphabet[i]] = i - 12

def get_params_from_word(orig_word) :
    word = orig_word.lower()
    counts = Counter(word)
    global letter_freq
    global letter_dict
    global pitch_dict
    params = {}
    # length param
    length = len(word)
    params['length'] = length

    # modulation and carrier params

    num_frames = get_num_frames_static(length)
    mod_params = np.zeros((num_frames, 8))
    word_index = 0

    for i in range(8) :
        count = counts[letter_freq[i]]
        ratio1 = letter_dict[word[word_index]] / letter_dict[word[(word_index - 1) % length]] % 5
        ratio2 = letter_dict[word[(word_index + 1) % length]] / letter_dict[word[word_index]] % 5
        if count == 1 :
            mod_params[:, i] = ratio1
        else :
            mod_params[:, i] = (ratio1 + ratio2)/2 + (ratio1 - ratio2)/2 * sawtooth(2*np.pi*count/2*np.linspace(0, 1, num_frames), width=0.5)
        word_index = (word_index + 1) % length
    params['mod'] = mod_params

    car_params = np.zeros((num_frames, 8))
    for i in range(8) :
        count = counts[letter_freq[i+8]]
        ratio1 = letter_dict[word[word_index]] / letter_dict[word[(word_index + 1) % length]] % 5
        ratio2 = letter_dict[word[(word_index - 1) % length]] / letter_dict[word[word_index]] % 5
        if count == 1 :
            car_params[:, i] = ratio1
        else :
            car_params[:, i] = (ratio1 + ratio2)/2 + (ratio1 - ratio2)/2 * sawtooth(2*np.pi*count/2*np.linspace(0, 1, num_frames), width=0.5)
        word_index = (word_index + 1) % length
    params['car'] = car_params

    # pitch param
    most_common_letter = counts.most_common()[0][0]
    #params['pitch'] = pitch_dict[most_common_letter] if not most_common_letter == 'z' else (random.random() * 24) - 12
    params['pitch'] = pitch_dict[most_common_letter] if not most_common_letter == 'z' else (random.random() * 12) - 6

    # feedback
    params['feedback'] = sum([1 if counts[letter_freq[i+16]] > 0 else 0 for i in range(10)]) / 10

    # envelope
    if length < 3 or not orig_word.islower():
        params['envelope'] = 'exp'
    else :
        first_letter = word[0]
        last_letter = word[-1]
        mid_letter = word[length//2]
        vowel_total = counts['a'] + counts['e'] + counts['i'] + counts['o'] + counts['u']
        word_total = sum(counts.values())
        s_level = vowel_total / word_total
        a_level = 1.0 - s_level
        a_time = abs(letter_dict[mid_letter] - letter_dict[first_letter]) / 50 # / 25 * 0.5
        r_time = abs(letter_dict[last_letter] - letter_dict[mid_letter]) / 50 # / 25 * 0.5
        d_time = 1.0 - a_time-r_time
        # TODO: comment out asserts after testing
        assert(a_time >= 0.0 and a_time <= 1.0)
        assert(r_time >= 0.0 and r_time <= 1.0)
        assert(d_time >= 0.0 and d_time <= 1.0)
        #params['envelope'] = (a_time, a_level, d_time, s_level, r_time)
        params['envelope'] = (a_time*length, a_level, d_time*length, s_level, r_time*length)

    return params

def get_num_frames_static(length) :
    spec = librosa.core.stft(np.sin(2*np.pi*np.linspace(0, length, 44100*length)), n_fft = 4096, hop_length = 1024)
    return spec.shape[1]

def handle_params(q, device, mode) :
    arch = sa.Architecture('root', np.zeros(8) + 1.0)
    arch.add_network('car', 'root')
    reverb_sig, fs = sf.read('ir.wav')
    reverb_sig = np.mean(reverb_sig, axis = 1)
    s = sd.OutputStream(samplerate=44100, device=device, channels=2)
    s.start()
    while True :
        time.sleep(0.25)
        if not q.empty() :
            word = q.get()
            # hacky
            if word is None :
                break
            params = get_params_from_word(word)
            arch.update_params('root', params['mod'])
            arch.update_params('car', params['car'])
            arch.update_envelope(params['envelope'])
            arch.update_feedback('car', params['feedback'])
            mono = scipy.signal.fftconvolve(arch.generate_audio(params['length'], params['pitch'])[0], reverb_sig)
            #mono = arch.generate_audio(params['length'], params['pitch'])[0]
            audio = np.zeros((mono.shape[0], 2))
            if mode == 0 :
                audio[:,0] = 0.5 * mono
                audio[:,1] = 0.5 * mono
            elif mode == 1 :
                audio[:,0] = 0.66 * mono
                audio[:,1] = 0.33 * mono
            elif mode == 2:
                audio[:,0] = 0.33 * mono
                audio[:,1] = 0.66 * mono
            elif mode == 3:
                audio[:,0] = mono
            else :
                audio[:,1] = mono
            s.write(np.float32(audio))

if __name__ == '__main__' :
    queues = [queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()]
    q_index = 0

    #queue = queue.Queue()

    devices = sd.query_devices()
    device = 0
    for i in range(len(devices)) :
        #if devices[i]['name'] == 'Soundflower (2ch)' :
        if devices[i]['name'] == 'Built-in Output' :
            device = i

    t1 = threading.Thread(target=handle_params, args=(queues[0],device,0))
    t2 = threading.Thread(target=handle_params, args=(queues[1],device,1))
    t3 = threading.Thread(target=handle_params, args=(queues[2],device,2))
    t4 = threading.Thread(target=handle_params, args=(queues[3],device,3))
    t5 = threading.Thread(target=handle_params, args=(queues[4],device,4))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    inp = input('type whatever: ')
    while not inp == 'theend' :
        inps = inp.split()
        for i in inps :
            word = ''.join(c for c in i if c.isalpha())
            if len(inp) > 0 :
                #queue.put(params)
                queues[q_index].put(word)
                q_index = (q_index + 1) % 5
        inp = input('type whatever: ')
    #queue.put(None)
    for i in [0,1,2,3,4] :
        queues[i].put(None)
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
