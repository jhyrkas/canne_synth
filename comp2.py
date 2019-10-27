# currently unnamed compositional system
# jeremy hyrkas, 2019

import random
from collections import Counter
import numpy as np
import synth_architecture as sa
from scipy.signal import sawtooth
import sounddevice as sd

# english letter frequency according to http://letterfrequency.org/
alphabet = 'abcdefghijklmnopqrstuvwxyz'
letter_freq = ['e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'j', 'q', 'z']
letter_dict = {}
for i in range(26) :
    letter_dict[alphabet[i]] = i+1
pitch_dict = {}
for i in range(25) :
    pitch_dict[alphabet[i]] = i - 12

def get_params_from_word(orig_word, arch) :
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

    num_frames = arch.get_num_frames(length)
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
    params['pitch'] = pitch_dict[most_common_letter] if not most_common_letter == 'z' else (random.random() * 24) - 12

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
        params['envelope'] = (a_time, a_level, d_time, s_level, r_time)

    return params

if __name__ == '__main__' :
    arch = sa.Architecture('root', np.zeros(8) + 1.0)
    arch.add_network('car', 'root')

    while True :
        inp = input('type whatever: ')
        inp = ''.join(c for c in inp if c.isalpha())
        if len(inp) > 0 :
            params = get_params_from_word(inp, arch)
            print (params)
            arch.update_params('root', params['mod'])
            arch.update_params('car', params['car'])
            arch.update_envelope(params['envelope'])
            arch.update_feedback('car', params['feedback'])
            audio = arch.generate_audio(params['length'], params['pitch'])[0]
            sd.play(audio, 44100)
