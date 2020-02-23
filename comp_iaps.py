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
import sys
from select import select

# imports for google code
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from oauth2client.service_account import ServiceAccountCredentials
import pickle
import os.path

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# The ID and range of a sample spreadsheet.
#SAMPLE_SPREADSHEET_ID = '1uei2BUZZZj3I4Gm09VAA6JrxeRRnrWy7tcDTSWvpr_Y'
SAMPLE_SPREADSHEET_ID = '1ysrVMCaUbQ23foUH8_d35jKkZouEKFvthqQkqrdw6TY'
SAMPLE_RANGE_NAME = 'A1:J25'
NUMROWS = 25
NUMCOLS = 10
# MAKE SURE THESE MATCH WITH THE QUERY

# function and variables for google sheets
def create_service() :
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scopes=SCOPES)
            #flow = InstalledAppFlow.from_client_secrets_file(
            #    'credentials.json', SCOPES)
            #creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        #with open('token.pickle', 'wb') as token:
            #pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    return service

# get the rows returned from google, need to handle stupid omissions...
def get_array_from_api_vals(vals) :
    strs = [[''] * NUMCOLS for i in range(NUMROWS)]
    str_arr = np.array(strs, dtype=object)
    for i in range(NUMROWS) :
        if not len(vals) > i :
            break
        row = vals[i]
        for j in range(NUMCOLS) :
            if not len(row) > j :
                break
            str_arr[i,j] = row[j]

    return str_arr

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
    word_length = len(word)
    length = word_length + 2 #longer fade-in
    params['length'] = length

    # modulation and carrier params

    num_frames = get_num_frames_static(length)
    mod_params = np.zeros((num_frames, 8))
    word_index = 0

    for i in range(8) :
        count = counts[letter_freq[i]]
        ratio1 = letter_dict[word[word_index]] / letter_dict[word[(word_index - 1) % word_length]] % 5
        ratio2 = letter_dict[word[(word_index + 1) % word_length]] / letter_dict[word[word_index]] % 5
        if count == 1 :
            mod_params[:, i] = ratio1
        else :
            mod_params[:, i] = (ratio1 + ratio2)/2 + (ratio1 - ratio2)/2 * sawtooth(2*np.pi*count/2*np.linspace(0, 1, num_frames), width=0.5)
        word_index = (word_index + 1) % word_length
    params['mod'] = mod_params

    car_params = np.zeros((num_frames, 8))
    for i in range(8) :
        count = counts[letter_freq[i+8]]
        ratio1 = letter_dict[word[word_index]] / letter_dict[word[(word_index + 1) % word_length]] % 5
        ratio2 = letter_dict[word[(word_index - 1) % word_length]] / letter_dict[word[word_index]] % 5
        if count == 1 :
            car_params[:, i] = ratio1
        else :
            car_params[:, i] = (ratio1 + ratio2)/2 + (ratio1 - ratio2)/2 * sawtooth(2*np.pi*count/2*np.linspace(0, 1, num_frames), width=0.5)
        word_index = (word_index + 1) % word_length
    params['car'] = car_params

    # pitch param
    most_common_letter = counts.most_common()[0][0]
    #params['pitch'] = pitch_dict[most_common_letter] if not most_common_letter == 'z' else (random.random() * 24) - 12
    params['pitch'] = pitch_dict[most_common_letter] if not most_common_letter == 'z' else (random.random() * 12) - 6

    # feedback
    params['feedback'] = sum([1 if counts[letter_freq[i+16]] > 0 else 0 for i in range(10)]) / 10

    # envelope
    if word_length < 3 or not orig_word.islower():
        params['envelope'] = 'exp'
    else :
        first_letter = word[0]
        last_letter = word[-1]
        mid_letter = word[word_length//2]
        vowel_total = counts['a'] + counts['e'] + counts['i'] + counts['o'] + counts['u']
        word_total = sum(counts.values())
        s_level = vowel_total / word_total
        a_level = 1.0 - s_level
        a_time = abs(letter_dict[mid_letter] - letter_dict[first_letter]) / 50 # / 25 * 0.5
        r_time = abs(letter_dict[last_letter] - letter_dict[mid_letter]) / 50 # / 25 * 0.5
        d_time = 1.0 - a_time-r_time

        # lengthening fade-in and fade-out
        if a_time < 0.25 :
            a_time = 0.25
        if d_time < 0.25 :
            d_time = 0.25

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
            mono = arch.generate_audio(params['length'], params['pitch'])[0]
            reverb = scipy.signal.fftconvolve(mono, reverb_sig)
            mono = (reverb / np.max(np.abs(reverb))) * np.max(np.abs(mono)) # normalize to original volume
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
        if devices[i]['name'] == 'Soundflower (2ch)' :
        #if devices[i]['name'] == 'Built-in Output' :
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

    service = create_service()
    sheet = service.spreadsheets()

    inp = input('when everything is ready, type "begin": ')
    while not inp == 'begin' :
        time.sleep(0.1)

    keep_looping = True
    last_vals = np.array([[''] * NUMCOLS for i in range(NUMROWS)], dtype=object)

    sheetNo = 1

    while keep_looping : # find a different way here
        result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range='Sheet'+ str(sheetNo) + '!' + SAMPLE_RANGE_NAME,
                                majorDimension='ROWS').execute()
        values = result.get('values', [])
        if values is not None :
            str_arr =  get_array_from_api_vals(values)
            rows_to_check = str_arr != last_vals
            for phrase in str_arr[rows_to_check].flatten() :
                words = phrase.split()
                for w in words :
                    word = ''.join(c for c in w if c.isalpha())
                    if word == 'theend' :
                        keep_looping = False
                        break
                    elif len(word) > 0 :
                        queues[q_index].put(word) 
                        q_index = (q_index + 1) % 5
            last_vals = str_arr
        timeout = 5
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            s = sys.stdin.readline()
            try :
                sheetNo = int(s)
                if sheetNo < 1 or sheetNo > 4 :
                    sheetNo = 1
            except :
                print('weird input')

    #queue.put(None)
    for i in [0,1,2,3,4] :
        queues[i].put(None)
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
