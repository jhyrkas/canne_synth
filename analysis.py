import synth_architecture as sa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

synth = sa.Architecture('root', np.zeros(8) + 1.0)
fs = 44100
path1 = 'analysis_files/orig_param_sweep/'
path2 = 'analysis_files/predict_and_sweep/'
path3 = 'analysis_files/feedback/'

if not os.path.exists(path1) :
    os.makedirs(path1)

if not os.path.exists(path2) :
    os.makedirs(path2)

if not os.path.exists(path3) :
    os.makedirs(path3)

n_secs = 10.0
n_frames = synth.get_num_frames(n_secs)

# original file for feedback
params = np.zeros((n_frames, 8)) + 1.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path3+'root_audio.wav', audio, fs)

# generate parameter sweeps

for i in range (8) :
    params = np.zeros((n_frames, 8))
    params[:, i ] = np.linspace(0.0, 5.0, n_frames)
    synth.update_params('root', params)
    audio = synth.generate_audio(n_secs)[0]
    sf.write(path1+'param'+str(i)+'_zero_start.wav', audio, fs)

for i in range (8) :
    params = np.zeros((n_frames, 8)) + 2.0
    params[:, i ] = np.linspace(0.0, 5.0, n_frames)
    synth.update_params('root', params)
    audio = synth.generate_audio(n_secs)[0]
    sf.write(path1+'param'+str(i)+'_two_start.wav', audio, fs)

#spectral effect of prediction
params = np.zeros(8) + 1.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_1_original.wav', audio, fs)
params = np.zeros(8) + 2.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_2_original.wav', audio, fs)
params = np.zeros(8) + 3.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_3_original.wav', audio, fs)
params = np.zeros(8) + 4.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_4_original.wav', audio, fs)

synth.add_network('pred', 'root')

params = np.zeros(8) + 1.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_1_predicted.wav', audio, fs)
params = np.zeros(8) + 2.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_2_predicted.wav', audio, fs)
params = np.zeros(8) + 3.0
synth.update_params('root', params)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_3_predicted.wav', audio, fs)
params = np.zeros(8) + 4.0
synth.update_params('root', params)
params2 = np.zeros((n_frames, 8)) + 2.0
params2[:, 3 ] = np.linspace(0.0, 5.0, n_frames)
synth.update_params('pred', params2)
audio = synth.generate_audio(n_secs)[0]
sf.write(path2+'constant_param_4_predicted_sweep.wav', audio, fs)

# feedback effect
synth.update_params('root', np.zeros(8) + 3.0)
synth.update_params('pred', None)
feedback_amounts = [0, 25, 50, 75, 90]
for fa in feedback_amounts :
    synth.update_feedback('pred', fa / 100.0)
    audio = synth.generate_audio(n_secs)[0]
    sf.write(path3+'feedback_rate_' + str(fa) + '.wav', audio, fs)
    # clearing the cached last frame
    synth.update_feedback('pred', 0.0)
    audio = synth.generate_audio(1)[0]

# predictive feedback
synth.add_network('pred2', 'root', predictive_feedback_mode = True)
audio = synth.generate_audio(n_secs)[1]
sf.write(path3+'predictive_feedback.wav', audio, fs)
audio = synth.generate_audio(n_secs*3)[1]
sf.write(path3+'predictive_feedback_30sec.wav', audio, fs)

# analysis

# spectrograms and 1st order difference

for i in range (8) :
    y, sr = librosa.load(path1+'param'+str(i)+'_zero_start.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Parameter ' + str(i) + ' (other params = 0)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path1+'param'+str(i)+'_zero_start_stft.pdf')
    plt.clf()

    y, sr = librosa.load(path1+'param'+str(i)+'_two_start.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Parameter ' + str(i) + ' (other params = 2)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path1+'param'+str(i)+'_two_start_stft.pdf')
    plt.clf()

    # first order diff
    #diff = np.diff(db, n = 1, axis = 1)
    #librosa.display.specshow(diff, x_axis='time', y_axis='log', sr=sr, cmap='twilight')
    #plt.title('1st-Order Diff Parameter ' + str(i) + ' (other params = 2)')
    #plt.colorbar(format='%+2.0f dB')
    #plt.tight_layout()
    #plt.savefig(path1+'param'+str(i)+'_two_start_1od.pdf')
    #plt.clf()

# spectrograms for predicted audio

for i in range (1, 5) :
    y, sr = librosa.load(path2+'constant_param_'+str(i)+'_original.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Original, Parameters = ' + str(i))
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path2+'constant_param_'+str(i)+'_original_stft.pdf')
    plt.clf()

    name = path2+'constant_param_'+str(i)+'_predicted'
    if i == 4 :
        name += '_sweep'
    y, sr = librosa.load(name+'.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Predicted, Parameter ' + str(i))
    if i == 4 :
        plt.title('STFT Predicted, Parameters = ' + str(i) + ' with parameter sweep')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(name + '_stft.pdf')
    plt.clf()

for fa in feedback_amounts :
    y, sr = librosa.load(path3+'feedback_rate_' + str(fa) + '.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Feedback Rate = ' + str(fa) + '%')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path3+'feedback_rate_' + str(fa) + '_stft.pdf')
    plt.clf()

y, sr = librosa.load(path3+'root_audio.wav', sr=None)
S = librosa.stft(y)
db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
plt.title('Root network')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig(path3+'root_audio_stft.pdf')
plt.clf()
y, sr = librosa.load(path3+'predictive_feedback.wav', sr=None)
S = librosa.stft(y)
db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
plt.title('Predictive feedback')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig(path3+'predictive_feedback_stft.pdf')
plt.clf()
y, sr = librosa.load(path3+'predictive_feedback_30sec.wav', sr=None)
S = librosa.stft(y)
db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
plt.title('Predictive feedback')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig(path3+'predictive_feedback_30sec_stft.pdf')
plt.clf()
