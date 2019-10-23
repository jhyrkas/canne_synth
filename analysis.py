import synth_architecture as sa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

synth = sa.Architecture('root', np.zeros(8) + 1.0)
fs = 44100
path = 'analysis_files/'

# generate parameter sweeps
n_secs = 10.0
n_frames = synth.get_num_frames(n_secs)

for i in range (8) :
    params = np.zeros((n_frames, 8))
    params[:, i ] = np.linspace(0.0, 5.0, n_frames)
    synth.update_params('root', params)
    audio = synth.generate_audio(n_secs)[0]
    sf.write(path+'param'+str(i)+'_zero_start.wav', audio, fs)

for i in range (8) :
    params = np.zeros((n_frames, 8)) + 2.0
    params[:, i ] = np.linspace(0.0, 5.0, n_frames)
    synth.update_params('root', params)
    audio = synth.generate_audio(n_secs)[0]
    sf.write(path+'param'+str(i)+'_two_start.wav', audio, fs)

# analysis

# spectrograms and 1st order difference

for i in range (8) :
    y, sr = librosa.load(path+'param'+str(i)+'_zero_start.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Parameter ' + str(i) + ' (other params = 0)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path+'param'+str(i)+'_zero_start_stft.pdf')
    plt.clf()

    diff = np.diff(db, n = 1, axis = 1)
    librosa.display.specshow(diff, x_axis='time', y_axis='log', sr=sr, cmap='twilight')
    plt.title('1st-Order Diff Parameter ' + str(i) + ' (other params = 0)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path+'param'+str(i)+'_zero_start_1od.pdf')
    plt.clf()

for i in range (8) :
    y, sr = librosa.load(path+'param'+str(i)+'_two_start.wav', sr=None)
    S = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(db, x_axis='time', y_axis='log', sr=sr)
    plt.title('STFT Parameter ' + str(i) + ' (other params = 2)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path+'param'+str(i)+'_two_start_stft.pdf')
    plt.clf()

    diff = np.diff(db, n = 1, axis = 1)
    librosa.display.specshow(diff, x_axis='time', y_axis='log', sr=sr, cmap='twilight')
    plt.title('1st-Order Diff Parameter ' + str(i) + ' (other params = 2)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path+'param'+str(i)+'_two_start_1od.pdf')
    plt.clf()
