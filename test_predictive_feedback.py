import compose_utility
import numpy as np
import sounddevice as sd
import soundfile as sf

cu = compose_utility.ComposeUtility()
params = np.zeros(8) + 1.0
audio = cu.make_note_osc(params, 5.0)
audio2 = cu.predictive_feedback(cu.make_note_osc(params, 30.0))
sd.play(audio, 44100)
sd.wait()
sd.play(audio2, 44100)
audio3 = cu.predict_audio(cu.make_note_osc(params, 30.0), feedback_rate = 0.5)[0]
sd.wait()
sd.play(audio3, 44100)

#params = np.zeros(8) + 2.0
#audio3 = cu.make_note_osc(params, 5.0)
#audio4 = cu.predictive_feedback(cu.make_note_osc(params, 30.0))
#sd.wait()
#sd.play(audio3, 44100)
#sd.wait()
#sd.play(audio4, 44100)

n_frames = cu.get_num_frames_new(30.0)
params = np.zeros((n_frames, 8))
params[:n_frames, 0] = np.linspace(0, 3.0, n_frames)
params[:n_frames, 6] = np.logspace(3.0, 0, n_frames)
params[:n_frames, 3] = 2.0 + (2.0 * np.sin(2*np.pi*2.0*np.linspace(0, 30, n_frames)))
audio4 = cu.adsr_env(cu.make_note_osc(params, 30.0), 0.05, 1.0, 0.1, 0.95, 0.2)
sd.wait()
sd.play(audio4, 44100)
audio5 = cu.predictive_feedback(audio4)
sd.wait()
sd.play(audio5, 44100)
audio6 = cu.predict_audio(audio4, feedback_rate = 0.5)[0]
sd.wait()
sd.play(audio6, 44100)
sd.wait()

sf.write('weird_example.wav', audio4, 44100)
