import sys
from canne import *
import os
import numpy as np
import soundfile as sf
import librosa
from librosa.effects import pitch_shift

class ComposeUtility :

    def __init__(self):
        mode = OperationMode(train=False,new_init=False,control=True,bias=True, zero_out_biases=True)
        self.synth = ANNeSynth(mode)
        self.synth.load_weights_into_memory()
        self.out_size = 2049
        self.len_window = 4096
        self.hop_length = 1024
        self.fs = 44100
        self.prev_frame = np.zeros((1, self.out_size))

    # generate note from middle layer
    def make_note_osc(self, params, length) :
        num_frames = self.get_num_frames(length)
        mag_buf = np.zeros((self.out_size, num_frames))
        if params.shape == (8,) or params.shape == (1, 8) :
            params = params.reshape(1, 8)
            for i in range(num_frames) :
                mag_buf[:,i] = self.synth.generate_audio(params, bass_boost = True)
        else :
            # shape must be (n, 8)
            assert(params.shape[1] == 8)
            for i in range(num_frames) :
                j = i
                if j >= params.shape[0] :
                    j = params.shape[0] - 1
                act = params[j,:].reshape(1, 8)
                mag_buf[:,i] = self.synth.generate_audio(act, bass_boost = True)

        sig = do_rtpghi_gaussian_window(mag_buf, self.len_window, self.hop_length)
        return sig

    # generate note by full prediction, including feedback
    # prev_frame will be for successive calls to predict audio, perhaps...
    def predict_audio(self, audio, params = None, feedback_rate = 0, use_prev_frame = False) :
        input_frames = self.get_input_frames(audio)
        num_frames = input_frames.shape[1]
        mag_buf = np.zeros(input_frames.shape)
        feedback_frame = np.zeros((1, self.out_size))
        if use_prev_frame :
            # NOTE: copy here is probably unnecessary and the assignment at the end of this function
            # would be unnecessary. however, this is probably preferable for debugging
            # so no one has to worry about finding weird pass-by-reference bugs
            feedback_frame = np.copy(self.prev_frame)
        if params is None :
            params = np.zeros(8)
        if params.shape == (8,) or params.shape == (1, 8) :
            params = params.reshape(8,)
            for i in range(num_frames) :
                in_frame = ((1.0 - feedback_rate) * input_frames[:,i].reshape(1, input_frames.shape[0])) + \
                    (feedback_rate * feedback_frame)
                mag_buf[:,i] = self.synth.generate_audio(
                    in_frame, bass_boost = True, full_mode = True, middle_weights = params)
                feedback_frame = np.copy(mag_buf[:,i].reshape(1, input_frames.shape[0]))
                feedback_frame /= np.max(np.abs(feedback_frame))
        else :
            assert(params.shape[1] == 8)
            for i in range(num_frames) :
                j = i
                if j >= params.shape[0] :
                    j = params.shape[0] - 1
                act = params[j,:].reshape(8,)
                in_frame = ((1.0 - feedback_rate) * input_frames[:,i].reshape(1, input_frames.shape[0])) + \
                    (feedback_rate * feedback_frame)
                mag_buf[:,i] = self.synth.generate_audio(
                    in_frame, bass_boost = True, full_mode = True, middle_weights = act)
                feedback_frame = np.copy(mag_buf[:,i].reshape(1, input_frames.shape[0]))
                feedback_frame /= np.max(np.max(feedback_frame))
        # end predictions

        self.prev_frame = feedback_frame
        sig = do_rtpghi_gaussian_window(mag_buf, self.len_window, self.hop_length)
        return sig
       
    def predict_and_get_middle_weights(self, audio, weights = None) :
        input_frames = self.get_input_frames(audio)
        num_frames = input_frames.shape[1]
        mag_buf = np.zeros(input_frames.shape)

        if weights is not None :
            self.synth.reassign_middle_weights(weights)

        for i in range(num_frames) :
            if i == 0: # maybe check some more later?
                in_frame = input_frames[:,i].reshape(1, input_frames.shape[0])
                sig = self.synth.predict_and_get_middle_weights(in_frame)
                print(sig)

        if weights is not None :
            self.synth.reassign_middle_weights(np.zeros(8))
            

    # mag frames for input audio, used for prediction
    def get_input_frames(self, audio) :
        # TODO might need to to some work here if the input size changes
        frames = librosa.core.stft(
            audio, n_fft = self.len_window, hop_length = self.hop_length)
        frames = frames[0:(self.len_window//2 + 1),:]
        mag_frames = np.abs(frames)
        mag_max = np.max(mag_frames, axis=0)
        return mag_frames / mag_max

    # utility function
    # TODO can i get this more accurate? not quite there
    def get_num_frames(self, length) :
        target_length = float(length * self.fs)
        num_frames = int(np.ceil((target_length - self.out_size) / self.hop_length))
        return num_frames

    # TODO: add dreaming effect
    def lstm_feedback(self, audio, length) :
        pass

    # shift is given in fractional half steps, per the librosa algorithm
    def change_pitch(self, audio, shift) :
        return pitch_shift(audio, self.fs, shift)

    # levels are between [0, 1] and time is in samples
    def adsr_env(self, audio, a_time, a_level, d_time, d_level, s_time, s_level, r_time) :
        env = np.zeros(a_time + d_time + s_time + r_time)
        t = 0
        env[t:t+a_time] = np.linspace(0, a_level, a_time)
        t += a_time
        env[t:t+d_time] = np.linspace(a_level, d_level, d_time)
        t += d_time
        env[t:t+s_time] = np.linspace(d_level, s_level, s_time)
        t += s_time
        env[t:t+r_time] = np.linspace(s_level, 0, r_time)
        if len(env) < len(audio) :
            env = np.append(env, np.zeros(len(audio) - len(env)))

        return audio * env[:len(audio)]

    def adsr_env_tuple(self, audio, params) :
        return self.adsr_env(audio, params[0], params[1], params[2], params[3], params[4], params[5], params[6])

    def exp_decay_env(self, audio) :
        env = np.power(.9995, np.linspace(0, 2500, len(audio)))
        return audio * env
