import compose_utility
import numpy as np

class ModeError(Exception):
    def __init__(self, mode) :
        if mode == 0 :
            self.message = 'modular network cannot use method generate_audio_car'
        else :
            self.message = 'carrier network cannot use method generate_audio_mod'

class OscillatorNetwork :
    def __init__(self, name, mode, comp_utility, params, feedback = 0.0) :
        if not (mode == 'mod' or mode == 'carr') :
            raise ValueError('invalid mode')
        self.name = name
        self.mode = 0 if mode == 'mod' else 1
        self.cu = comp_utility
        self.prev_frame = np.zeros((1, self.cu.out_size))
        if self.mode == 1 :
            self.fr = feedback
        self.outputNets = []
        self.params = params

    def generate_audio_mod(self, length) :
        if self.mode == 1 :
            raise ModeError(self.mode)
        return self.cu.make_note_osc(self.params, length)

    def generate_audio_car(self, audio) :
        if self.mode == 0 :
            raise ModeError(self.mode)
        sig,frame = self.cu.predict_audio(audio, self.params, self.fr, prev_frame = self.prev_frame)
        self.prev_frame = frame
        return sig

    def envelope_audio(self, audio) :
        if self.env is None :
            return audio
        elif self.env == 'exp' :
            return self.cu.exp_decay_env(audio)
        else :
            return self.cu.adsr_env(audio, self.env)

    def add_carrier(self, net_name) :
        self.outputNets.append(net_name)

    def get_carriers(self) :
        return self.outputNets

    def get_name(self) :
        return self.name

    def set_params(self, params) :
        self.params = params

class Architecture :
    def __init__(self, root_name, root_params, env = None) :
        if not (env is None or env == 'exp' or isinstance(env, tuple)) :
            raise ValueError('env must be None, "exp", or a tuple of ADSR pairs')
        self.name_to_net = {}
        self.root_name = root_name
        self.cu = compose_utility.ComposeUtility()
        root_net = OscillatorNetwork(root_name, 'mod', self.cu, root_params)
        self.name_to_net[root_name] = root_net
        self.env = env

    def add_network(self, net_name, mod_name, net_params = None, feedback = 0.0) :
        if net_name in self.name_to_net :
            raise ValueError('network names must be unique')
        if mod_name not in self.name_to_net :
            raise ValueError('modulating network must already exist')
        net = OscillatorNetwork(net_name, 'carr', self.cu, net_params, feedback)
        self.name_to_net[net_name] = net
        self.name_to_net[mod_name].add_carrier(net_name)

    def update_params(self, net_name, net_params) :
        self.name_to_net[net_name].set_params(net_params)

    def update_envelope(self, env) :
        if not (env is None or env == 'exp' or isinstance(env, tuple)) :
            raise ValueError('env must be None, "exp", or a tuple of ADSR pairs')
        self.env = env

    def generate_audio(self, note_length, pitch_shift = 0) :
        name = self.root_name
        audio_channels = self.generate_audio_recurse(name, note_length)
        for i in range(len(audio_channels)) :
            audio_channels[i] = self.envelope_audio(self.cu.change_pitch(audio_channels[i], pitch_shift))
        return audio_channels

    # param is either the note length or the audio input....pretty hacky
    def generate_audio_recurse(self, net_name, param) :
        net = self.name_to_net[net_name]
        audio = None
        if net_name == self.root_name :
            audio = net.generate_audio_mod(param)
        else : 
            audio = net.generate_audio_car(param)

        if len(net.get_carriers()) == 0 :
            return [audio]
        
        # theoretically there could be multiple channels
        returned_audio = []
        for n in net.get_carriers() :
            returned_audio += self.generate_audio_recurse(n, audio)
        return returned_audio

    def envelope_audio(self, audio) :
        if self.env is None :
            return audio
        elif self.env == 'exp' :
            return self.cu.exp_decay_env(audio)
        else :
            return self.cu.adsr_env(audio, self.env)

    # utility function for parameter generation
    def get_num_frames(self, seconds) :
        return self.cu.get_num_frames(seconds)
