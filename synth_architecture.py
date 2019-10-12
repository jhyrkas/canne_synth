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

    def add_carrier(self, net_name) :
        self.outputNets.append(net_name)

    def get_carriers(self) :
        return self.outputNets

    def get_name(self) :
        return self.name

    def set_params(self, params) :
        self.params = params

class Architecture :
    def __init__(self, root_name, root_params) :
        self.name_to_net = {}
        self.root_name = root_name
        self.cu = compose_utility.ComposeUtility()
        root_net = OscillatorNetwork(root_name, 'mod', self.cu, root_params)
        self.name_to_net[root_name] = root_net

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

    def generate_audio(self, note_length) :
        name = self.root_name
        return self.generate_audio_recurse(name, note_length)

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
