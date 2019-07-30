import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM,  Dense

class TimeDomainLSTM :
    def __init__(self, input_audio, segment_length, input_dropout = 0.0, recurrent_dropout = 0.0) :
        self.seed_size = segment_length
        self.create_training(input_audio, segment_length)
        self.create_model(segment_length, input_dropout, recurrent_dropout)
        
    def create_training(self, input_audio, segment_length) :
        self.x = np.zeros((len(input_audio) - segment_length + 1, segment_length))
        self.y = np.zeros((len(input_audio) - segment_length + 1, 1))

        for i in range(self.x.shape[0]) :
            self.x[i,:] = input_audio[i:i+segment_length]
            if i != self.x.shape[0] - 1 :
                self.y[i,0] = input_audio[i+segment_length]
            else :
                self.y[i,0] = input_audio[0]

    # TODO: see if this causes any bugs
    def update_training(self, input_audio) :
        self.create_training(input_audio, self.seed_size)

    def create_model(self, segment_length, input_dropout, recurrent_dropout) :
        self.model = Sequential()
        # trying with two LSTMs
        self.model.add(
            LSTM(segment_length, dropout = input_dropout, return_sequences = True, \
                 recurrent_dropout = recurrent_dropout, input_shape=(None,self.x.shape[1]))
            )
        self.model.add(
            LSTM(segment_length, dropout = input_dropout, recurrent_dropout = recurrent_dropout)
            )
        self.model.add(Dense(1)) # output shape of 1
        # dream uses squared error, but not sure when range is [-1 1]
        #self.model.compile(optimizer='adam', loss='mean_absolute_error')
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, num_epochs) :
        data = self.x.reshape(self.x.shape[0], 1, self.x.shape[1])
        self.model.fit(data, self.y, epochs=num_epochs)

    def dream(self, seed, num_samples) :
        seed_dim1 = seed.shape[0]
        seed_dim2 = seed.shape[1]
        seed = seed.reshape(seed_dim1, 1, seed_dim2)
        samples = np.zeros(num_samples)
        for i in range(num_samples) :
            pred = self.model.predict(seed)
            samples[i] = pred[-1,0]
            tmp = seed.flatten()
            tmp = np.array(tmp.tolist()[1:] + [pred[-1,0]])
            seed = tmp.reshape(seed_dim1, 1, seed_dim2)

        return samples
