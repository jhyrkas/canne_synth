import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM,  Dense

class TimeDomainLSTM :
    def __init__(self, input_audio, segment_length, input_dropout = 0.0, recurrent_dropout = 0.0) :
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

        print('training size = ' + str(self.x.shape))

    def create_model(self, segment_length, input_dropout, recurrent_dropout) :
        self.model = Sequential()
        self.model.add(
            LSTM(segment_length, dropout = input_dropout, \
                 recurrent_dropout = recurrent_dropout, input_shape=(None,self.x.shape[1])))
        self.model.add(Dense(1)) # output shape of 1
        # dream uses squared error, but not sure when range is [-1 1]
        self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    def train_model(self, num_epochs) :
        data = self.x.reshape(self.x.shape[0], 1, self.x.shape[1])
        self.model.fit(data, self.y, num_epochs)
