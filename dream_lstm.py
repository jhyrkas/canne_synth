import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM,  Dense

class TimeDomainLSTM :
    def __init__(self, input_audio, segment_length, input_dropout = 0.0, recurrent_dropout = 0.0) :
        self.seed_size = segment_length
        self.batch_size = 100 # parameterize?
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
                 recurrent_dropout = recurrent_dropout, input_shape=(None,self.x.shape[1]),
                 stateful = True, batch_input_shape = (self.batch_size, 1, self.x.shape[1]))
            )
        self.model.add(Dense(1)) # output shape of 1
        # dream uses squared error, but not sure when range is [-1 1]
        self.model.compile(optimizer='adam', loss='mean_absolute_error')

    def train_model(self, num_epochs) :
        data = self.x.reshape(self.x.shape[0], 1, self.x.shape[1])
        # bit of a hack: truncate for stateful execution
        mod = data.shape[0] % self.batch_size
        for i in range(num_epochs) :
            print('epoch #' + str(i+1))
            self.model.fit(data[0:data.shape[0] - mod, :, :], self.y[0:data.shape[0] - mod,:], 
                    epochs=1, batch_size=self.batch_size, shuffle=False)
            self.reset_model()

    def dream(self, num_samples) :
        self.reset_model()
        seed = self.x[0:self.batch_size, :]
        seed = seed.reshape(self.batch_size, 1, self.seed_size)
        samples = np.zeros(num_samples)
        for i in range(num_samples) :
            pred = self.model.predict(seed)
            #if i == 0 :
            #    print(pred.shape)
            #    print(pred)
            samples[i] = pred[-1,0]
            tmp = seed.flatten()
            tmp = np.array(tmp.tolist()[1:] + [pred[-1,0]])
            seed = tmp.reshape(self.batch_size, 1, self.seed_size)

        self.reset_model()
        return samples

    # use for stateful LSTMS, either before training or before dreaming
    def reset_model(self) :
        self.model.reset_states()
