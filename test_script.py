import numpy as np
import dream_lstm

sig = np.sin(2*np.pi*np.array(range(88200))/44100)
temp = dream_lstm.TimeDomainLSTM(sig, 1024, 0.1, 0.05)
temp.train_model(5)
