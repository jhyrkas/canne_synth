import compose_utility
import numpy as np
import random
cu = compose_utility.ComposeUtility()
act = np.zeros(8)
for i in range(8) :
    act[0] = random.random() * 5
sig = cu.make_note_osc(act, 5)
sig2 = cu.predict_audio(sig, None)
cu.predict_and_get_middle_weights(sig)

for i in range(10) :
    act[0] += 0.1
    sig = cu.make_note_osc(act, 5)
    cu.predict_and_get_middle_weights(sig)

# correcting for zeros
#weights = np.zeros(8)
#weights[5] = 1.0
#weights[7] = 1.0
#sig = cu.make_note_osc(act, 5)
#cu.predict_and_get_middle_weights(sig, np.zeros(8) + 0.05)
