import numpy as np
import matplotlib.pyplot as plt
import compose_utility
from matplotlib.animation import FuncAnimation
#import librosa
#import librosa.display

cu = compose_utility.ComposeUtility()
length = cu.get_num_frames(5.0)
path = 'analysis_files/middle_weights/'

params = np.zeros((length, 8)) + 1.0
params[:, 3] = np.linspace(0.0, 3.0, length)

middle_weights = np.array(cu.predict_and_get_middle_weights(cu.make_note_osc(params, 5.0)))
middle_weights = middle_weights.reshape(middle_weights.shape[0], middle_weights.shape[2])

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
fig.set_tight_layout(True)
ax1.set_ylim(0, 4.0)
ax2.set_ylim(0, 4.0)
line1, = ax1.plot(params[0,:], 'b.')
line2, = ax2.plot(middle_weights[0,:], 'b.')

def update(i) :
    line1.set_ydata(params[i,:])
    line2.set_ydata(middle_weights[i,:])
    return line1, line2

anim = FuncAnimation(fig, update, frames = np.arange(0, length), interval = 50)
anim.save(path+'params.gif', dpi=8)
plt.clf()

params = np.zeros((length, 8)) + 1.0
params[:, 7] = np.linspace(0.0, 3.0, length)

middle_weights = np.array(cu.predict_and_get_middle_weights(cu.make_note_osc(params, 5.0)))
middle_weights = middle_weights.reshape(middle_weights.shape[0], middle_weights.shape[2])

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
fig.set_tight_layout(True)
ax1.set_ylim(0, 4.0)
ax2.set_ylim(0, 4.0)
line1, = ax1.plot(params[0,:], 'b.')
line2, = ax2.plot(middle_weights[0,:], 'b.')

def update2(i) :
    line1.set_ydata(params[i,:])
    line2.set_ydata(middle_weights[i,:])
    return line1, line2

anim = FuncAnimation(fig, update2, frames = np.arange(0, length), interval = 50)
anim.save(path+'params2.gif', dpi=8)
plt.clf()

params = np.zeros((length, 8)) + 1.0
params[:, 3] = np.linspace(0.0, 3.0, length)
params[:, 5] = np.linspace(3.0, 0.0, length)

middle_weights = np.array(cu.predict_and_get_middle_weights(cu.make_note_osc(params, 5.0)))
middle_weights = middle_weights.reshape(middle_weights.shape[0], middle_weights.shape[2])

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
fig.set_tight_layout(True)
ax1.set_ylim(0, 4.0)
ax2.set_ylim(0, 4.0)
line1, = ax1.plot(params[0,:], 'b.')
line2, = ax2.plot(middle_weights[0,:], 'b.')

def update3(i) :
    line1.set_ydata(params[i,:])
    line2.set_ydata(middle_weights[i,:])
    return line1, line2

anim = FuncAnimation(fig, update3, frames = np.arange(0, length), interval = 50)
anim.save(path+'params3.gif', dpi=8)
plt.clf()



