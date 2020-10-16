import synth_architecture as sa
import numpy as np
import time
import sounddevice as sd

arch = sa.Architecture('root', np.zeros(8) + 1.0)
out_file = open("analysis_files/timing.csv", "w")
out_file.write('Method,num_seconds,average_time\n')

# vanilla CANNe, static params

for i in range(1, 11) :
    average = 0.0
    for j in range(50) :
        arch.update_params('root', np.random.random(8) * 4.0)
        start = time.process_time()
        audio = arch.generate_audio(i)
        end = time.process_time()
        average += (end - start)

    average /= 50.0
    out_file.write('vanilla_static,' + str(i) + ',' + str(average) + '\n')


# vanilla CANNe, changing params
for i in range(1, 11) :
    average = 0.0
    num_frames = arch.get_num_frames(i)
    for j in range(50) :
        arch.update_params('root', np.random.random((num_frames,8)) * 4.0)
        start = time.process_time()
        audio = arch.generate_audio(i)
        end = time.process_time()
        average += (end - start)

    average /= 50.0
    out_file.write('vanilla_changing,' + str(i) + ',' + str(average) + '\n')

arch.add_network('carr', 'root')
arch.update_params('root', np.random.random(8) * 4.0)

# network modulation, static params
for i in range(1, 11) :
    average = 0.0
    for j in range(50) :
        arch.update_params('carr', np.random.random(8) * 4.0)
        start = time.process_time()
        audio = arch.generate_audio(i)
        end = time.process_time()
        average += (end - start)

    average /= 50.0
    out_file.write('network_mod_static,' + str(i) + ',' + str(average) + '\n')

# network modulation, changing params
for i in range(1, 11) :
    average = 0.0
    num_frames = arch.get_num_frames(i)
    for j in range(50) :
        arch.update_params('carr', np.random.random((num_frames,8)) * 4.0)
        start = time.process_time()
        audio = arch.generate_audio(i)
        end = time.process_time()
        average += (end - start)

    average /= 50.0
    out_file.write('network_mod_changing,' + str(i) + ',' + str(average) + '\n')

out_file.close()
