import synth_architecture as sa
import numpy as np
import time

arch = sa.Architecture('root', np.zeros(8) + 1.0)
arch.add_network('carr', 'root', predictive_feedback_mode=True)
out_file = open("analysis_files/timing_predictive.csv", "w")
out_file.write('Method,num_seconds,average_time\n')

# network modulation, predictive feedback
for i in range(1, 11) :
    average = 0.0
    for j in range(50) :
        arch.update_params('root', np.random.random(8) * 4.0)
        start = time.process_time()
        audio = arch.generate_audio(i)
        end = time.process_time()
        average += (end - start)

    average /= 50.0
    out_file.write('network_mod_static,' + str(i) + ',' + str(average) + '\n')

out_file.close()
