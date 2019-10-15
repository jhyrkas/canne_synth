from canne import *
import numpy as np

mode = OperationMode(train=False,new_init=False,control=True,bias=True, zero_out_biases=True)
synth = ANNeSynth(mode)
synth.load_weights_into_memory()

weights = np.zeros(8) + 1.0
dg=tf.get_default_graph()
o=dg.get_operation_by_name('b_7')
#with synth._sess as sess:
#    print(o.values()[0].eval())

synth.reassign_middle_weights(weights)
#o=dg.get_operation_by_name('b_7')
#with synth._sess as sess:
#    print(o.values()[0].eval())
