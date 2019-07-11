import tensorflow as tf

from distributed.sync import barrier

batch_size = 1024 * 1024

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

q = tf.FIFOQueue(1024, tf.float32, [batch_size], shared_name='fifo_queue')

start_barrier = barrier('start', 6)
complete_barrier = barrier('end', 6)
