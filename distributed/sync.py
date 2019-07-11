import tensorflow as tf


def barrier(shared_name: str, n_workers: int):
	passing_q = tf.FIFOQueue(n_workers, tf.bool, (), shared_name=shared_name + '_count_q')
	increment_size = passing_q.enqueue(True)
	blocking_q = tf.FIFOQueue(n_workers, tf.bool, (), shared_name=shared_name + '_barrier_q')
	with tf.control_dependencies([increment_size]):
		incremented_size = passing_q.size()
		return tf.cond(tf.equal(incremented_size, n_workers),
					   lambda: tf.group(
						   [blocking_q.enqueue_many([[True] * n_workers]), passing_q.dequeue_many(n_workers)]),
					   lambda: blocking_q.dequeue()
					   )
