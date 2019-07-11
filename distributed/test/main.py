import numpy

from distributed.job import Job, Jobs
from distributed.producer_consumer import SimpleProducerTarget, start_local_cluster_multiprocessing
from distributed.session import SimpleDistributedSession


def producer_main(task_index: int, job: Job, _):
	from distributed.test.shared import q, batch_size
	SimpleProducerTarget(task_index, job, q, lambda: (numpy.random.random(batch_size) for _ in range(128)),
						 SimpleDistributedSession(jobs)).main()


def consumer_main(task_index: int, job: Job, _):
	def log(message):
		print('Consumer {}: {}'.format(task_index, message))

	import tensorflow as tf
	from distributed.test.shared import q

	de_q = q.dequeue()

	log('Started')
	with SimpleDistributedSession(jobs)(job.name, task_index) as sess:
		try:
			while True:
				sess.run(de_q)
		except tf.errors.OutOfRangeError:
			log('Complete')


jobs = Jobs(
	Job('producer', producer_main, [
		'localhost:2221',
		'localhost:2222',
		'localhost:2223',
		'localhost:2224',
		'localhost:2225',
		'localhost:2226',
	]),
	Job('consumer', consumer_main, [
		'localhost:2231'
	])
)

if __name__ == '__main__':
	start_local_cluster_multiprocessing(jobs, 'producer', 'consumer')
