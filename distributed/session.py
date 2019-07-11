from contextlib import contextmanager

import tensorflow as tf

from distributed.job import Jobs


class DistributedSession:
	def __call__(self, job_name: str, task_index: int):
		raise NotImplementedError


class SimpleDistributedSession(DistributedSession):
	def __init__(self, jobs: Jobs, config=None):
		self.jobs = jobs
		self.config = config

		from distributed.sync import barrier

		n_tasks = len(self.jobs.get_handles())
		self.start_barrier = barrier('start', n_tasks)
		self.complete_barrier = barrier('end', n_tasks)

	@contextmanager
	def __call__(self, job_name: str, task_index: int) -> tf.Session:
		if self.config is None:
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
		else:
			config = self.config

		with tf.Session(
				tf.train.Server(
					tf.train.ClusterSpec(self.jobs.get_cluster_dict()),
					job_name, task_index, config=config
				).target
		) as sess:
			print('%s %d: Waiting for peers to start session' % (job_name, task_index))
			sess.run(self.start_barrier)
			yield sess
			while True:
				try:
					print('%s %d: Waiting for peers before terminating session' % (job_name, task_index))
					sess.run(self.complete_barrier, options=tf.RunOptions(timeout_in_ms=5000))
					break
				except tf.errors.DeadlineExceededError:
					print('%s %d: Still waiting for peers before terminating session' % (job_name, task_index))
			print('%s %d: Terminating session' % (job_name, task_index))
