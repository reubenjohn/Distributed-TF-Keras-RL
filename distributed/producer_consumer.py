from contextlib import contextmanager
from typing import Callable, Generator, Any

from distributed.job import Job, Jobs
from util import grouper, DISCARD_REMAINDER


class SimpleProducerTarget:
	def __init__(self, session_creator,
				 task_index: int, job: Job, queue,
				 generator_fn: Callable[[Any], Generator], generator_args=None,
				 enqueue_many: int = None, fill_value=DISCARD_REMAINDER):
		self.session_creator = session_creator
		self.task_index = task_index
		self.job = job
		self.q = queue
		self.enqueue_many = enqueue_many
		self.fill_value = fill_value
		self.generator_fn = generator_fn
		self.generator_args = generator_args if generator_args is not None else tuple()
		self.generator = None

		self.placeholders = self.en_q = self.q_close = self.producer_complete_barrier = None  # see build()

	@contextmanager
	def start_session(self):
		with self.session_creator(self.job.name, self.task_index) as sess:
			self.build()
			yield sess
			sess.run(self.producer_complete_barrier)
			if self.task_index == 0:
				sess.run(self.q_close)

	def build(self):
		self.print('Build graph')
		import tensorflow as tf
		from distributed.sync import barrier

		if self.enqueue_many is None:
			revised_shapes = self.q.shapes
			en_q_op = self.q.enqueue
		else:
			revised_shapes = [tf.TensorShape([None]).concatenate(shape) for shape in self.q.shapes]
			en_q_op = self.q.enqueue_many

		if len(self.q.dtypes) > 1:
			self.placeholders = [tf.placeholder(dtype, shape) for dtype, shape in zip(self.q.dtypes, revised_shapes)]
		else:
			self.placeholders = tf.placeholder(self.q.dtypes[0], revised_shapes[0])

		self.en_q = en_q_op(self.placeholders)
		self.q_close = self.q.close()

		self.producer_complete_barrier = barrier('producer_complete', len(self.job.handles))

	def exhaust_generator(self, sess):
		self.print('Starting exhaust_generator')
		self.generator = self.generator_fn(*self.generator_args)
		if self.enqueue_many is not None:
			self.generator = (tuple(zip(*group))
							  for group in grouper(self.generator, self.enqueue_many, self.fill_value))
		(self.multi_loop if isinstance(self.placeholders, list) else self.mono_loop)(sess)
		self.print('Loop ended')

	def multi_loop(self, sess):
		for values in self.generator:
			print('.', end='', flush=True)
			sess.run(self.en_q, feed_dict={placeholder: value for placeholder, value in zip(self.placeholders, values)})

	def mono_loop(self, sess):
		for value in self.generator:
			sess.run(self.en_q, feed_dict={self.placeholders: value})

	def print(self, message):
		print('Producer {}: {}'.format(self.task_index, message))


def start_local_cluster_multiprocessing(jobs: Jobs, *local_job_names: str):
	[handle.start() for job_name in local_job_names for handle in jobs[job_name].handles]
	for job_name in local_job_names:
		for index, handle in enumerate(jobs[job_name].handles):
			handle.join()
			print('Joined with process: job_name=%s, index=%d' % (job_name, index))
