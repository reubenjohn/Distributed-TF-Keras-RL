import datetime
import threading
from contextlib import contextmanager
from random import random
from time import sleep

import gym
import numpy as np
from timer_cm import Timer

from distributed.job import Jobs, Job
from distributed.producer_consumer import start_local_cluster_multiprocessing
from rl.agent.agent import Agent, ACTION
from rl.agent.wrappers import EpsilonDecayWrapper
from rl.gym.wrappers import LogisticObservationWrapper, IndexedActionWrapper
from rl.gym.generator import GymExperienceGenerator

n_producers = 1
gamma = 0.9


def shared_setup():
	import tensorflow as tf

	class MyAgent(Agent):
		def __init__(self, q_vals):
			self.sess = None
			self.graph = tf.get_default_graph()
			ob = tf.keras.layers.Input([ob_space.shape[0]])
			# self.ob = tf.placeholder(tf.float32, [None, ob_space.shape[0]])
			q_ob = q_vals(ob)
			action = tf.keras.layers.Lambda(lambda x: tf.argmax(x, 1))(q_ob)
			self.agent_id = random()

			self.q_action_model = tf.keras.Sequential([
				tf.keras.layers.InputLayer([ob_space.shape[0]]),
				tf.keras.layers.Dense(1)
			])

		# tf.K.Sequential([
		# 	q,
		# 	tf.K.layers.Lambda(lambda x: tf.argmax(x, 1))
		# ])(self.ob)

		def set_session(self, sess: tf.Session):
			self.sess = sess

		def act(self, ob) -> ACTION:
			tf.keras.backend.set_session(self.sess)
			print(tf.get_default_graph(), self.sess, tf.get_default_session(), tf.keras.backend.get_session())
			return self.q_action_model.predict_on_batch(np.expand_dims(ob, 0))[0]

	class Shared:
		frame_shapes = (list(ob_space.shape), list(ac_space.shape), [], list(ob_space.shape))
		frame_dtypes = tuple([tf.float32] * len(frame_shapes))

		def __init__(self):
			self.sess = None
			self.graph = tf.get_default_graph()
			self.q_vals = q_vals = tf.keras.layers.Dense(ac_space.n, activation=tf.keras.layers.LeakyReLU())

			self.producer_datasets = []
			self.iters = []

			self.agent = EpsilonDecayWrapper(MyAgent(q_vals), ac_space, epsilon_decay=0.99999)

			for i in range(n_producers):
				env = MyEnv()

				with tf.device('/job:explorer/task:' + str(i)):
					generator = GymExperienceGenerator(env, self.agent, 1024 * 512).generator
					ds = tf.data.Dataset.from_generator(
						generator, self.frame_dtypes, self.frame_shapes
					).prefetch(512).batch(128)
					self.producer_datasets.append(ds)
					it = ds.make_initializable_iterator('shared_iterator_' + str(i))
					self.iters.append(it)
			self.transitions = [tf.concat(batch, 0) for batch in zip(*(it.get_next() for it in self.iters))]

		@contextmanager
		def context(self):
			with self.agent.agent.graph.as_default():
				with self.agent.agent.sess.as_default():
					yield None

		def set_session(self, sess: tf.Session):
			print(tf.get_default_graph())
			print(self.agent.act(np.zeros([6])))
			self.sess = sess
			self.agent.agent.sess = sess

	return Shared()


def explorer_target(task_index, job, _):
	from distributed.session import SimpleDistributedSession

	setup = shared_setup()
	with SimpleDistributedSession(jobs)(job.name, task_index) as sess:
		setup.set_session(sess)
		sess.run(setup.iters[task_index].initializer)
		sleep(1)
		print(sess.run(setup.transitions))
		sleep(10)


def evaluator_target(sess, setup, writer, epoch_complete: threading.Barrier):
	sleep(10)
	agent = setup.agent.agent
	with MyEnv() as test_env:
		while epoch_complete.n_waiting == 0:
			total_reward = 0
			done = False
			ob = test_env.reset()
			with setup.graph.as_default():
				with setup.sess.as_default():
					while not done:
						action = agent.act(ob)
						ob, rew, done, _ = test_env.step(action)
						total_reward += rew
					writer.add_summary(
						sess.run(setup.test_summaries,
								 feed_dict={setup.test_episode_reward: total_reward}))
					sleep(.1)
		epoch_complete.wait()


def learner_target(task_index, job, _):
	from distributed.session import SimpleDistributedSession
	import tensorflow as tf

	setup = shared_setup()

	s, a, r, s2 = setup.transitions
	q_s = setup.q_vals(s)
	q_s2 = setup.q_vals(s2)
	target = tf.stop_gradient(r + gamma * tf.reduce_max(q_s2, 1))
	prediction = tf.reduce_sum(q_s * tf.one_hot(tf.cast(a, tf.int32), ac_space.n), axis=1)
	td_error = target - prediction
	loss_op = tf.reduce_mean(abs(td_error))

	minimize = tf.train.AdamOptimizer(.01).minimize(loss_op)

	# inputs = [K.layers.Input(tensor=tensor) for tensor in setup.transitions]

	# model = K.Model(inputs=inputs, outputs=[td_error])
	# model.compile(tf.train.AdamOptimizer(), K.losses.mean_absolute_error)

	epoch_complete = threading.Barrier(2)
	with tf.variable_scope('test_performance'):
		setup.test_episode_reward = tf.placeholder(tf.float32)
		setup.test_summaries = tf.summary.merge([
			tf.summary.scalar('test_episode_reward', setup.test_episode_reward),
			# tf.summary.histogram('agent_qs', setup.greedy_agent.greedy_agent.q_vs)
		])
	with tf.variable_scope('train_performance'):
		loss_summary = tf.summary.scalar('loss', loss_op)

	writer = tf.summary.FileWriter('./logs/rl/tf_distributed_np_ql/' + datetime.datetime.now().isoformat())

	with SimpleDistributedSession(jobs)(job.name, task_index) as sess:
		setup.set_session(sess)
		sleep(4)

		evaluator_thread = threading.Thread(target=evaluator_target, args=[sess, setup, writer, epoch_complete])

		with Timer('Learner complete'):
			for _ in range(4):
				sess.run(tf.global_variables_initializer())
				# sess.run([it.initializer for it in setup.iters])
				evaluator_thread.start()
				with Timer('\nEpoch complete'):
					try:
						while True:
							loss, _ = sess.run([loss_op, minimize])
							writer.add_summary(sess.run(loss_summary))
							writer.flush()

					except tf.errors.OutOfRangeError:
						epoch_complete.wait(1000)
						evaluator_thread.join(1000)


jobs = Jobs(
	Job('learner', learner_target, ['localhost:2231'])
)
jobs.add(Job('explorer', explorer_target, [
	'localhost:2221',
	# 'localhost:2222',
	# 'localhost:2223',
	# 'localhost:2224',
	# 'localhost:2225',
	# 'localhost:2226',
	# 'localhost:2227',
	# 'localhost:2228',
]))

MyEnv = lambda: IndexedActionWrapper(LogisticObservationWrapper(gym.make('Copy-v0')))
if __name__ == '__main__':
	with MyEnv() as ref_env:
		ob_space = ref_env.observation_space
		ac_space = ref_env.action_space
		start_local_cluster_multiprocessing(jobs, 'explorer', 'learner')
