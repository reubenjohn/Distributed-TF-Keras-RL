from time import sleep

import numpy
from tensorflow import keras


class AgentPerformanceCallback(keras.callbacks.Callback):
	def __init__(self, agent, env, sess):
		super().__init__()
		self.sess = sess
		self.agent = agent
		self.test_env = env()

	def on_train_begin(self, logs=None):
		self.test_env.__enter__()

	def on_train_end(self, logs=None):
		self.test_env.__exit__()

	def set_session(self, sess):
		self.sess = sess

	def on_batch_end(self, batch, logs=None):
		if batch % 10 == 0:
			total_reward = 0
			done = False
			ob = self.test_env.reset()
			while not done:
				action = self.agent.act(ob)
				ob, rew, done, _ = self.test_env.step(action)
				total_reward += rew
			logs['episode_return'] = numpy.array(total_reward)
			self.render_env_trajectory()
		return super().on_batch_end(batch, logs)

	def render_env_trajectory(self):
		done = False
		ob = self.test_env.reset()
		self.test_env.render()
		while not done:
			action = self.agent.act(ob)
			ob, _, done, _ = self.test_env.step(action)
			self.test_env.render()
		sleep(.5)

	def on_epoch_begin(self, epoch, logs=None):
		pass
		# Thread(target=self.render_env_trajectory, daemon=False).start()


class ScalarTensorLogger(keras.callbacks.BaseLogger):
	def __init__(self, sess=None, **scalar_tensors):
		super().__init__()
		self.sess = sess
		self.scalars = scalar_tensors

	def set_session(self, sess):
		self.sess = sess

	def on_batch_end(self, batch, logs=None):
		keys = self.scalars.keys()
		scalar_vals = self.sess.run([self.scalars[key] for key in keys])
		for name, val in zip(keys, scalar_vals):
			logs[name] = val
		return super().on_batch_end(batch, logs)
