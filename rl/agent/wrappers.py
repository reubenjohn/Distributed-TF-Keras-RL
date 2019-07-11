import math
import random

import gym
import numpy as np
import tensorflow as tf

from rl.agent.agent import Agent, ACTION


class AgentWrapper(Agent):
	def __init__(self, agent: Agent):
		self.agent = agent

	def act(self, ob) -> ACTION:
		return self.agent.act(ob)

	def save(self, path: str):
		return self.agent.save(path)

	def load(self, path: str):
		return self.agent.load(path)


class EpsilonAgentWrapper(AgentWrapper):
	def __init__(self, agent: Agent, action_space: gym.Space, epsilon: float = 1):
		self.epsilon = epsilon
		self.action_space = action_space
		super().__init__(agent)

	def act(self, ob):
		return self.action_space.sample() if random.random() < self.epsilon else super().act(ob)


class EpsilonDecayWrapper(AgentWrapper):
	def __init__(self, agent: Agent, action_space: gym.Space,
				 total_steps, epsilon_after_steps=0.001):
		self.decay_coeff = math.log((2 / epsilon_after_steps - 1)) / total_steps
		self.action_space = action_space
		self.epsilon = 1.
		self.training_step = 0
		super().__init__(agent)

	def set_training_step(self, training_step):
		self.training_step = training_step

	def act(self, ob):
		self.epsilon = 1 / (math.exp(self.training_step * self.decay_coeff))
		return self.action_space.sample() if random.random() < self.epsilon else super().act(ob)


class EpsilonDecayLayer(tf.keras.layers.Lambda):
	def __init__(self, action_space, training_step, total_steps, epsilon_after_steps=.001, true_fn=lambda x: x,
				 **kwargs):
		self.training_step = training_step
		self.epsilon = None
		self.decay_coeff = math.log((2 / epsilon_after_steps - 1)) / total_steps
		self.action_space = action_space
		self.true_fn = true_fn
		super().__init__(self.function, **kwargs)

	def function(self, ob):
		self.epsilon = tf.divide(1, tf.exp(tf.cast(self.training_step, tf.float32) * self.decay_coeff))
		return tf.cond(
			tf.random.uniform((), 0, 1) >= self.epsilon,
			lambda: self.true_fn(ob),
			lambda: tf.py_func(
				lambda _: np.expand_dims(np.asarray(self.action_space.sample(), np.int32), 0), [ob], tf.int32)
		)
