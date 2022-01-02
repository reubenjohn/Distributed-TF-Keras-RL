import itertools
from queue import Queue
from threading import Thread
from time import sleep
from typing import Callable

import gym

from rl.agent.agent import Agent


class ExperienceGenerator:
	def __call__(self):
		raise NotImplementedError


class GymExperienceGenerator(ExperienceGenerator):
	def __init__(self, env: gym.Env, agent: Agent, n_loops: int):
		self.env = env
		self.agent = agent
		self.n_loops = n_loops

	def __call__(self):
		with self.env:
			ob = self.env.reset()

			for _ in range(self.n_loops):
				a = self.agent.act(ob)
				ob2, r, done, _ = self.env.step(a)
				yield ob, a, r, ob2
				ob = ob2 if not done else self.env.reset()


class GymExperienceSequenceGenerator(ExperienceGenerator):
	def __init__(self, env: gym.Env, agent: Agent, n_loops: int, seq_len: int):
		self.env = env
		self.agent = agent
		self.n_loops = n_loops
		self.seq_len = seq_len
		self.seq = []

	def __call__(self):
		with self.env:
			ob = self.env.reset()

			for _ in range(self.n_loops):
				a = self.agent.act(ob)
				ob2, r, done, _ = self.env.step(a)
				yield ob, a, r, ob2
				ob = ob2 if not done else self.env.reset()


class ExperienceBuffer(ExperienceGenerator):
	def __init__(self, xp_generator: Callable, buffer_size: int = 1000000):
		self.xp_generator = xp_generator
		self.buffer = []
		self.buffer_size = buffer_size
		self.insertion_index = 0

	def fill_buffer(self):
		gen = self.xp_generator()
		try:
			for self.insertion_index in range(self.buffer_size):
				self.buffer.append(next(gen))
			while True:
				for self.insertion_index in range(self.buffer_size):
					self.buffer[self.insertion_index] = next(gen)
		except StopIteration:
			print('Reached end of experience generator')

	def __call__(self):
		"""

		:return:
		>>> q = Queue()
		>>> def generator():
		...		for _ in range(6):
		...			yield q.get(True, 1)
		>>> g = ExperienceBuffer(generator, 4)()
		>>> q.put(0);q.put(1);sleep(.01)
		>>> list(itertools.islice(g, 2))
		[1, 0]
		>>> q.put(2);q.put(3, True);sleep(.01)
		>>> list(itertools.islice(g, 4))
		[3, 2, 1, 0]
		>>> q.put(4);q.put(5, True);sleep(.01)
		Reached end of experience generator
		>>> list(itertools.islice(g, 4))
		[5, 4, 3, 2]
		>>> list(itertools.islice(g, 4))
		[5, 4, 3, 2]
		"""
		Thread(target=self.fill_buffer).start()
		while True:
			print('Yielding %d experience buffer elements' % len(self.buffer))
			for frame in reversed(self.buffer[self.insertion_index:] + self.buffer[:self.insertion_index]):
				yield frame
