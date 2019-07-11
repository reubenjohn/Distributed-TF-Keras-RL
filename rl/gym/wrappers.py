import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple

from rl.gym.spaces import DiscreteSpaceFlattener
from util import next_n


def _test_space():
	box_shape = [2, 3]
	return Tuple([
		Discrete(4),
		Tuple([
			Discrete(3), Box(np.array([[-2, -1, -1], [0, 0, 0]]), np.ones(box_shape))
		])
	])


class LogisticSpaceWrapper:
	def __init__(self, space: gym.Space, env: gym.Env = None):
		self.underlying_space = space
		ob_shape = LogisticSpaceWrapper.get_flattened_size(space)
		self.logistic_space = Box(np.zeros(ob_shape), np.ones(ob_shape), dtype=np.float32)
		super().__init__() if env is None else super().__init__(env)

	@staticmethod
	def get_flattened_size(space: gym.Space) -> int:
		"""
		>>> LogisticSpaceWrapper.get_flattened_size(Discrete(4))
		4
		>>> high = np.ones([2, 3])
		>>> LogisticSpaceWrapper.get_flattened_size(Box(high * 0, high))
		6
		>>> LogisticSpaceWrapper.get_flattened_size(_test_space())
		13
		"""
		if isinstance(space, Discrete):
			return space.n
		elif isinstance(space, Box):
			return int(np.prod(space.shape))
		elif isinstance(space, Tuple):
			return int(sum(
				LogisticSpaceWrapper.get_flattened_size(sub_space)
				for sub_space in space.spaces
			))

	def to_logistic_coordinate(self, coordinate):
		"""
		>>> test_env = gym.Env()
		>>> list(LogisticSpaceWrapper(Discrete(3)).to_logistic_coordinate(1))
		[0.0, 1.0, 0.0]
		>>> list(LogisticSpaceWrapper(Box(np.array([-2, 0]), np.array([0, 1]))).to_logistic_coordinate([0, 0]))
		[1.0, -1.0]
		>>> list(LogisticSpaceWrapper(_test_space()).to_logistic_coordinate((3, (1, [[-2, -1, 0], [1, 0.5, 0.25]]))))
		[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5]
		"""
		return self.recurse_to_logistic_coordinate(coordinate, self.underlying_space)

	def recurse_to_logistic_coordinate(self, coordinate, space: gym.Space):
		if isinstance(space, Discrete):
			return np.eye(space.n)[coordinate]
		elif isinstance(space, Box):
			return np.reshape((coordinate - space.low) * 2 / (space.high - space.low), -1) - 1
		elif isinstance(space, Tuple):
			return np.concatenate(
				tuple(self.recurse_to_logistic_coordinate(ob, space) for ob, space in zip(coordinate, space.spaces)))
		else:
			raise AssertionError(
				'Unsupported coordinate logistic_space type: %s of %s' % (str(space), str(self.logistic_space))
			)

	def from_logistic_coordinate(self, coordinate):
		"""
		>>> test_env = gym.Env()
		>>> LogisticSpaceWrapper(Discrete(3)).from_logistic_coordinate([0., 1., 0.])
		1
		>>> LogisticSpaceWrapper(Box(np.array([-2, 0]), np.array([0, 1]))).from_logistic_coordinate([0, 0])
		array([-1. ,  0.5])
		>>> LogisticSpaceWrapper(_test_space()).from_logistic_coordinate([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -0.5])
		(3, (1, array([[-2.  , -1.  ,  0.  ],
		       [ 1.  ,  0.5 ,  0.25]])))
		"""
		return self.recurse_from_logistic_coordinate(iter(coordinate), self.underlying_space)

	def recurse_from_logistic_coordinate(self, coordinate_iter, space: gym.Space):
		if isinstance(space, Discrete):
			return int(np.argmax(next_n(coordinate_iter, space.n)))
		elif isinstance(space, Box):
			end_index = int(np.prod(space.shape))
			return space.low + ((np.reshape(next_n(coordinate_iter, end_index), space.shape) + 1) / 2) * (
					space.high - space.low)
		elif isinstance(space, Tuple):
			return tuple(self.recurse_from_logistic_coordinate(coordinate_iter, space) for space in space.spaces)
		else:
			raise AssertionError(
				'Unsupported coordinate_iter logistic_space type: %s of %s' % (str(space), str(self.logistic_space))
			)


class LogisticActionWrapper(LogisticSpaceWrapper, gym.ActionWrapper):
	def __init__(self, env):
		super().__init__(env.action_space, env)
		self.action_space = self.logistic_space

	def action(self, action):
		return self.from_logistic_coordinate(action)


class LogisticObservationWrapper(LogisticSpaceWrapper, gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env.observation_space, env)
		self.observation_space = self.logistic_space

	def observation(self, observation):
		return self.to_logistic_coordinate(observation)


class IndexedSpaceWrapper:
	def __init__(self, space: gym.Space, env: gym.Env = None):
		"""

		:param space:
		:param env:
		>>> IndexedSpaceWrapper(gym.spaces.Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])])).index_space
		Discrete(24)
		"""
		self.space_flattener = DiscreteSpaceFlattener(space)
		self.index_space = Discrete(self.space_flattener.n_discrete_coordinates)
		super().__init__() if env is None else super().__init__(env)


class IndexedActionWrapper(IndexedSpaceWrapper, gym.ActionWrapper):
	def __init__(self, env):
		super().__init__(env.action_space, env)
		self.action_space = self.index_space

	def action(self, action_index):
		"""

		:param action:
		:return:
		>>> IndexedActionWrapper(gym.make('Copy-v0')).action(16)
		[1, 1, 1]
		"""
		return self.space_flattener.from_flat_index(action_index)
