import typing

import gym
import numpy as np
from gym.spaces import Tuple, Discrete, Box


class DiscreteSpaceFlattener:
	def __init__(self, space: gym.Space):
		"""

		:param space:

		>>> DiscreteSpaceFlattener(Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])])).n_discrete_coordinates
		24
		"""
		self.is_deeply_discrete(space)
		self.space = space

		self.discrete_dimension_shape = DiscreteSpaceFlattener.get_discrete_dimension_size(space)
		self.flattened_discrete_dimension_shape = DiscreteSpaceFlattener.flatten(self.discrete_dimension_shape)
		self.flattened_position_power = DiscreteSpaceFlattener.get_position_power(
			self.flattened_discrete_dimension_shape)
		self.n_discrete_coordinates = np.prod(self.flattened_discrete_dimension_shape)

		self.from_flat_index = self.build_flat_index_converter()

	@staticmethod
	def is_deeply_discrete(space: gym.Space):
		"""

		:param space:
		:return:

		>>> DiscreteSpaceFlattener.is_deeply_discrete(gym.spaces.Discrete(4))
		True
		>>> DiscreteSpaceFlattener.is_deeply_discrete(gym.spaces.Box(0, 1, tuple(), np.float))
		False
		>>> DiscreteSpaceFlattener.is_deeply_discrete(Tuple([Discrete(4), Discrete(2)]))
		True
		>>> DiscreteSpaceFlattener.is_deeply_discrete(Tuple([Discrete(4), Box(0, 1, tuple(), np.float)]))
		False
		"""
		return isinstance(space, Discrete) or \
			   isinstance(space, Tuple) and all(
			DiscreteSpaceFlattener.is_deeply_discrete(inner_space) for inner_space in space.spaces
		)

	@staticmethod
	def get_discrete_dimension_size(space: gym.Space):
		"""

		:param space:
		:return:

		>>> DiscreteSpaceFlattener.get_discrete_dimension_size(Discrete(4))
		4
		>>> DiscreteSpaceFlattener.get_discrete_dimension_size(Box(0, 1, tuple(), np.float))
		Traceback (most recent call last):
		...
		AssertionError: Ensure that the space is deeply discrete. See DiscreteSpaceFlattener.is_deeply_discrete.
		>>> DiscreteSpaceFlattener.get_discrete_dimension_size(Tuple([Discrete(4), Discrete(2)]))
		(4, 2)
		>>> DiscreteSpaceFlattener.get_discrete_dimension_size(Tuple([Discrete(4), Box(0, 1, tuple(), np.float)]))
		Traceback (most recent call last):
		...
		AssertionError: Ensure that the space is deeply discrete. See DiscreteSpaceFlattener.is_deeply_discrete.
		"""
		if isinstance(space, Discrete):
			return space.n
		elif isinstance(space, Tuple):
			return tuple(DiscreteSpaceFlattener.get_discrete_dimension_size(space) for space in space.spaces)
		else:
			assert DiscreteSpaceFlattener.is_deeply_discrete(space), \
				'Ensure that the space is deeply discrete. See DiscreteSpaceFlattener.is_deeply_discrete.'

	@staticmethod
	def flatten(shape) -> typing.List[int]:
		"""

		:param shape:
		:return:

		>>> DiscreteSpaceFlattener.flatten(4)
		[4]
		>>> DiscreteSpaceFlattener.flatten([4, [[1, 2], [3]]])
		[4, 1, 2, 3]
		"""
		if isinstance(shape, int):
			return [shape]
		elif isinstance(shape, (tuple, list)):
			flattened_shape = []
			for sub_shape in shape:
				sub_shape = DiscreteSpaceFlattener.flatten(sub_shape)
				flattened_shape += sub_shape
			return flattened_shape
		else:
			raise AssertionError('shape can only be an integer or a list/tuple of integers or list/tuple and so on')

	@staticmethod
	def get_position_power(flattened_shape: typing.List[int]) -> typing.List[int]:
		"""
		:param flattened_shape:
		:return:

		>>> DiscreteSpaceFlattener.get_position_power([4, 1, 2, 3])
		[6, 6, 3, 1]
		"""
		return list(reversed(np.cumprod(list(reversed(flattened_shape[1:]))))) + [1] if len(flattened_shape) > 1 \
			else [1]

	def to_flat_index(self, action) -> int:
		"""

			:param space:
			:return:

			>>> f = DiscreteSpaceFlattener(Discrete(4))
			>>> f.to_flat_index(3)
			3
			>>> f = DiscreteSpaceFlattener(Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])]))
			>>> f.to_flat_index(3)
			Traceback (most recent call last):
			...
			AssertionError: Ensure that you pass a value that belongs to the expected space: Tuple(Discrete(4), Tuple(Discrete(3), Discrete(2)))
			>>> f = DiscreteSpaceFlattener(Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])]))
			>>> f.to_flat_index([3,2,1])
			23
			"""
		flattened_action = DiscreteSpaceFlattener.flatten(action)
		assert len(self.flattened_position_power) == len(flattened_action), \
			'Ensure that you pass a value that belongs to the expected space: ' + str(self.space)
		return int(np.sum(np.multiply(self.flattened_position_power, flattened_action)))

	def build_flat_index_converter(self) -> typing.Callable[[int], typing.Union[int, list]]:
		"""

		:return:

		>>> f = DiscreteSpaceFlattener(Discrete(4))
		>>> f.from_flat_index(3)
		3
		>>> f = DiscreteSpaceFlattener(Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])]))
		>>> f.from_flat_index(23)
		[3, [2, 1]]
		"""

		def _from_flat_index(index: int, discrete_dimension_shape, flattened_shape_iter: typing.Iterator[int]):
			action = []
			for dim_shape_part in discrete_dimension_shape:
				if isinstance(dim_shape_part, int):
					dim_size = next(flattened_shape_iter)
					action_part = int(index / dim_size)
					index -= action_part * dim_size
					action.append(action_part)
				elif isinstance(dim_shape_part, tuple):
					action_part, index = _from_flat_index(index, dim_shape_part, flattened_shape_iter)
					action.append(action_part)

			return action, index

		if isinstance(self.discrete_dimension_shape, int):
			return lambda index: index
		else:
			return lambda index: _from_flat_index(index, self.discrete_dimension_shape,
												  iter(self.flattened_position_power))[0]

	def __str__(self) -> str:
		"""

		:return:

		>>> str(DiscreteSpaceFlattener(Tuple([Discrete(4), Tuple([Discrete(3), Discrete(2)])])))
		'Tuple(Discrete(4), Tuple(Discrete(3), Discrete(2)))'
		"""
		return str(self.space)
