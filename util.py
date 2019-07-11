from itertools import repeat, zip_longest
from typing import Iterator, Iterable

import tensorflow as tf

K = tf.keras

DISCARD_REMAINDER = 'DISCARD_REMAINDER'


def next_n(it: Iterator, n: int):
	return list(map(next, repeat(it, n)))


def longest_grouper(iterable: Iterable, group_size: int, fillvalue=None):
	"""
	Collect data into fixed-length chunks or blocks, filling `fillvalue` for when the shorter iterables stop
	:param iterable:
	:param group_size:
	:param fillvalue:
	:return:
	>>> list(longest_grouper('ABCDEFG', 3, 'x'))
	[('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
	"""
	#
	args = [iter(iterable)] * group_size
	return zip_longest(*args, fillvalue=fillvalue)


def shortest_grouper(iterable: Iterable, group_size: int):
	"""
	Collect data into fixed-length chunks or blocks, stopping with the shortest iterable
	:param iterable:
	:param group_size:
	:return:
	>>> list(shortest_grouper('ABCDEFG', 3))
	[('A', 'B', 'C'), ('D', 'E', 'F')]
	"""
	return zip(*[iter(iterable)] * group_size)


def grouper(iterable: Iterable, group_size: int, fill_value=DISCARD_REMAINDER):
	if fill_value == DISCARD_REMAINDER:
		return shortest_grouper(iterable, group_size)
	else:
		return longest_grouper(iterable, group_size, fill_value)


def with_device(device, op):
	with device:
		return op()


to_keras_lambda = lambda fn, name, *args: K.layers.Lambda(lambda args: fn(*args), name=name)(args)

index_by = lambda tensor, indices, axis=-1: tf.reduce_sum(
	tensor * tf.one_hot(tf.cast(indices, tf.int32), tf.shape(tensor)[axis]), axis)


def extend_model(model, extension):
	if isinstance(extension, list):
		extension = K.Sequential([
			K.layers.InputLayer(),
			*extension
		])
	return K.Model(inputs=model.input, outputs=extension(model.output))
