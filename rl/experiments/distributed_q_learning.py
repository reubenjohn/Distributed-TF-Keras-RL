import datetime
from time import sleep

import gym
import numpy as np
import tensorflow as tf

import distributed as dist
from rl.agent import KerasAgent, EpsilonAgentWrapper
from rl.gym import GymExperienceGenerator, ExperienceBuffer, IndexedActionWrapper, LogisticObservationWrapper
from rl.metrics import AgentPerformanceCallback, ScalarTensorLogger
from util import with_device, to_keras_lambda, index_by, extend_model

K = tf.keras
LambdaCallback = K.callbacks.LambdaCallback

env_id = 'MountainCar-v0'
MyEnv = lambda: IndexedActionWrapper(LogisticObservationWrapper(gym.make(env_id)))
log_dir = './logs/rl/distributed_keras/%s/%s' % (env_id, datetime.datetime.now().isoformat())

n_explorers, batch_size_per_explorer = 6, 256
batch_size = n_explorers * batch_size_per_explorer
total_epochs, steps_per_epoch = 10, 100
total_steps = total_epochs * steps_per_epoch
initial_epsilon = 1
epsilon_sigma = .5
gamma = 0.9


class Shared:
	def __init__(self):
		self.training_step = tf.Variable(0, False, name='training_step')
		self.increment_step = self.training_step.assign_add(1, True, 'increment_step')
		with tf.variable_scope('epsilon'):
			step_ratio = self.training_step / total_steps
			# step_logit = tf.log(step_ratio / (1 - tf.minimum(step_ratio, 1. - 1e-9)))
			# self.epsilon = .5 - .5 * tf.math.erf((step_logit - .5) / (math.sqrt(2) * epsilon_sigma))
			self.epsilon = tf.maximum(1 - 2 * step_ratio, .2)

		with tf.variable_scope('q'):
			self.q = K.Sequential([
				K.layers.InputLayer(ob_space_shape, name='q_input'),
				K.layers.Dense(sum(ob_space_shape) ** 2, name='dense_hidden'),
				K.layers.Dense(ac_space.n, name='dense')
			])
		with tf.variable_scope('explorer'):
			with tf.variable_scope('agent'):
				sample_q_action = extend_model(self.q, [
					K.layers.Softmax(),
					K.layers.Lambda(lambda x: tf.distributions.Categorical(probs=x).sample(), name='sample_action')
				])
			self.exploring_agent = EpsilonAgentWrapper(KerasAgent(sample_q_action), ac_space)
			generator = GymExperienceGenerator(MyEnv(), self.exploring_agent, batch_size_per_explorer * steps_per_epoch)
			generator = ExperienceBuffer(generator)
		with tf.variable_scope('experience'):
			dataset = tf.data.Dataset().from_generator(
				generator, tuple([tf.float32] * 4), (ob_space_shape, ac_space.shape, [], ob_space_shape)
			).prefetch(batch_size_per_explorer * steps_per_epoch) \
				.batch(batch_size_per_explorer, True)
			self.iters = [with_device(tf.device('/job:explorer/task:%d' % i),
									  lambda: dataset.make_initializable_iterator('shared_iter_%d' % i))
						  for i in range(n_explorers)]
			self.nxt = [tf.concat(batch, 0) for batch in zip(*(it.get_next(name='next_edge') for it in self.iters))]

	def set_session(self, sess):
		tf.keras.backend.set_session(sess)
		self.exploring_agent.agent.act(np.zeros(ob_space_shape))


def explorer_target(task_index, job, _):
	setup = Shared()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with dist.SimpleDistributedSession(jobs)(job.name, task_index) as sess:
		setup.set_session(sess)
		training_step = 0
		while training_step < total_steps:
			setup.exploring_agent.epsilon, training_step = sess.run([setup.epsilon, setup.training_step])
			sleep(1.0)


def learner_target(task_index, job, _):
	setup = Shared()
	with tf.variable_scope('q_learning'):
		s, a, r, s2, = [K.Input(tensor=inp) for inp in setup.nxt]
		q_s, q_s2 = setup.q(s), setup.q(s2)

		target_fn = lambda r, q_s2: tf.stop_gradient(tf.expand_dims(r + .9 * tf.reduce_max(q_s2, 1), 1))
		prediction_fn = lambda q_s, a: tf.expand_dims(index_by(q_s, a), 1)
		target = to_keras_lambda(target_fn, 'target', r, q_s2)
		prediction = to_keras_lambda(prediction_fn, 'prediction', q_s, a)

		learning_model = K.Model(inputs=[s, a, r, s2], outputs=[prediction])
		learning_model.compile(tf.train.AdamOptimizer(), K.losses.mean_absolute_error)

	with tf.variable_scope('metrics'):
		with tf.variable_scope('greedy_agent'):
			greedy_agent_model = K.Model(
				inputs=setup.q.input,
				outputs=K.layers.Lambda(lambda x: tf.argmax(x, 1), name='arg_max_action')(setup.q.output))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with dist.SimpleDistributedSession(jobs)(job.name, task_index) as sess:
		sess.run([tf.global_variables_initializer()] + [it.initializer for it in setup.iters])
		setup.set_session(sess)

		learning_model.fit(y=[target], steps_per_epoch=steps_per_epoch, epochs=total_epochs, callbacks=[
			LambdaCallback(on_batch_begin=lambda batch, logs: sess.run(setup.increment_step)),
			AgentPerformanceCallback(KerasAgent(greedy_agent_model), MyEnv, sess),
			ScalarTensorLogger(sess, epsilon=setup.epsilon),
			K.callbacks.TensorBoard(log_dir=log_dir),
			LambdaCallback(on_epoch_end=lambda x, _: sess.run([it.initializer for it in setup.iters]))
		])


jobs = dist.Jobs(
	dist.Job('learner', learner_target, ['localhost:2231']),
	dist.Job('explorer', explorer_target, ['localhost:' + str(i) for i in range(2221, 2221 + n_explorers)])
)
if __name__ == '__main__':
	with MyEnv() as ref_env:
		ob_space, ac_space = ref_env.observation_space, ref_env.action_space
		ob_space_shape = [ob_space.shape[0]]
	dist.start_local_cluster_multiprocessing(jobs, 'explorer', 'learner')
