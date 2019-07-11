from typing import TypeVar, Generic, Callable, List, Union

import numpy

ACTION = TypeVar('ACTION')


class Agent(Generic[ACTION]):
	def act(self, ob) -> ACTION:
		raise NotImplementedError

	def save(self, path: str):
		raise NotImplementedError

	def load(self, path: str):
		raise NotImplementedError

	def __str__(self):
		return super().__str__()


class AgentBatch(Generic[ACTION]):
	def __init__(self, agents: Union[List[Agent], Callable[[int], Agent]], batch_size: int):
		self.batch_size = batch_size
		self.agents = [agents(i) for i in range(batch_size)] if callable(agents) else agents

	def act(self, obs) -> ACTION:
		return [agent.act(ob) for agent, ob in zip(self.agents, obs)]

	def save(self, paths: List[str]):
		return [agent.save(path) for agent, path in zip(self.agents, paths)]

	def load(self, paths: List[str]):
		return [agent.load(path) for agent, path in zip(self.agents, paths)]

	def __str__(self):
		return str([agent.__str__() for agent, path in self.agents])


class KerasAgent(Agent):
	def __init__(self, model):
		self.model = model

	def act(self, ob):
		return self.model.predict_on_batch(numpy.expand_dims(ob, 0))[0]
