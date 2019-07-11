from multiprocessing import Process
from typing import Callable, List, Any


class Job:
	def __init__(self, name: str, target: Callable, hosts: List[str]):
		self.name = name
		self.target = target
		self.hosts = hosts
		self.handles = [Process(target=target, args=(task_index, self, host))
						for task_index, host in enumerate(hosts)]

	def each_handle(self, operation: Callable[[Process], Any]):
		return [operation(handle) for handle in self.handles]


class Jobs:
	def __init__(self, *jobs: Job):
		self.jobs = {job.name: job for job in jobs}

	def add(self, job: Job):
		self.jobs[job.name] = job

	def get_cluster_dict(self):
		return {job.name: job.hosts for job in self}

	def get_handles(self):
		return [job_handle for job in self.jobs.values() for job_handle in job.handles]

	def each_handle(self, operation: Callable):
		return [operation(handle) for handle in self.get_handles()]

	def __iter__(self):
		return iter(self.jobs.values())

	def __getitem__(self, job_name: str) -> Job:
		return self.jobs[job_name]


class StandbyTarget:
	def __init__(self, jobs: Jobs, setup: Callable, run: Callable = None):
		self.jobs = jobs
		self.setup = setup
		self.run = run

	def __call__(self, task_index, job, _):
		from distributed.session import SimpleDistributedSession

		self.setup()
		with SimpleDistributedSession(self.jobs)(job.name, task_index) as sess:
			if self.run is not None:
				self.run(sess)
