import os

import torch
import torch.distributed as dist

backend = None


def get_comm_backend():
	global backend
	if backend is None:
		backend = _get_comm_backend()
	return backend


def _get_comm_backend():
	return TorchBackend()


class TorchBackend(object):
	def size(self):
		return dist.get_world_size()

	def rank(self):
		return dist.get_rank()

	def local_rank(self):
		try:
			return os.environ['LOCAL_RANK']
		except:
			raise RuntimeError('LOCAL_RANK must be set in the environment'
				'when using torch.distributed')

	def barrier(self):
		dist.barrier()

	def allreduce(self, tensor, async_op=True, average=False):
		# the result are averaged, not summed
		if average:
			operator = torch.distributed.ReduceOp.AVG
			average = False
		else:
			operator = torch.distributed.ReduceOp.SUM
		handle = dist.all_reduce(tensor, async_op=async_op, op=operator)

		if async_op == False:
			return
		return (handle, tensor)


	def reduce(self, tensor, dst, async_op=True, average=False):
		# the result are averaged, not summed
		if average:
			operator = torch.distributed.ReduceOp.AVG
			average = False
		else:
			operator = torch.distributed.ReduceOp.SUM
		handle = dist.reduce(tensor, dst=dst, async_op=async_op, op=operator)

		if async_op == False:
			return
		return (handle, tensor)

	def allgather(self, tensor, async_op=True):
		tensor_list = [torch.zeros_like(tensor) for _ in range(self.size())]
		handle = dist.all_gather(tensor_list, tensor, async_op=async_op)

		if async_op == False:
			return tensor_list
		return (handle, tensor_list)

	def broadcast(self, tensor, src, async_op=True):
		return dist.broadcast(tensor.contiguous(), src=src, async_op=async_op)

	def sync(self, handles):
		if isinstance(handles, list):
			if len(handles) == 0:
				return
			if isinstance(handles[0], tuple):
				for handle, tensor in handles:
					self.wait(handle)
			else: # async broadcast
				for handle in handles:
					self.wait(handle)
		else:
			if isinstance(handles, tuple):
				handle, tensor = handles
				self.wait(handle)
				if isinstance(tensor, list): # async allgather
					pass
				else: # async allreduce
					tensor /= self.size()
			else: # async broadcast
				self.wait(handles)

	def wait(self, handle):
		if handle:
			handle.wait()
