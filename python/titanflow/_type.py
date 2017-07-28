import numpy as np

class type(object):
	def exchange(self, input):
		raise NotImplementedError


class float32(type):
	def exchange(self, input):
		return input.astype(np.float32)

