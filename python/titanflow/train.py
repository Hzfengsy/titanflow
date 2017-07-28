from .ops import *
from ._session import *

class GradientDescentOptimizer(object):
	def __init__(self, learning_rate, name = "GradientDescent"):
		self.learning_rate = learning_rate
		self.name = name
	
	def minimize(self, loss, var_list = None, name = None):
		if var_list == None:
			vars = all_var
		elif isinstance(var_list, list):
			vars = var_list
		else:
			vars = [var_list]
		grads = gradients(loss, vars)
		assign_list = []
		for i in range(len(grads)):
			assign_list.append(assign(vars[i], vars[i] - self.learning_rate * grads[i]))
		return runner(assign_list)
